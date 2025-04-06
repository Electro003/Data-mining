import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    roc_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    StackingClassifier
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print("\nSample data:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nClass distribution:")
    print(df['Class'].value_counts())
    return df


def get_smoothing_factor(df, column, base_factor=0.01):
    """
    Calculate an appropriate smoothing factor based on data characteristics

    Parameters:
    df: DataFrame containing the data
    column: Column name to calculate smoothing for
    base_factor: Base percentage of total rows to use (default 1%)

    Returns:
    int: Recommended smoothing factor
    """
    total_rows = len(df)
    n_categories = df[column].nunique()
    avg_samples = total_rows / n_categories

    # Start with base_factor of total rows
    smoothing = int(total_rows * base_factor)

    # Adjust based on average samples per category
    if avg_samples < 10:
        # For very sparse categories, increase smoothing
        smoothing = max(smoothing, int(total_rows * 0.05))
    elif avg_samples > 1000:
        # For very common categories, reduce smoothing
        smoothing = min(smoothing, int(total_rows * 0.005))

    return smoothing


def apply_smoothed_target_encoding(df, column, target='Class', min_samples=10, global_mean=None):
    # Make a copy
    result_df = df.copy()

    # If global mean isn't provided, calculate it
    if global_mean is None:
        global_mean = df[target].mean()

    # Group by the column and calculate stats
    aggregates = df.groupby(column)[target].agg(['mean', 'count'])

    # Calculate smoothed means
    smoothed_means = (aggregates['count'] * aggregates['mean'] + min_samples * global_mean) / (
                aggregates['count'] + min_samples)

    # Map smoothed means back to the dataframe
    result_df[f'{column}_smooth'] = result_df[column].map(smoothed_means)

    return result_df


def add_time_features(df, time_column='Time'):
    """Add time-based features from the Time column

    Parameters:
    df (DataFrame): Input dataframe
    time_column (str): Column name containing time values

    Returns:
    DataFrame: DataFrame with added time features
    """
    # Make a copy of the dataframe
    data = df.copy()

    # Extract hour from time (integer division by 100)
    data['Hour'] = (data[time_column] // 100).astype(int)

    # Extract minutes (modulo 100)
    data['Minute'] = (data[time_column] % 100).astype(int)

    # Fix any potential time format issues
    # If minutes >= 60, add to hours and correct minutes
    hours_to_add = data['Minute'] // 60
    data['Hour'] = data['Hour'] + hours_to_add
    data['Minute'] = data['Minute'] % 60

    # Make sure hours are within 0-23 range
    data['Hour'] = data['Hour'] % 24

    # Create time of day categories
    data['TimeOfDay'] = pd.cut(
        data['Hour'],
        bins=[0, 5, 12, 17, 21, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Night'],
        ordered=False
    )
    # Fix night category (0-5 and 21-24 are both night)
    data.loc[data['Hour'] >= 21, 'TimeOfDay'] = 'Night'

    # Create rush hour flag (typical rush hours at airports)
    data['IsRushHour'] = (
            ((data['Hour'] >= 7) & (data['Hour'] <= 9)) |  # Morning rush
            ((data['Hour'] >= 16) & (data['Hour'] <= 18))  # Evening rush
    ).astype(int)

    # Create weekend flag
    data['IsWeekend'] = (data['DayOfWeek'] >= 6).astype(int)

    # Create improved time representation (decimal hours)
    data['TimeDecimal'] = data['Hour'] + (data['Minute'] / 60)

    return data


def create_interaction_features(df):
    """Create interaction features between important predictors

    Parameters:
    df (DataFrame): Input dataframe with smoothed features

    Returns:
    DataFrame: DataFrame with added interaction features
    """
    # Make a copy of the dataframe
    data = df.copy()

    # Create interactions between features
    if 'Airline_smooth' in data.columns and 'Route_smooth' in data.columns:
        data['Airline_Route'] = data['Airline_smooth'] * data['Route_smooth']

    if 'Flight_smooth' in data.columns and 'Length' in data.columns:
        data['Flight_Length'] = data['Flight_smooth'] * data['Length'] / 100  # Scale for numerical stability

    if 'TimeDecimal' in data.columns and 'Length' in data.columns:
        data['Time_Length'] = data['TimeDecimal'] * data['Length'] / 100  # Scale for numerical stability

    return data


def preprocess_and_select_features(df, n_features=None, use_routes=True, use_time_features=True, use_interactions=True):
    """
    Preprocess data with advanced feature engineering and select important features

    Parameters:
    df (DataFrame): Input dataframe
    n_features (int): Number of features to select
    use_routes (bool): Whether to use route-based features
    use_time_features (bool): Whether to add time-based features
    use_interactions (bool): Whether to add interaction features

    Returns:
    Various: Selected features and related information
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier

    data = df.copy()

    # Handle missing values
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype == 'object':
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)

    # Calculate global delay rate once (for consistency across encodings)
    global_delay_rate = data['Class'].mean()
    print(f"Global delay rate: {global_delay_rate:.4f}")

    # Apply target encoding to Flight ID
    flight_smoothing = get_smoothing_factor(data, 'Flight')
    data = apply_smoothed_target_encoding(data, column='Flight', target='Class', min_samples=flight_smoothing,
                                          global_mean=global_delay_rate)

    # Basic target encoding for Airline
    airline_smoothing = get_smoothing_factor(data, 'Airline')
    data = apply_smoothed_target_encoding(data, column='Airline', target='Class', min_samples=airline_smoothing,
                                          global_mean=global_delay_rate)

    # Target encoding for origin and destination airports
    airport_from_smoothing = get_smoothing_factor(data, 'AirportFrom')
    airport_to_smoothing = get_smoothing_factor(data, 'AirportTo')

    data = apply_smoothed_target_encoding(data, column='AirportFrom', target='Class',
                                          min_samples=airport_from_smoothing, global_mean=global_delay_rate)
    data = apply_smoothed_target_encoding(data, column='AirportTo', target='Class',
                                          min_samples=airport_to_smoothing, global_mean=global_delay_rate)

    # Apply route-based grouping
    if use_routes:
        print("\n=== Implementing Route-Based Grouping ===")

        # Create route column
        data['Route'] = data['AirportFrom'] + '_' + data['AirportTo']

        # Calculate route statistics
        route_stats = data.groupby('Route').agg({
            'Class': ['mean', 'count'],
            'Length': 'mean',
            'Time': 'mean'
        })
        route_stats.columns = ['DelayRate', 'Count', 'AvgLength', 'AvgTime']
        route_stats = route_stats.reset_index()

        threshold = route_stats['Count'].quantile(0.75)
        common_routes = route_stats[route_stats['Count'] >= threshold]['Route'].tolist()

        print(f"Identified {len(common_routes)} common routes out of {len(route_stats)} total routes")

        data['IsCommonRoute'] = data['Route'].apply(lambda x: 1 if x in common_routes else 0)

        route_smoothing = get_smoothing_factor(data, 'Route')
        data = apply_smoothed_target_encoding(data, column='Route', target='Class',
                                              min_samples=route_smoothing, global_mean=global_delay_rate)

        common_route_stats = route_stats[route_stats['Route'].isin(common_routes)]

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        top_delay_routes = common_route_stats.nlargest(10, 'DelayRate')
        sns.barplot(x='DelayRate', y='Route', data=top_delay_routes)
        plt.title('Top 10 Routes with Highest Delay Rates')
        plt.xlabel('Delay Rate')
        plt.ylabel('Route')

        plt.subplot(1, 2, 2)
        bottom_delay_routes = common_route_stats.nsmallest(10, 'DelayRate')
        sns.barplot(x='DelayRate', y='Route', data=bottom_delay_routes)
        plt.title('Top 10 Routes with Lowest Delay Rates')
        plt.xlabel('Delay Rate')
        plt.ylabel('Route')

        plt.tight_layout()
        plt.show()

    # Add time-based features
    if use_time_features:
        print("\n=== Adding Time-Based Features ===")
        data = add_time_features(data)

        # Visualize delay rates by time of day
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        tod_delay = data.groupby('TimeOfDay')['Class'].mean().reset_index()
        sns.barplot(x='TimeOfDay', y='Class', data=tod_delay, order=['Morning', 'Afternoon', 'Evening', 'Night'])
        plt.title('Delay Rate by Time of Day')
        plt.xlabel('Time of Day')
        plt.ylabel('Delay Rate')

        plt.subplot(1, 2, 2)
        hour_delay = data.groupby('Hour')['Class'].mean().reset_index()
        sns.lineplot(x='Hour', y='Class', data=hour_delay, marker='o')
        plt.title('Delay Rate by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Delay Rate')
        plt.xticks(range(0, 24, 2))

        plt.tight_layout()
        plt.show()

    # Create interaction features
    if use_interactions:
        print("\n=== Adding Interaction Features ===")
        data = create_interaction_features(data)

    # Define features for model
    numeric_features = ['Time', 'Length', 'Flight_smooth', 'Airline_smooth',
                        'AirportFrom_smooth', 'AirportTo_smooth']
    categorical_features = ['DayOfWeek']

    # Add route features if used
    if use_routes:
        numeric_features.extend(['Route_smooth'])
        categorical_features.extend(['IsCommonRoute'])

    # Add time features if used
    if use_time_features:
        numeric_features.extend(['TimeDecimal', 'Hour'])
        categorical_features.extend(['TimeOfDay', 'IsRushHour', 'IsWeekend'])

    # Add interaction features if used
    if use_interactions and use_time_features:
        if 'Airline_Route' in data.columns:
            numeric_features.append('Airline_Route')
        if 'Flight_Length' in data.columns:
            numeric_features.append('Flight_Length')
        if 'Time_Length' in data.columns:
            numeric_features.append('Time_Length')

    # Create feature dataframe
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)

    # Get feature names after preprocessing
    num_feature_names = numeric_features

    # Get categorical feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)

    all_feature_names = np.concatenate([num_feature_names, cat_feature_names])

    # Scale features
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_processed)

    print("\n=== Using Tree-based Feature Selection ===")
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        threshold='median' if n_features is None else 'mean',
        max_features=n_features,
    )
    selector.fit(X_scaled, y)
    selected_indices = selector.get_support()

    # Get feature importances
    importances = selector.estimator_.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': importances
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Most Important Features (Tree-based)')
    plt.tight_layout()
    plt.show()

    X_selected = X_scaled[:, selected_indices]
    selected_feature_names = all_feature_names[selected_indices]

    print(f"Selected {len(selected_feature_names)} features out of {len(all_feature_names)}")
    print("Selected features:")
    for feature in selected_feature_names:
        print(f"- {feature}")

    return X_selected, y, selected_feature_names, preprocessor, selected_indices


# 3. Model evaluation with cross-validation
def evaluate_model(model, X, y, cv=5, balance=False):
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # For storing metrics
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    # Cross-validation loop
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Apply class balancing if requested
        if balance:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except:
            # If model doesn't support predict_proba
            y_prob = model.decision_function(X_test) if hasattr(model, 'decision_function') else y_pred

        # Collect predictions
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

    # Calculate metrics
    conf_matrix = confusion_matrix(y_true_all, y_pred_all)
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, average=None)
    recall = recall_score(y_true_all, y_pred_all, average=None)
    f1 = f1_score(y_true_all, y_pred_all, average=None)
    f1_global = f1_score(y_true_all, y_pred_all, average='weighted')

    # Calculate AUC if possible
    try:
        auc = roc_auc_score(y_true_all, y_prob_all)
        fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)
    except:
        auc = None
        fpr, tpr = None, None

    results = {
        'confusion_matrix': conf_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f1_global': f1_global,
        'auc': auc,
        'roc_curve': (fpr, tpr),
        'y_true': y_true_all,
        'y_pred': y_pred_all,
        'y_prob': y_prob_all
    }

    return results


# 4. Report results
def report_results(model_name, results):
    print(f"\n=== {model_name} Results ===")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    print(f"Accuracy: {results['accuracy']:.4f}")

    # For binary classification
    print(f"Precision - Class 0: {results['precision'][0]:.4f}, Class 1: {results['precision'][1]:.4f}")
    print(f"Recall - Class 0: {results['recall'][0]:.4f}, Class 1: {results['recall'][1]:.4f}")
    print(f"F1 Score - Class 0: {results['f1_score'][0]:.4f}, Class 1: {results['f1_score'][1]:.4f}")
    print(f"Global F1 Score: {results['f1_global']:.4f}")

    if results['auc'] is not None:
        print(f"AUC: {results['auc']:.4f}")

    # Plot ROC curve if available
    if results['roc_curve'][0] is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(results['roc_curve'][0], results['roc_curve'][1],
                 lw=2, label=f'{model_name} (AUC = {results["auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

    return


# 5.3 Naive Bayes
def optimize_naive_bayes(X, y, cv=5):
    print("\n=== Naive Bayes Optimization ===")

    param_grid = {
        'var_smoothing': np.logspace(-9, -5, 5)  # Default is 1e-9
    }

    grid_search = GridSearchCV(
        GaussianNB(),
        param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1
    )
    grid_search.fit(X, y)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    best_nb = GaussianNB(**grid_search.best_params_)

    best_nb.fit(X, y)

    return best_nb


def main():
    df = load_data('airlines_delay.csv')

    n_features = 8

    X_selected, y, selected_features, preprocessor, selected_indices = preprocess_and_select_features(
        df, n_features=n_features)

    print(f"\nSelected {len(selected_features)} features for model training")

    best_models = {}

    # Naive Bayes
    best_nb = optimize_naive_bayes(X_selected, y)
    best_models['nb'] = best_nb
    nb_results = evaluate_model(best_nb, X_selected, y)
    report_results("Naive Bayes", nb_results)




if __name__ == "__main__":
    # Load data
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your 'airlines_delay.csv' file is in the correct location and format.")