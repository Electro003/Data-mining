import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


def get_smoothing_factor(df, column, base_factor=0.01):
    """Calculate an appropriate smoothing factor based on data characteristics"""
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
    """Apply smoothed target encoding to a categorical column"""
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
    """Add time-based features from the Time column"""
    # Make a copy of the dataframe
    data = df.copy()

    # Extract hour from time (integer division by 100)
    data['Hour'] = (data[time_column] // 100).astype(int)

    # Extract minutes (modulo 100)
    data['Minute'] = (data[time_column] % 100).astype(int)

    # Fix any potential time format issues
    hours_to_add = data['Minute'] // 60
    data['Hour'] = data['Hour'] + hours_to_add
    data['Minute'] = data['Minute'] % 60
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

    # Create rush hour flag and weekend flag
    data['IsRushHour'] = (
            ((data['Hour'] >= 7) & (data['Hour'] <= 9)) |  # Morning rush
            ((data['Hour'] >= 16) & (data['Hour'] <= 18))  # Evening rush
    ).astype(int)

    data['IsWeekend'] = (data['DayOfWeek'] >= 6).astype(int)
    data['TimeDecimal'] = data['Hour'] + (data['Minute'] / 60)

    return data


def create_interaction_features(df):
    """Create interaction features between important predictors"""
    # Make a copy of the dataframe
    data = df.copy()

    # Create interactions between features
    if 'Airline_smooth' in data.columns and 'Route_smooth' in data.columns:
        data['Airline_Route'] = data['Airline_smooth'] * data['Route_smooth']

    if 'Flight_smooth' in data.columns and 'Length' in data.columns:
        data['Flight_Length'] = data['Flight_smooth'] * data['Length'] / 100

    if 'TimeDecimal' in data.columns and 'Length' in data.columns:
        data['Time_Length'] = data['TimeDecimal'] * data['Length'] / 100

    return data


def preprocess_and_select_features(df, n_features=None, use_routes=True, use_time_features=True, use_interactions=True):
    """Preprocess data with feature engineering and select important features"""
    # Handle missing values
    data = df.copy()
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype == 'object':
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)

    # Calculate global delay rate
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
        print("=== Implementing Route-Based Grouping ===")
        data['Route'] = data['AirportFrom'] + '_' + data['AirportTo']
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

    # Add time-based features
    if use_time_features:
        print("=== Adding Time-Based Features ===")
        data = add_time_features(data)

    # Create interaction features
    if use_interactions:
        print("=== Adding Interaction Features ===")
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

    # Create preprocessor - use sparse=False instead of sparse_output=False for older scikit-learn
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)

    # Get feature names
    num_feature_names = numeric_features

    # Get categorical feature names - handle potential differences in scikit-learn versions
    cat_encoder = preprocessor.named_transformers_['cat']
    try:
        # For newer scikit-learn
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    except AttributeError:
        # For older scikit-learn
        cat_feature_names = np.array([f"{col}_{val}" for col in categorical_features
                                      for val in cat_encoder.categories_[categorical_features.index(col)]])

    all_feature_names = np.concatenate([num_feature_names, cat_feature_names])

    # Scale features
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_processed)

    # Feature selection
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        threshold='median' if n_features is None else 'mean',
        max_features=n_features,
    )
    selector.fit(X_scaled, y)
    selected_indices = selector.get_support()

    X_selected = X_scaled[:, selected_indices]
    selected_feature_names = all_feature_names[selected_indices]

    print(f"Selected {len(selected_feature_names)} features out of {len(all_feature_names)}")
    print("Selected features:")
    for feature in selected_feature_names:
        print(f"- {feature}")

    return X_selected, y, selected_feature_names, preprocessor, selected_indices