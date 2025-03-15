library(reshape2)
library(ggplot2)
library(gridExtra)
library(moments)
library(psych)
library(tidyverse)
library(factoextra)
library(Rtsne)
library(umap)
library(tourr)
library(gifski)
library(grid)
library(cowplot)


df <- read.csv("airlines_delay.csv")

categorical_columns <- c("Airline", "AirportFrom",
                         "AirportTo", "DayOfWeek", "Class")

numerical_columns <- c("Flight", "Time", "Length")

create_boxplots <- function(mean, median, mode, column, df) {
  mean <- mean(df[[column]])
  median <- median(df[[column]])
  mode <- as.numeric(names(sort(table(df[[column]]), decreasing = TRUE)[1]))

  return(ggplot(df, aes(y = .data[[column]])) +
           geom_boxplot() +
           ggtitle(paste("Boxplot of", column, "\n",
                         "Mean:", mean, "Median:", median, "Mode:", mode)))
}

create_histograms <- function(column_name, target_column_name, df) {
  column_sym <- sym(column_name)
  target_sym <- sym(target_column_name)

  df[[target_column_name]] <- factor(df[[target_column_name]],
                                     levels = c(0, 1),
                                     labels = c("Class 0", "Class 1"))

  return(ggplot(df, aes(x = !!column_sym, fill = !!target_sym)) +
    geom_histogram(position = "stack", bins = 30, alpha = 0.8) +
    scale_fill_manual(values = c("blue", "red")) +
    labs(
      title = paste(column_name, "by", target_column_name),
      x = column_name,
      y = "Count",
      fill = "Class"
    ) +
    theme_minimal())
}

compute_disspersion <- function(column, df) {
  range <- max(df[[column]]) - min(df[[column]])
  iqr <- IQR(df[[column]])
  variange <- var(df[[column]])
  sd <- sd(df[[column]])
  mean_absolue_deviation <- mean(abs(df[[column]] - mean(df[[column]])))
  cv_value <- sd / mean(df[[column]])

  print(paste("Column:", column))
  print(paste("Range:", range))
  print(paste("IQR:", iqr))
  print(paste("Variance:", variange))
  print(paste("Standard Deviation:", sd))
  print(paste("Mean Absolute Deviation:", mean_absolue_deviation))
  print(paste("Coefficient of Variation:", cv_value))
  print("\n")
  return(c(range, iqr, variange, sd, mean_absolue_deviation, cv_value))
}

compute_skewness_kurtosis <- function(column, df) {
  skewness <- skewness(df[[column]])
  kurtosis <- kurtosis(df[[column]])
  print(paste("Column:", column))
  print(paste("Skewness:", skewness))
  print(paste("Kurtosis:", kurtosis))
  return(c(skewness, kurtosis))
}

create_skewness_kurtosis_plot <- function(column, df) {
  skew_value <- skewness(df[[column]])
  kurt_value <- kurtosis(df[[column]])
  numeric_column <- df[[column]]
  return(ggplot(data.frame(x = numeric_column), aes(x = x)) +
           geom_histogram(aes(y = ..density..), bins = 30,
                          fill = "lightblue", color = "black") +
           geom_density(color = "red", linewidth = 1) +
           stat_function(fun = dnorm, args = list(mean = mean(numeric_column),
                                                  sd = sd(numeric_column)),
                         color = "blue", linewidth = 1, linetype = "dashed") +
           labs(title = paste("Distribution with Skewness:",
                              round(skew_value, 2),
                              "and Kurtosis:",
                              round(kurt_value, 2)),
                x = "Value", y = "Density") +
           theme_minimal())
}

create_q_q_plot <- function(column, df) {
  skew_value <- skewness(df[[column]])
  kurt_value <- kurtosis(df[[column]])
  numeric_column <- df[[column]]
  return(ggplot(data.frame(x = numeric_column), aes(sample = x)) +
           stat_qq() +
           stat_qq_line() +
           labs(title = paste("Q-Q Plot with Skewness:", round(skew_value, 2),
                              "and Kurtosis:", round(kurt_value, 2))) +
           theme_minimal())
}

create_frequency_plot <- function(column, df, top_n = 20) {
  freq_data <- as.data.frame(table(df[[column]]))
  names(freq_data) <- c("Category", "Count")
  freq_data <- freq_data[order(-freq_data$Count), ]

  if (nrow(freq_data) > top_n) {
    others_sum <- sum(freq_data$Count[(top_n + 1):nrow(freq_data)])
    freq_data <- rbind(
      freq_data[1:top_n, ],
      data.frame(Category = "Others", Count = others_sum)
    )
  }

  freq_data$Percentage <- freq_data$Count / sum(freq_data$Count) * 100
  freq_data$Label <- sprintf("%s\n%.1f%% (n=%d)",
                             freq_data$Category,
                             freq_data$Percentage,
                             freq_data$Count)


  return(ggplot(freq_data, aes(x = reorder(Category, Count), y = Count)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
    geom_text(aes(label = sprintf("%.1f%%", Percentage)),
              hjust = -0.1,
              size = 3) +
    coord_flip() +
    labs(
      title = paste("Frequency Distribution of", column),
      subtitle = paste("Showing top", min(top_n, nrow(freq_data)),
                       "categories"),
      x = column,
      y = "Count"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      axis.text = element_text(size = 9),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank()
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))))

}

create_correlation_viz <- function(df, numerical_columns) {
  cor_matrix <- cor(df[numerical_columns])

  cor_long <- reshape2::melt(cor_matrix)

  cor_plot <- ggplot(cor_long, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "red",
      mid = "white",
      high = "blue",
      midpoint = 0,
      limits = c(-1, 1)
    ) +
    geom_text(aes(label = sprintf("%.2f", value)),
              size = 4) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(face = "bold"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank()
    ) +
    labs(
      title = "Correlation Matrix Heatmap",
      x = "",
      y = "",
      fill = "Correlation"
    ) +
    coord_fixed()

  cat("\nCorrelation Analysis:\n")
  cat("=====================\n")
  for(i in 1:(length(numerical_columns) - 1)) {
    for(j in (i + 1):length(numerical_columns)) {
      cor_test <- cor.test(df[[numerical_columns[i]]],
                          df[[numerical_columns[j]]])
      cat(sprintf("\n%s vs %s:\n", 
                  numerical_columns[i], 
                  numerical_columns[j]))
      cat(sprintf("Correlation: %.3f\n", cor_test$estimate))
      cat(sprintf("p-value: %.3e\n", cor_test$p.value))
    }
  }

  return(cor_plot)
}

analyze_categorical_ind <- function(df, categorical_columns) {
  # Initialize results matrix
  n_vars <- length(categorical_columns)
  chi_square_results <- matrix(NA, nrow = n_vars, ncol = n_vars)
  cramer_v_results <- matrix(NA, nrow = n_vars, ncol = n_vars)
  rownames(chi_square_results) <- colnames(chi_square_results) <- categorical_columns
  rownames(cramer_v_results) <- colnames(cramer_v_results) <- categorical_columns

  # Store detailed results
  detailed_results <- list()

  # Function to interpret Cramer's V
  interpret_cramer_v <- function(v) {
    if (v < 0.1) return("Negligible")
    else if (v < 0.2) return("Weak")
    else if (v < 0.3) return("Moderate")
    else return("Strong")
  }

  # Perform chi-square tests
  for(i in 1:(n_vars-1)) {
    for(j in (i+1):n_vars) {
      # Create contingency table
      cont_table <- table(df[[categorical_columns[i]]], 
                         df[[categorical_columns[j]]])

      # Perform chi-square test with simulation for small expected frequencies
      tryCatch({
        chi_test <- chisq.test(cont_table, simulate.p.value = TRUE, B = 2000)

        # Calculate Cramer's V
        n <- sum(cont_table)
        min_dim <- min(dim(cont_table)) - 1
        cramer_v <- sqrt(chi_test$statistic / (n * min_dim))

        # Store results
        chi_square_results[i,j] <- chi_square_results[j,i] <- chi_test$p.value
        cramer_v_results[i,j] <- cramer_v_results[j,i] <- cramer_v

        # Store detailed results
        detailed_results[[paste(categorical_columns[i], categorical_columns[j], sep="_")]] <- list(
          chi_square = chi_test$statistic,
          p_value = chi_test$p.value,
          cramer_v = cramer_v,
          df = chi_test$parameter,
          association = interpret_cramer_v(cramer_v)
        )
      }, error = function(e) {
        cat(sprintf("\nWarning: Could not compute test for %s vs %s: %s\n", 
                   categorical_columns[i], categorical_columns[j], e$message))
      })
    }
  }

  # Convert matrices to long format for plotting
  chi_square_long <- reshape2::melt(chi_square_results)
  cramer_v_long <- reshape2::melt(cramer_v_results)

  # Create p-value heatmap
  p_value_plot <- ggplot(chi_square_long, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "blue",
      mid = "white",
      high = "red",
      midpoint = 0.05,
      limits = c(0, 1),
      na.value = "grey90"
    ) +
    geom_text(aes(label = ifelse(
      !is.na(value), 
      sprintf("%.3f", value),
      ""
    )), size = 3) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(face = "bold"),
      panel.grid = element_blank()
    ) +
    labs(
      title = "Chi-Square Test P-Values",
      x = "",
      y = "",
      fill = "p-value"
    ) +
    coord_fixed()

  # Create Cramer's V heatmap
  cramer_v_plot <- ggplot(cramer_v_long, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient(
      low = "white",
      high = "darkblue",
      limits = c(0, 1),
      na.value = "grey90"
    ) +
    geom_text(aes(label = ifelse(
      !is.na(value), 
      sprintf("%.3f", value),
      ""
    )), size = 3) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(face = "bold"),
      panel.grid = element_blank()
    ) +
    labs(
      title = "Cramer's V Values",
      x = "",
      y = "",
      fill = "Cramer's V"
    ) +
    coord_fixed()

  cat("\nDetailed Independence Test Results:\n")
  cat("================================\n")
  for(pair in names(detailed_results)) {
    cat(sprintf("\n%s:\n", gsub("_", " vs ", pair)))
    cat(sprintf("Chi-square statistic: %.3f\n", detailed_results[[pair]]$chi_square))
    cat(sprintf("Degrees of freedom: %d\n", detailed_results[[pair]]$df))
    cat(sprintf("p-value: %.3e\n", detailed_results[[pair]]$p_value))
    cat(sprintf("Cramer's V: %.3f\n", detailed_results[[pair]]$cramer_v))
    cat(sprintf("Association strength: %s\n", detailed_results[[pair]]$association))
  }

  return(list(
    p_value_plot = p_value_plot, 
    cramer_v_plot = cramer_v_plot,
    chi_square_results = chi_square_results,
    cramer_v_results = cramer_v_results,
    detailed_results = detailed_results
  ))
}


boxplot_list <- list()
histogram_list <- list()
skewness_kurtosis_list <- list()
q_q_list <- list()
for (column in numerical_columns) {
  boxplot_list[[column]] <- create_boxplots(mean, median, mode, column, df)
  histogram_list[[column]] <- create_histograms(column, "Class", df)
  compute_disspersion(column, df)
  skewness_kurtosis_list[[column]] <- create_skewness_kurtosis_plot(column, df)
  q_q_list[[column]] <- create_q_q_plot(column, df)
}
grid.arrange(grobs = boxplot_list, ncol = 2)
grid.arrange(grobs = histogram_list, ncol = 2)
grid.arrange(grobs = skewness_kurtosis_list, ncol = 2)
grid.arrange(grobs = q_q_list, ncol = 2)


freq_plot_list <- list()
for (column in c("Airline", "AirportFrom", "AirportTo", "DayOfWeek", "Class")) {
  freq_plot_list[[column]] <- create_frequency_plot(column, df)
}
grid.arrange(grobs = freq_plot_list, ncol = 1)

results <- analyze_categorical_ind(df, categorical_columns)
grid.arrange(results$p_value_plot, results$cramer_v_plot, ncol = 2)


pbcor_flight <- biserial(df$Flight, df$Class)
pbcor_time <- biserial(df$Time, df$Class)
pbcor_length <- biserial(df$Length, df$Class)

pb_results <- data.frame(
  Variable = c("Flight Number", "Departure Time", "Flight Length"),
  Correlation = c(pbcor_flight[], pbcor_time[,1], pbcor_length[,1])
)

ggplot(pb_results, aes(x = Variable, y = Correlation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(Correlation, 3)), vjust = -0.5) +
  theme_minimal() +
  labs(title = "Point-Biserial Correlation with Flight Delay",
       x = "Variable",
       y = "Correlation Coefficient")


airline_delay_rates <- df %>%
  group_by(Airline) %>%
  summarize(
    total_flights = n(),
    delayed_flights = sum(Class == 1),
    delay_rate = delayed_flights / total_flights
  ) %>%
  arrange(desc(delay_rate))

dow_delay_rates <- df %>%
  group_by(DayOfWeek) %>%
  summarize(
    total_flights = n(),
    delayed_flights = sum(Class == 1),
    delay_rate = delayed_flights / total_flights
  ) %>%
  arrange(DayOfWeek)

ggplot(airline_delay_rates, aes(x = reorder(Airline, -delay_rate), y = delay_rate)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  geom_text(aes(label = sprintf("%.1f%%", delay_rate * 100)), vjust = -0.5, size = 3) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Flight Delay Rates by Airline",
       x = "Airline",
       y = "Delay Rate")

ggplot(dow_delay_rates, aes(x = factor(DayOfWeek), y = delay_rate)) +
  geom_bar(stat = "identity", fill = "lightgreen") +
  geom_text(aes(label = sprintf("%.1f%%", delay_rate * 100)), vjust = -0.5, size = 3) +
  theme_minimal() +
  scale_x_discrete(labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")) +
  labs(title = "Flight Delay Rates by Day of Week",
       x = "Day of Week",
       y = "Delay Rate")

airline_anova <- aov(as.numeric(Class) - 1 ~ Airline, data = df)
anova_airline_summary <- summary(airline_anova)
print("ANOVA Results - Airline Effect on Delays:")
print(anova_airline_summary)

dow_anova <- aov(as.numeric(Class) - 1 ~ factor(DayOfWeek), data = df)
anova_dow_summary <- summary(dow_anova)
print("ANOVA Results - Day of Week Effect on Delays:")
print(anova_dow_summary)


num_vars <- df %>% select(Flight, Time, Length)
num_vars_scaled <- scale(num_vars)

pca_result <- prcomp(num_vars_scaled, center = TRUE, scale. = TRUE)
summary(pca_result)

fviz_eig(pca_result, addlabels = TRUE)

fviz_pca_var(pca_result, col.var = "contrib",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE)

df$Class <- factor(df$Class,
                   levels = c(0, 1),
                   labels = c("On Time", "Delayed"))

fviz_pca_ind(pca_result,
             geom.ind = "point",
             col.ind = df$Class,
             palette = c("#00AFBB", "#FC4E07"),
             addEllipses = TRUE,
             legend.title = "Delay Status")

fviz_pca_biplot(pca_result,
                col.ind = df$Class,
                col.var = "black",
                label = "var",
                repel = TRUE,
                legend.title = "Delay Status",
                title = "PCA Biplot - Airlines Delay Data")

set.seed(42)
sample_size <- min(5000, nrow(df))
sample_idx <- sample(seq_len(nrow(df)), sample_size)
airlines_sample <- df[sample_idx, ]

tsne_data <- as.matrix(airlines_sample %>% select(Flight, Time, Length))

tsne_result <- Rtsne(tsne_data,
                     dims = 2,
                     perplexity = 30,
                     verbose = TRUE,
                     max_iter = 1000,
                     check_duplicates = FALSE)

tsne_df <- data.frame(
  x = tsne_result$Y[, 1],
  y = tsne_result$Y[, 2],
  delay_status = airlines_sample$Class
)



ggplot(tsne_df, aes(x = x, y = y, color = delay_status)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = c("On Time" = "blue", "Delayed" = "red")) +
  theme_minimal() +
  theme(legend.position = "right") +
  labs(title = "t-SNE Visualization of Airline Delay Data",
       subtitle = "Dimensionality reduction to 2D space",
       color = "Delay Status",
       x = "t-SNE Dimension 1",
       y = "t-SNE Dimension 2")



umap_result <- umap(tsne_data)


umap_df <- data.frame(
  x = umap_result$layout[,1],
  y = umap_result$layout[,2],
  delay_status = airlines_sample$Class
)

ggplot(umap_df, aes(x = x, y = y, color = delay_status)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_manual(values = c("On Time" = "green", "Delayed" = "purple")) +
  theme_minimal() +
  theme(legend.position = "right") +
  labs(title = "UMAP Visualization of Airline Delay Data",
       subtitle = "Preserving both local and global structure",
       color = "Delay Status",
       x = "UMAP Dimension 1",
       y = "UMAP Dimension 2")


tour_path <- guided_tour(cmass())

if (interactive()) {
  animate_xy(num_vars_scaled,
             tour_path = tour_path,
             col = as.numeric(df$Class))
}