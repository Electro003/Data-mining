if(!require(arules)) install.packages("arules")
if(!require(arulesViz)) install.packages("arulesViz")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(gridExtra)) install.packages("gridExtra")

library(arules)
library(arulesViz)
library(ggplot2)
library(gridExtra)

# Load the data
df <- read.csv("airlines_delay1.csv")
df$Flight <- NULL

# 1. ORIGINAL DISCRETIZATION METHOD
df_orig <- df
df_orig$Time <- discretize(df_orig$Time, method = "interval", breaks = 4,
                           labels = c("Early", "Midday", "Evening", "Night"))
df_orig$Length <- discretize(df_orig$Length, method = "cluster", breaks = 3,
                             labels = c("Short", "Medium", "Long"))
df_orig$DayOfWeek <- factor(df_orig$DayOfWeek, levels = 1:7,
                            labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
df_orig$Class <- factor(df_orig$Class, levels = c(0, 1), labels = c("NotDelayed", "Delayed"))

# 2. EQUAL FREQUENCY DISCRETIZATION
df_equal_freq <- df
df_equal_freq$Time <- discretize(df_equal_freq$Time, method = "frequency", breaks = 4,
                                 labels = c("Q1_Time", "Q2_Time", "Q3_Time", "Q4_Time"))
df_equal_freq$Length <- discretize(df_equal_freq$Length, method = "frequency", breaks = 3,
                                   labels = c("Short", "Medium", "Long"))
df_equal_freq$DayOfWeek <- factor(df_equal_freq$DayOfWeek, levels = 1:7,
                                  labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
df_equal_freq$Class <- factor(df_equal_freq$Class, levels = c(0, 1), labels = c("NotDelayed", "Delayed"))

# 3. K-MEANS DISCRETIZATION
df_kmeans <- df
df_kmeans$Time <- discretize(df_kmeans$Time, method = "cluster", breaks = 4,
                             labels = c("Time_Clust1", "Time_Clust2", "Time_Clust3", "Time_Clust4"))
df_kmeans$Length <- discretize(df_kmeans$Length, method = "cluster", breaks = 4,
                               labels = c("Length_Clust1", "Length_Clust2", "Length_Clust3", "Length_Clust4"))
df_kmeans$DayOfWeek <- factor(df_kmeans$DayOfWeek, levels = 1:7,
                              labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
df_kmeans$Class <- factor(df_kmeans$Class, levels = c(0, 1), labels = c("NotDelayed", "Delayed"))

# 4. FIXED THRESHOLD DISCRETIZATION
df_fixed <- df
df_fixed$Time <- cut(df_fixed$Time, breaks = c(-Inf, 360, 720, 1080, Inf), 
                     labels = c("Morning_0-6", "Midday_6-12", "Afternoon_12-18", "Evening_18-24"))
df_fixed$Length <- cut(df_fixed$Length, breaks = c(-Inf, 60, 180, 300, Inf), 
                       labels = c("VeryShort", "Short", "Medium", "Long"))
df_fixed$DayOfWeek <- factor(df_fixed$DayOfWeek, levels = 1:7,
                             labels = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
df_fixed$Class <- factor(df_fixed$Class, levels = c(0, 1), labels = c("NotDelayed", "Delayed"))


#Bug fix

df_orig[] <- lapply(df_orig, as.factor)
df_equal_freq[] <- lapply(df_equal_freq, as.factor)
df_kmeans[] <- lapply(df_kmeans, as.factor)
df_fixed[] <- lapply(df_fixed, as.factor)


# Convert to transactions
trans_orig <- as(df_orig, "transactions")
trans_equal_freq <- as(df_equal_freq, "transactions")
trans_kmeans <- as(df_kmeans, "transactions")
trans_fixed <- as(df_fixed, "transactions")

# Parameters
supp <- 0.0001
conf <- 0.85

run_apriori <- function(trans, name) {
  cat("\n==== APRIORI FOR", name, "====\n")
  rules <- apriori(trans, parameter = list(supp = supp, conf = conf),
                   appearance = list(rhs = c("Class=Delayed", "Class=NotDelayed"), default = "lhs"))
  cat("Generated", length(rules), "rules.\n")
  top_rules <- head(sort(rules, by = "confidence"), 5)
  inspect(top_rules)
  return(rules)
}

rules_orig <- run_apriori(trans_orig, "ORIGINAL DISCRETIZATION")
rules_equal_freq <- run_apriori(trans_equal_freq, "EQUAL FREQUENCY DISCRETIZATION")
rules_kmeans <- run_apriori(trans_kmeans, "K-MEANS DISCRETIZATION")
rules_fixed <- run_apriori(trans_fixed, "FIXED THRESHOLD DISCRETIZATION")

get_max_confidence <- function(rules) {
  if (length(rules) > 0) max(quality(rules)$confidence) else 0
}

# Compare
confidences <- c(
  Original = get_max_confidence(rules_orig),
  EqualFreq = get_max_confidence(rules_equal_freq),
  KMeans = get_max_confidence(rules_kmeans),
  Fixed = get_max_confidence(rules_fixed)
)

best_name <- names(which.max(confidences))
best_rules <- switch(best_name,
                     Original = rules_orig,
                     EqualFreq = rules_equal_freq,
                     KMeans = rules_kmeans,
                     Fixed = rules_fixed)
best_confidence <- max(confidences)

cat("\nBest method:", best_name, "with max confidence =", best_confidence, "\n")

top_best_rules <- head(sort(best_rules, by = "lift"), 150)
inspect(top_best_rules)

# Plots
dir.create("association_rule_plots", showWarnings = FALSE)
png("association_rule_plots/support_confidence_scatter.png", width = 1200, height = 1000, res = 150)
plot(top_best_rules, method = "scatter", measure = c("support", "confidence"), shading = "lift")
dev.off()

png("association_rule_plots/rules_network.png", width = 1500, height = 1200, res = 150)
plot(top_best_rules, method = "graph")
dev.off()

png("association_rule_plots/grouped_matrix.png", width = 1200, height = 1000, res = 150)
plot(top_best_rules, method = "grouped")
dev.off()

png("association_rule_plots/parallel_coordinates.png", width = 1500, height = 1000, res = 150)
plot(top_best_rules, method = "paracoord")
dev.off()

# Interactive HTML
if(require(htmlwidgets)) {
  library(htmlwidgets)
  saveWidget(plot(top_best_rules, method = "graph", engine = "htmlwidget"),
             "association_rule_plots/interactive_rules_network.html")
}

# Export rules
write.csv(as(top_best_rules, "data.frame"), "best_association_rules.csv", row.names = FALSE)

delayed <- subset(top_best_rules, rhs %in% "Class=Delayed")
not_delayed <- subset(top_best_rules, rhs %in% "Class=NotDelayed")

write.csv(as(head(sort(delayed, by = "confidence"), decreasing = TRUE), "data.frame"), "best_associations_delayed.csv", row.names = FALSE)
write.csv(as(head(sort(not_delayed, by = "confidence"), decreasing = TRUE), "data.frame"), "best_associations_not_delayed.csv", row.names = FALSE)

cat("\nApriori analysis complete. Results saved.\n")
