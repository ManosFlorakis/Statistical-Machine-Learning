###STATISTICAL MACHINE LEARNING ASSIGNMENT II

#Necessary Libraries
library(caret)
library(randomForest)
library(keras)
library(tensorflow)
library(dplyr)

#import data
data<- read.csv(file = "diabetes.csv")
str(data)

#change types of variables ton numeric and factor only
data$Pregnancies <- as.numeric(data$Pregnancies)
data$Glucose <- as.numeric(data$Glucose)
data$BloodPressure <- as.numeric(data$BloodPressure)
data$SkinThickness <- as.numeric(data$SkinThickness)
data$Insulin <- as.numeric(data$Insulin)
data$Age <- as.numeric(data$Age)
data$Outcome = factor(data$Outcome, levels = c(0, 1))

#Normalize data (variables only) and check the results
data[-9]<-scale(data[-9])
str(data)

#View(data) #if need to visually check the whole dataset

##Random Forest

#i) Random Forest for all variables

# Reproducible random sampling
set.seed(123)

# Creating training data as 80% of the dataset
random_sample <- createDataPartition(data$Outcome, p = 0.80, list = F)

# Generating training dataset from the random_sample
training_dataset  <- data[random_sample, ]

# Generating testing dataset from rows which are not included in random_sample
testing_dataset <- data[-random_sample, ]

# NTree --> 500 
RF <- randomForest(Outcome ~ ., data = training_dataset, ntree = 500, cv.fold = 5)

# View the cross-validated forest results.
print(RF)

# Importance of each predictor.
importance(RF, type = 2)

# Plot the cross-validated random forest.
plot(RF)

# Variable importance plot for cross-validated random forest.
varImpPlot(RF)

# Make predictions on the training dataset.
predictions_train <- predict(RF, data = training_dataset)

# Calculate the accuracy.
accuracy_train <- sum(predictions_train == training_dataset$Outcome) / length(training_dataset$Outcome)

# Make predictions on the testing dataset.
predictions_test <- predict(RF, testing_dataset)

# Create a confusion matrix for the testing dataset.
confusion_matrix_test <- confusionMatrix(data = predictions_test, reference = testing_dataset$Outcome)

# Extract and print accuracy for the testing dataset.
accuracy_test <- confusion_matrix_test$overall[1]

# NTree --> 1000 
RF1 <- randomForest(Outcome ~ ., data = training_dataset, ntree = 1000, cv.fold = 5)

# View the cross-validated forest results.
print(RF1)

# Importance of each predictor.
importance(RF1, type = 2)

# Plot the cross-validated random forest.
plot(RF1)

# Variable importance plot for cross-validated random forest.
varImpPlot(RF1)

# Make predictions on the training dataset.
predictions_train1 <- predict(RF1, data = training_dataset)

# Calculate the accuracy.
accuracy_train1 <- sum(predictions_train1 == training_dataset$Outcome) / length(training_dataset$Outcome)

# Make predictions on the testing dataset.
predictions_test1 <- predict(RF1, testing_dataset)

# Create a confusion matrix for the testing dataset.
confusion_matrix_test1 <- confusionMatrix(data = predictions_test1, reference = testing_dataset$Outcome)

# Extract and print accuracy for the testing dataset.
accuracy_test1 <- confusion_matrix_test1$overall[1]

# NTree --> 2000 
RF2 <- randomForest(Outcome ~ ., data = training_dataset, ntree = 2000, cv.fold = 5)

# View the cross-validated forest results.
print(RF2)

# Importance of each predictor.
importance(RF2, type = 2)

# Plot the cross-validated random forest.
plot(RF2)

# Variable importance plot for cross-validated random forest.
varImpPlot(RF2)

# Make predictions on the training dataset.
predictions_train2 <- predict(RF2, data = training_dataset)

# Calculate the accuracy.
accuracy_train2 <- sum(predictions_train2 == training_dataset$Outcome) / length(training_dataset$Outcome)

# Make predictions on the testing dataset.
predictions_test2 <- predict(RF2, testing_dataset)

# Create a confusion matrix for the testing dataset.
confusion_matrix_test2 <- confusionMatrix(data = predictions_test2, reference = testing_dataset$Outcome)

# Extract and print accuracy for the testing dataset.
accuracy_test2 <- confusion_matrix_test2$overall[1]

#Creating a training and testing accuracy dataframe 
data.frame( modelName = c("RF   (NTree=500)","RF1 (NTree=1000)","RF2 (NTree=2000)"),
            AccuracyTrain=c(accuracy_train,accuracy_train1,accuracy_train2),
            AccuracyTest=c(accuracy_test,accuracy_test1,accuracy_test2))

#ii) Random Forest for “Glucose” & “BMI”

#Downsizing the dataset as requested
data<-subset(data,select = c(Glucose,BMI,Outcome))
str(data)

# Reproducible random sampling
set.seed(123)

# Creating training data as 80% of the dataset
random_sample <- createDataPartition(data$Outcome, p = 0.80, list = F)

# Generating training dataset from the random_sample
training_dataset  <- data[random_sample, ]

# Generating testing dataset from rows which are not included in random_sample
testing_dataset <- data[-random_sample, ]

# NTree --> 500 
RF <- randomForest(Outcome ~ ., data = training_dataset, ntree = 500, cv.fold = 5)

# View the cross-validated forest results.
print(RF)

# Importance of each predictor.
importance(RF, type = 2)

# Plot the cross-validated random forest.
plot(RF)

# Variable importance plot for cross-validated random forest.
varImpPlot(RF)

# Make predictions on the training dataset.
predictions_train <- predict(RF, data = training_dataset)

# Calculate the accuracy.
accuracy_train <- sum(predictions_train == training_dataset$Outcome) / length(training_dataset$Outcome)

# Make predictions on the testing dataset.
predictions_test <- predict(RF, testing_dataset)

# Create a confusion matrix for the testing dataset.
confusion_matrix_test <- confusionMatrix(data = predictions_test, reference = testing_dataset$Outcome)

# Extract and print accuracy for the testing dataset.
accuracy_test <- confusion_matrix_test$overall[1]

# NTree --> 1000 
RF1 <- randomForest(Outcome ~ ., data = training_dataset, ntree = 1000, cv.fold = 5)

# View the cross-validated forest results.
print(RF1)

# Importance of each predictor.
importance(RF1, type = 2)

# Plot the cross-validated random forest.
plot(RF1)

# Variable importance plot for cross-validated random forest.
varImpPlot(RF1)

# Make predictions on the training dataset.
predictions_train1 <- predict(RF1, data = training_dataset)

# Calculate the accuracy.
accuracy_train1 <- sum(predictions_train1 == training_dataset$Outcome) / length(training_dataset$Outcome)

# Make predictions on the testing dataset.
predictions_test1 <- predict(RF1, testing_dataset)

# Create a confusion matrix for the testing dataset.
confusion_matrix_test1 <- confusionMatrix(data = predictions_test1, reference = testing_dataset$Outcome)

# Extract and print accuracy for the testing dataset.
accuracy_test1 <- confusion_matrix_test1$overall[1]

# NTree --> 2000 
RF2 <- randomForest(Outcome ~ ., data = training_dataset, ntree = 2000, cv.fold = 5)

# View the cross-validated forest results.
print(RF2)

# Importance of each predictor.
importance(RF2, type = 2)

# Plot the cross-validated random forest.
plot(RF2)

# Variable importance plot for cross-validated random forest.
varImpPlot(RF2)

# Make predictions on the training dataset.
predictions_train2 <- predict(RF2, data = training_dataset)

# Calculate the accuracy.
accuracy_train2 <- sum(predictions_train2 == training_dataset$Outcome) / length(training_dataset$Outcome)

# Make predictions on the testing dataset.
predictions_test2 <- predict(RF2, testing_dataset)

# Create a confusion matrix for the testing dataset.
confusion_matrix_test2 <- confusionMatrix(data = predictions_test2, reference = testing_dataset$Outcome)

# Extract and print accuracy for the testing dataset.
accuracy_test2 <- confusion_matrix_test2$overall[1]

#Creating a training and testing accuracy dataframe 
data.frame( ModelName = c("RF   (NTree=500)","RF1 (NTree=1000)","RF2 (NTree=2000)"),
            AccuracyTrain=c(accuracy_train,accuracy_train1,accuracy_train2),
            AccuracyTest=c(accuracy_test,accuracy_test1,accuracy_test2))







##Feedforward Neural Network

#import data
data<- read.csv(file = "diabetes.csv")

#change types of variables ton numeric and factor only
data$Pregnancies <- as.numeric(data$Pregnancies)
data$Glucose <- as.numeric(data$Glucose)
data$BloodPressure <- as.numeric(data$BloodPressure)
data$SkinThickness <- as.numeric(data$SkinThickness)
data$Insulin <- as.numeric(data$Insulin)
data$Age <- as.numeric(data$Age)
data$Outcome <- as.factor(data$Outcome)

#Normalize data (variables only) and check the results
data[-9]<-scale(data[-9])
str(data)

#View(data) #if need to visually check the whole dataset

#i)FNN for all variables

# Create a reproducible random sampling
set.seed(123)

# Create training data as 80% of the dataset
random_sample <- createDataPartition(data$Outcome, p = 0.8, list = FALSE)
train_data <- data[random_sample, ]
test_data <- data[-random_sample, ]

# Build and train the neural network model
train_and_evaluate_model <- function(neurons, learning_rate) {
  model <- keras_model_sequential() %>%
    layer_dense(units = neurons, activation = "relu", input_shape = c(8)) %>%
    layer_dense(units = neurons, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  # Compile the model
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = "accuracy"
  )
  
  # Train the model
  history <- model %>% fit(
    x = as.matrix(train_data[, 1:8]),
    y = as.numeric(train_data$Outcome) - 1,
    epochs = 300,
    batch_size = 100,
    validation_split = 0.2
  )
  
  # Evaluate the model on the test set
  test_metrics <- model %>% evaluate(
    x = as.matrix(test_data[, 1:8]),
    y = as.numeric(test_data$Outcome) - 1
  )
  
  # Evaluate the model on the training set
  train_metrics <- model %>% evaluate(
    x = as.matrix(train_data[, 1:8]),
    y = as.numeric(train_data$Outcome) - 1
  )
  
  return(c(neurons, learning_rate, train_metrics["accuracy"], test_metrics["accuracy"]))
}

# Matrix for results
results <- matrix(NA, nrow = 6, ncol = 4)
colnames(results) <- c("Neurons", "Learning Rate", "Train Accuracy", "Test Accuracy")

# Different parameters
neurons_list <- c(4, 8, 16)
learning_rates <- c(0.001, 0.01)

# Running the models and extracting results
index <- 1
for (neurons in neurons_list) {
  for (lr in learning_rates) {
    result <- train_and_evaluate_model(neurons, lr)
    results[index, ] <- result
    index <- index + 1
  }
}

# Display a summary table of results
summary_table <- as.data.frame(results)
print(summary_table)


#ii) FNN for “Glucose” & “BMI”

# Build and train the neural network model
train_and_evaluate_model <- function(neurons, learning_rate) {
  model <- keras_model_sequential() %>%
    layer_dense(units = neurons, activation = "relu", input_shape = c(2)) %>%
    layer_dense(units = neurons, activation = "relu") %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = "accuracy"
  )
  
  history <- model %>% fit(
    x = as.matrix(train_data[, c("Glucose", "BMI")]),
    y = as.numeric(train_data$Outcome) - 1,
    epochs = 300,
    batch_size = 100,
    validation_split = 0.2
  )
  
  train_metrics <- model %>% evaluate(
    x = as.matrix(train_data[, c("Glucose", "BMI")]),
    y = as.numeric(train_data$Outcome) - 1
  )
  
  test_metrics <- model %>% evaluate(
    x = as.matrix(test_data[, c("Glucose", "BMI")]),
    y = as.numeric(test_data$Outcome) - 1
  )
  
  return(c(neurons, learning_rate, train_metrics["accuracy"], test_metrics["accuracy"]))
}

# Model parameters
neurons_list <- c(4, 8, 16)
learning_rates <- c(0.001, 0.01)

# Matrix for results
results <- matrix(NA, nrow = length(neurons_list) * length(learning_rates), ncol = 4)
colnames(results) <- c("Neurons", "Learning Rate", "Train Accuracy", "Test Accuracy")

# Running the models and extracting results
index <- 1
for (neurons in neurons_list) {
  for (lr in learning_rates) {
    result <- train_and_evaluate_model(neurons, lr)
    results[index, ] <- result
    index <- index + 1
  }
}

# Summary table of results
summary_table <- as.data.frame(results)
print(summary_table)




