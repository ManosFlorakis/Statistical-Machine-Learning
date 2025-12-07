###Statistical Machine Learning
##Assingment 1

#Neccesary Libraries
library(data.table)
library(caret)
library(leaps)
library(glmnet)
library(plotly)
library(e1071)
library(pROC)

#Exercise 1

#Load Data
data <- fread(file = "carsales.csv")
#First Look
str(data)
psych::describe(data)
names(data)

#Removing the names columns
data1 <- subset(data,select=-c(Car_Name))

#set all columns as numeric and factor
data1$Year <- as.numeric(data1$Year)
data1$Kms_Driven <- as.numeric(data1$Kms_Driven)
data1$Fuel_Type <- as.factor(data1$Fuel_Type)
data1$Seller_Type <- as.factor(data1$Seller_Type)
data1$Transmission <- as.factor(data1$Transmission)
data1$Owner <- as.factor(data1$Owner)
str(data1)

##Linear Regression
# Reproducible random sampling
set.seed(123)

# Creating training data as 75% of the dataset
random_sample <- createDataPartition(data1$Selling_Price, p = 0.75, list = FALSE)

# Generating training dataset from the random_sample
training_dataset  <- data1[random_sample, ]

# Generating testing dataset from rows which are not included in random_sample
testing_dataset <- data1[-random_sample, ]

# Defining training control as cross-validation and value of K equal to 10
train_control <- trainControl(method = "cv",number = 10)


#Training the model by assigning Selling_Price column as target variable and rest other column
# as independent variable
model <- train(Selling_Price ~., data = training_dataset, method = "lm", trControl = train_control)
summary(model)

# Predicting the target variable
predictions <- predict(model, testing_dataset)

# Computing model performance metrics
data.frame( R2 = R2(predictions, testing_dataset$Selling_Price),
            RMSE = RMSE(predictions, testing_dataset$Selling_Price),
            MAE = MAE(predictions, testing_dataset$Selling_Price))

#Variables Fuel_Type, Kms_Driven and Owners aren't in total significant so we will test the above again

data2 <- subset(data1,select=c(Year,Selling_Price,Present_Price,Seller_Type,Transmission))
# Reproducible random sampling
set.seed(123)
random_sample1 <- createDataPartition(data2$Selling_Price, p = 0.75, list = FALSE)
training_dataset2  <- data2[random_sample1, ]
testing_dataset2 <- data2[-random_sample1, ]
#train_control remains the same
model2 <- train(Selling_Price ~., data = training_dataset2, method = "lm", trControl = train_control)
summary(model2)
predictions2 <- predict(model2, testing_dataset2)
data.frame( R2 = R2(predictions2, testing_dataset2$Selling_Price),
            RMSE = RMSE(predictions2, testing_dataset2$Selling_Price),
            MAE = MAE(predictions2, testing_dataset2$Selling_Price))

# Plotting
plot_data1 <- data.frame(Actual = testing_dataset$Selling_Price, Predicted = predictions)
plot_data2 <- data.frame(Actual = testing_dataset2$Selling_Price, Predicted = predictions2)
# Combining data for both models
plot_data_combined <- rbind(cbind(plot_data1, Model = "Model 1"), cbind(plot_data2, Model = "Model 2"))

# Creating a combined scatterplot using facets
ggplot(plot_data_combined, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "Actual Selling_Price", y = "Predicted Selling_Price") +
  ggtitle("Actual vs Predicted Selling Price") +
  facet_wrap(~ Model, scales = "free")  # Faceting by model

#As we can see we can gain the same amount of accuracy with lesser variables so we can conclude
#that the optimal model will have Year,Selling_Price,Present_Price,Seller_Type and Transmission
#as the most significant variables

##Leaps

#Keep only quantitative variables
data_Leaps <- subset(data1,select=c(Year,Selling_Price,Present_Price,Kms_Driven))
str(data_Leaps)

# Use all predictor variables
regfit_full3 <- regsubsets(Selling_Price ~., data = data_Leaps ,nvmax = 3 ,nbest = 1)
reg_summary <- summary(regfit_full3)
reg_summary

#We can see that the Kms_Driven variable don't contribute that much to the model
plot(reg_summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
reg_summary$adjr2
#By the plot seems that addingthe 3rd variable isn't gaining that much for the model
adj_r2_max <-  which.max(reg_summary$adjr2)
#Highest R-square gained with the 2nd model

#Seeing the coefficients for the to highest rated models
print(c(coef(regfit_full3, 2),coef(regfit_full3, 3)))

#Checking our model of choice (2) with plots for R-square and BIC
plot(regfit_full3, scale = "adjr2")
plot(regfit_full3, scale = "bic")



##Ridge

# Reproducible random sampling
set.seed(123)

#Provide only numeric data
data_Ridge <- as.data.frame(sapply(data1, as.numeric))

# Generating training dataset from the random_sample
training_dataset  <- data_Ridge[random_sample, ]
train_x <- as.matrix(training_dataset[,-2])
train_y <- training_dataset$Selling_Price

# Generating testing dataset from rows which are not included in random_sample
testing_dataset <- data_Ridge[-random_sample, ]
test_x <- as.matrix(testing_dataset[,-2])
test_y <- testing_dataset$Selling_Price

#perform k-fold cross-validation to find optimal lambda value
cv_model <- cv.glmnet(x = train_x, y = train_y, alpha = 0,nfolds = 10)
cv_model

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda

#produce plot of test MSE by lambda value
plot(cv_model) 
c(log(cv_model$lambda.min), log(cv_model$lambda.1se))

#produce Ridge trace plot
plot(cv_model$glmnet.fit,xvar = "lambda",label = TRUE)
t(colnames(train_x))

#find coefficients and R2 of best model (lambda min)
best_model <- glmnet(x = train_x, y = train_y, alpha = 0, lambda = best_lambda)
coef(best_model)
#We can see that 'Kms_Driven' have coefficient close to zero
#suggesting lesser influence on the model.

y_pred <- as.numeric( predict(object = best_model,newx = test_x) )
R2(y_pred, test_y)

#find coefficients and R2 of best model (lambda 1se)
best_model_1se <- glmnet(x = train_x, y = train_y, alpha = 0, lambda = cv_model$lambda.1se)
coef(best_model_1se)
#We can see that 'Kms_Driven' have coefficient close to zero
#suggesting lesser influence on the model.
y_pred_1se <- as.numeric( predict(object = best_model_1se,newx = test_x) )
R2(y_pred_1se, test_y)

dataForPlot <- data.frame(test_y,RidgePred_lambdaMin = y_pred, RidgePred_lambda1se = y_pred_1se)
dataForPlot$ID <- as.factor(1:nrow(dataForPlot))
dataForPlotMelted <- reshape2::melt(data = dataForPlot,value.name = "Selling_Price")
ggplot(data = dataForPlotMelted,aes(x=ID,y=Selling_Price,group=variable,col=variable))+geom_line()


##Lasso

# Reproducible random sampling
set.seed(123)

#Train x and y and Test x and y are the same as before

#perform k-fold cross-validation to find optimal lambda value
cv_model_lasso <- cv.glmnet(x = train_x, y = train_y, alpha = 1,nfolds = 10)
cv_model_lasso

#find optimal lambda value that minimizes test MSE
best_lambda_lasso <- cv_model_lasso$lambda.min
best_lambda_lasso

#produce plot of test MSE by lambda value
plot(cv_model_lasso) 
c(log(cv_model_lasso$lambda.min), log(cv_model_lasso$lambda.1se))

#produce Lasso trace plot
plot(cv_model_lasso$glmnet.fit,xvar = "lambda",label = TRUE)
t(colnames(train_x))


#find coefficients and R2 of best lasso model (lambda min)
best_model_lasso <- glmnet(x = train_x, y = train_y, alpha = 1, lambda = best_lambda_lasso)
coef(best_model_lasso)
# 'Kms_Driven', 'Transmission' and 'Owner' were removed or not retained in the model
# due to their coefficients being reduced to zero by the Lasso regularization.
y_pred_lasso <- as.numeric( predict(object = best_model_lasso,newx = test_x) )
R2(y_pred_lasso, test_y)

#find coefficients and R2 of best model (lambda 1se)
best_model_1se_lasso <- glmnet(x = train_x, y = train_y, alpha = 1, lambda = cv_model_lasso$lambda.1se)
coef(best_model_1se_lasso)
# 'Kms_Driven','Fuel_Type', 'Transmission', and 'Owner' were removed or not retained in the model
# due to their coefficients being reduced to zero by the Lasso regularization.
y_pred_1se_lasso <- as.numeric( predict(object = best_model_1se_lasso,newx = test_x) )
R2(y_pred_1se_lasso, test_y)

dataForPlotLasso <- data.frame(test_y,LassoPred_lambdaMin = y_pred_lasso, 
                               LassoPred_lambda1se = y_pred_1se_lasso)
dataForPlotLasso$ID <- as.factor(1:nrow(dataForPlotLasso))
dataForPlotLassoMelted <- reshape2::melt(data = dataForPlotLasso,value.name = "Selling_Price")
ggplot(data = dataForPlotLassoMelted,aes(x=ID,y=Selling_Price,group=variable,col=variable))+geom_line()



## Elastic net 

# Reproducible random sampling
set.seed(123)

#Train x and y and Test x and y are the same as before

alphasToTest <- 0:30/30
models <- list()
glm_models <- list()
results <- data.frame()

for(curID in 1:length(alphasToTest)){
  curAlpha <- alphasToTest[curID]
  cat("Alpha =",curAlpha,"\n")
  models[[curID]] <-
    cv.glmnet(x = train_x, y = train_y, alpha = curAlpha, nfolds = 10,type.measure="mse")
  curModel <- glmnet(x = train_x, y = train_y, alpha = curAlpha,
                     lambda = models[[curID]]$lambda.1se,)
  glm_models[[curID]] <- curModel
  
  predicted <- predict(curModel, newx=test_x)
  
  ## Calculate the Mean Squared Error
  mse <- mean((test_y - predicted)^2)
  
  ## Store the results
  tempDF <- data.frame(alpha=curAlpha, best_lambda.1se=models[[curID]]$lambda.1se, mse_test=mse)
  results <- rbind(results, tempDF)
}

plot(results$alpha, results$mse_test,type = "l")
results <- results[order(results$mse),]

# Best regularized model
best_alpha_elastic_net <- results$alpha[1]
best_lambda_elastic_net <- results$best_lambda.1se[1]
cat("best_alpha_elastic_net =",best_alpha_elastic_net,"\nbest_lambda_elastic_net =",best_lambda_elastic_net,"\n")

best_ID <- as.numeric(rownames(results)[1])
#To see which variable was the least contributing to the model
coef(glm_models[[best_ID]])
#As we can see was the variable 'Kms_Driven"
y_pred_1se_elastic_net <- predict(glm_models[[best_ID]],newx = test_x)

# plot 3d scatter of alpha, lambda and mse
plot_ly(data = results, x=~alpha, y=~best_lambda.1se, z=~mse_test) %>%
  add_markers()

data.frame( modelName = c("OLS","Ridge","Lasso","EL"),
            R2 = c(R2(predictions, test_y), R2(y_pred_1se, test_y), 
                   R2(y_pred_1se_lasso, test_y),R2(y_pred_1se_elastic_net,test_y)),
            RMSE = c( RMSE(predictions, test_y), RMSE(y_pred_1se, test_y), 
                      RMSE(y_pred_1se_lasso, test_y),RMSE(y_pred_1se_elastic_net,test_y)),
            MAE = c( MAE(predictions, test_y), MAE(y_pred_1se, test_y), 
                     MAE(y_pred_1se_lasso, test_y) ,MAE(y_pred_1se_elastic_net,test_y)),
            MSE=c(mean( (test_y - predictions)^2 ),mean( (test_y - y_pred_1se)^2 ),mean( (test_y - y_pred_1se_lasso)^2 ),
                  mean( (test_y - y_pred_1se_elastic_net)^2 ) ))





#Exercise 2

#Load Data
mydata <- fread(file = "diabetes.csv")
#First Look
str(mydata)
psych::describe(mydata)
names(mydata)

#set all columns as numeric and factor
mydata$Pregnancies <- as.numeric(mydata$Pregnancies)
mydata$Glucose <- as.numeric(mydata$Glucose)
mydata$BloodPressure <- as.numeric(mydata$BloodPressure)
mydata$SkinThickness <- as.numeric(mydata$SkinThickness)
mydata$Insulin <- as.numeric(mydata$Insulin)
mydata$Age <- as.numeric(mydata$Age)
mydata$Outcome <- factor(mydata$Outcome, levels = c(0, 1))
str(mydata)

##For all data

# Setting seed to generate a reproducible random sampling
set.seed(123)

# Randomly creating training data as 75% of the dataset
random_sample <- createDataPartition(mydata$Outcome, p = 0.75,list = F)

# Generating training dataset from the random_sample
training_dataset  <- mydata[random_sample,]

# Generating testing dataset from rows which are not included in random_sample
testing_dataset <- mydata[-random_sample, ]

##Logistic Regression

# Fit Logistic Regression model using 5-fold cross-validation
glm_model <- train(Outcome ~ ., data = training_dataset, method = "glm",family = "binomial")

summary(glm_model)

# Predicting the target variable
predictions <- predict(glm_model, testing_dataset)

# Κατασκευή data frame με τις προβλέψεις και τις πραγματικές τιμές Outcome
results <- data.frame(
  Predicted_Outcome = as.factor(predictions),
  Actual_Outcome = testing_dataset$Outcome
)
# Υπολογισμός απόδοσης του μοντέλου
confusionMatrix(
  data = results$Predicted_Outcome, 
  reference = results$Actual_Outcome
)

#Variables SkinThickness, Insulin and Age aren't in total significant so we will test the above again

mydata2 <- subset(mydata,select=c(Pregnancies,Glucose,BloodPressure,BMI,Outcome))
# Reproducible random sampling
set.seed(123)
random_sample1 <- createDataPartition(mydata2$Outcome, p = 0.75, list = FALSE)
training_dataset2  <- mydata2[random_sample1, ]
testing_dataset2 <- mydata2[-random_sample1, ]
model2 <- train(Outcome ~., data = training_dataset2, "glm",family = "binomial")
summary(model2)

predictions2 <- predict(model2, testing_dataset2)

results2 <- data.frame(Predicted_Outcome2 = as.factor(predictions2),
                       Actual_Outcome2 = testing_dataset2$Outcome)

confusionMatrix(data = results2$Predicted_Outcome2, reference = results2$Actual_Outcome2)
#As we can see we can gain the same amount of accuracy with lesser variables so we can conclude
#that the optimal model will have Pregnancies,Glucose,BloodPressure,BMI,Outcome as the most significant variables

# ROC curve for model 1
roc_data1 <- roc(testing_dataset$Outcome, as.numeric(predictions))

# ROC curve for model 2
roc_data2 <- roc(testing_dataset2$Outcome, as.numeric(predictions2))

# Plotting both ROC curves on the same graph
plot(roc_data1, col = "blue", main = "ROC Curves - Logistic Regression Models")
lines(roc_data2, col = "red")

#the plot can give us the same picturethat with lesser variables we have a better lighter model that performs the same.



## Support Vector Machines

# Setting seed to generate a reproducible random sampling
set.seed(123)

#Scale
mydata1 <- copy(mydata)  # Making a copy of the original data

numeric_columns <- names(mydata)[sapply(mydata, is.numeric) & names(mydata) != "Outcome"]
mydata1[, (numeric_columns) := lapply(.SD, scale), .SDcols = numeric_columns]
mydata1$Outcome <- as.factor(mydata1$Outcome)

# Setting seed to generate a reproducible random sampling
set.seed(123)

# Randomly creating training data as 75% of the dataset
random_sample <- createDataPartition(mydata1$Outcome, p = 0.75,list = F)

# Generating training dataset from the random_sample
training_dataset  <- mydata1[random_sample,]

# Generating testing dataset from rows which are not included in random_sample
testing_dataset <- mydata1[-random_sample, ]

### Identify the best kernel, cost C, and sigma
kernelValues <- c("linear", "radial")
costValuesToCheck <- 10^(-5:5)
sigmaValuesToCheck <- 10^(-3:3)

trainAccuracy <- array(NA, dim = c(length(kernelValues), length(costValuesToCheck), length(sigmaValuesToCheck)))
testAccuracy <- array(NA, dim = c(length(kernelValues), length(costValuesToCheck), length(sigmaValuesToCheck)))

for (k in 1:length(kernelValues)) {
  for (i in 1:length(costValuesToCheck)) {
    for (j in 1:length(sigmaValuesToCheck)) {
      cat("Calculate SVM accuracy for kernel =", kernelValues[k], ", cost C =", costValuesToCheck[i], "and sigma =", sigmaValuesToCheck[j], "\n")
      curSVM <- svm(formula = Outcome ~ ., 
                    data = training_dataset, 
                    type = 'C-classification', 
                    kernel = kernelValues[k],
                    scale = FALSE,
                    cost = costValuesToCheck[i],
                    gamma = 1 / (2 * sigmaValuesToCheck[j]^2))
      trainAccuracy[k, i, j] <- confusionMatrix(data = curSVM$fitted, 
                                                reference = training_dataset$Outcome)$overall[1]
      testAccuracy[k, i, j] <- confusionMatrix(data = predict(curSVM, testing_dataset), 
                                               reference = testing_dataset$Outcome)$overall[1]
    }
  }
}

# Finding the best parameters
best_accuracy <- max(testAccuracy)
best_indices <- which(testAccuracy == best_accuracy, arr.ind = TRUE)
best_kernel <- kernelValues[best_indices[1,1]]
best_cost <- costValuesToCheck[best_indices[1,2]]
best_sigma <- sigmaValuesToCheck[best_indices[1,3]]

best_accuracy
best_kernel
best_cost
best_sigma

# Defining training control as cross-validation and value of K equal to 5
train_control <- trainControl(method = "cv",number = 5)

# Fitting SVM to the Training set with the best parameters
SVMclassifier <- svm(
  formula = Outcome ~ ., 
  data = training_dataset, 
  type = 'C-classification', 
  kernel = best_kernel,  # Use the best kernel value
  scale = FALSE,
  cost = best_cost,  # Use the best 'C' parameter
  gamma = 1 / (2 * best_sigma^2), # Use the best 'sigma' parameter
  trControl = train_control  # Use the best 'sigma' parameter
)

summary(SVMclassifier)

# Training confusion matrix
cm1 <- confusionMatrix(data = SVMclassifier$fitted, reference = training_dataset$Outcome)

cm1$overall

# Predicting the Test set results 
y_pred <- predict(SVMclassifier, newdata = testing_dataset) 

# Making the Test Confusion Matrix 
cm <- table(testing_dataset$Outcome, y_pred) 
confusionMatrix(cm)

SVMclassifier
SVMclassifier$SV
dim(SVMclassifier$SV)


## For variables: "Glucose" and “BMI”

mydata3 <- mydata[, c("Glucose", "BMI", "Outcome")]
mydata3$Outcome <- as.factor(mydata3$Outcome)

# Setting seed to generate a reproducible random sampling
set.seed(123) 

# Randomly creating training data as 75% of the dataset
random_sample3 <- createDataPartition(mydata3$Outcome, p = 0.75,list = F)

# Generating training dataset from the random_sample
training_dataset3  <- mydata3[random_sample3,]

# Generating testing dataset from rows which are not included in random_sample
testing_dataset3 <- mydata3[-random_sample3, ]

##Logistic Regression

# Fit Logistic Regression model using 5-fold cross-validation
glm_model3 <- train(Outcome ~ ., data = training_dataset3, method = "glm",family = "binomial")

summary(glm_model3)

# Predicting the target variable
predictions3 <- predict(glm_model3, testing_dataset3)

# Κατασκευή data frame με τις προβλέψεις και τις πραγματικές τιμές Outcome
results <- data.frame(
  Predicted_Outcome = as.factor(predictions3),
  Actual_Outcome = testing_dataset3$Outcome
)
# Υπολογισμός απόδοσης του μοντέλου
confusionMatrix(
  data = results$Predicted_Outcome, 
  reference = results$Actual_Outcome
)
# ROC curve for model 
roc_data3 <- roc(testing_dataset3$Outcome, as.numeric(predictions3))

# Plotting
plot(roc_data3, col = "blue", main = "ROC Curves - Logistic Regression Models")

#We can assume a very efficient model


## Support Vector Machines

# Scale only the numeric columns: Glucose and BMI
numeric_cols <- names(mydata3)[sapply(mydata3, is.numeric)]
mydata3[, (numeric_cols) := lapply(.SD, scale), .SDcols = numeric_cols]

# Convert to data.table
setDT(mydata3)

# Ensure the 'Outcome' column remains unchanged
mydata3$Outcome <- as.factor(mydata3$Outcome)

# Setting seed to generate a reproducible random sampling
set.seed(400)

# Randomly creating training data as 75% of the dataset
random_sample <- createDataPartition(mydata3$Outcome, p = 0.75,list = F)

# Generating training dataset from the random_sample
training_dataset  <- mydata3[random_sample,]

# Generating testing dataset from rows which are not included in random_sample
testing_dataset <- mydata3[-random_sample, ]

### Identify the best kernel, cost C, and sigma as before

for (k in 1:length(kernelValues)) {
  for (i in 1:length(costValuesToCheck)) {
    for (j in 1:length(sigmaValuesToCheck)) {
      cat("Calculate SVM accuracy for kernel =", kernelValues[k], ", cost C =", costValuesToCheck[i], "and sigma =", sigmaValuesToCheck[j], "\n")
      curSVM <- svm(formula = Outcome ~ ., 
                    data = training_dataset, 
                    type = 'C-classification', 
                    kernel = kernelValues[k],
                    scale = FALSE,
                    cost = costValuesToCheck[i],
                    gamma = 1 / (2 * sigmaValuesToCheck[j]^2))
      trainAccuracy[k, i, j] <- confusionMatrix(data = curSVM$fitted, 
                                                reference = training_dataset$Outcome)$overall[1]
      testAccuracy[k, i, j] <- confusionMatrix(data = predict(curSVM, testing_dataset), 
                                               reference = testing_dataset$Outcome)$overall[1]
    }
  }
}

# Finding the best parameters
best_accuracy <- max(testAccuracy)
best_indices <- which(testAccuracy == best_accuracy, arr.ind = TRUE)
best_kernel <- kernelValues[best_indices[1,1]]
best_cost <- costValuesToCheck[best_indices[1,2]]
best_sigma <- sigmaValuesToCheck[best_indices[1,3]]

best_accuracy
best_kernel
best_cost
best_sigma

# Fitting SVM to the Training set with the best parameters
SVMclassifier3 <- svm(
  formula = Outcome ~ ., 
  data = training_dataset, 
  type = 'C-classification', 
  kernel = best_kernel,  # Use the best kernel value
  scale = FALSE,
  cost = best_cost,  # Use the best 'C' parameter
  gamma = 1 / (2 * best_sigma^2), # Use the best 'sigma' parameter
  trControl = train_control
  )


summary(SVMclassifier3)

# Training confusion matrix
cm3 <- confusionMatrix(data = SVMclassifier3$fitted, reference = training_dataset$Outcome)

cm3$overall

# Predicting the Test set results 
y_pred3 <- predict(SVMclassifier3, newdata = testing_dataset) 

# Making the Test Confusion Matrix 
cm <- table(testing_dataset$Outcome, y_pred3) 
cm
confusionMatrix(cm)

CreatePlot <- function(dataSet, SVMclassifierObject, plotTitle) {
  
  X1 <- seq(min(dataSet[, 1]) - 0.5, max(dataSet[, 1]) + 0.5, by = 0.01)
  X2 <- seq(min(dataSet[, 2]) - 0.5, max(dataSet[, 2]) + 0.5, by = 0.01)
  
  grid_set <- expand.grid(X1, X2)
  colnames(grid_set) <- c('Glucose', 'BMI')
  y_grid <- predict(SVMclassifierObject, newdata = grid_set)
  
  plot(dataSet[, -3],
       main = plotTitle,
       xlab = 'Glucose', ylab = 'BMI',
       xlim = range(X1), ylim = range(X2))
  
  contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
  
  # Change 'Yes' and 'No' to '1' and '0' for coloring
  points(grid_set$Glucose, grid_set$BMI, pch = '.', col = ifelse(y_grid == 1, 'aquamarine', 'coral1'))
  
  points(dataSet, pch = 21, cex = 1.5, bg = ifelse(dataSet[, 3] == 1, 'green4', 'red3'))
}

CreatePlot(dataSet = training_dataset,SVMclassifierObject = SVMclassifier3,
           plotTitle = "SVM (train set)")

dim(SVMclassifier3$SV)



##At last we will try the abovementioned for costs: C=1 and C=1000 

#C=1
# Fitting SVM to the Training set with the parameters
SVMclassifier4 <- svm(
  formula = Outcome ~ ., 
  data = training_dataset, 
  type = 'C-classification', 
  kernel = best_kernel,  # Use the best kernel value
  scale = FALSE,
  cost = 1,  
  gamma = 1 / (2 * best_sigma^2), # Use the best 'sigma' parameter
  trControl = train_control
)

summary(SVMclassifier4)
cm4 <- confusionMatrix(data = SVMclassifier4$fitted, reference = training_dataset$Outcome)
cm4$overall
y_pred4 <- predict(SVMclassifier4, newdata = testing_dataset) 
cm4 <- table(testing_dataset$Outcome, y_pred4) 
cm4
confusionMatrix(cm4)

CreatePlot(dataSet = training_dataset,SVMclassifierObject = SVMclassifier4,
           plotTitle = "SVM C=1")

dim(SVMclassifier4$SV)

#C=1000 
# Fitting SVM to the Training set with the parameters
SVMclassifier5 <- svm(
  formula = Outcome ~ ., 
  data = training_dataset, 
  type = 'C-classification', 
  kernel = best_kernel,  # Use the best kernel value
  scale = FALSE,
  cost = 1000,
  gamma = 1 / (2 * best_sigma^2), # Use the best 'sigma' parameter
  trControl = train_control
)

summary(SVMclassifier5)

# Training confusion matrix
cm5 <- confusionMatrix(data = SVMclassifier5$fitted, reference = training_dataset$Outcome)
cm5$overall

# Predicting the Test set results 
y_pred5 <- predict(SVMclassifier5, newdata = testing_dataset) 

# Making the Test Confusion Matrix 
cm5<- table(testing_dataset$Outcome, y_pred5) 
cm5
confusionMatrix(cm4)

CreatePlot(dataSet = training_dataset,SVMclassifierObject = SVMclassifier5,
           plotTitle = "SVM C= 1000")

dim(SVMclassifier5$SV)



