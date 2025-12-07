###Statistical Machine Learning
##Assingment 1

#Neccesary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
from numpy import arange
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn import datasets
warnings.filterwarnings('ignore')

#Exercise 1

#Load Data
data = pd.read_csv('carsales.csv')
data.head()


# Create a LabelEncoder object for each column
label_encoder_fuel = LabelEncoder()
label_encoder_seller = LabelEncoder()
label_encoder_transmission = LabelEncoder()

# Apply Label Encoding to each categorical column
data['Fuel_Type'] = label_encoder_fuel.fit_transform(data['Fuel_Type'])
data['Seller_Type'] = label_encoder_seller.fit_transform(data['Seller_Type'])
data['Transmission'] = label_encoder_transmission.fit_transform(data['Transmission'])
data.head()

# Define the StandardScaler
scale = StandardScaler()

x = data.iloc[:, [1, 3, 4, 5, 6, 7, 8]]
y = data['Selling_Price']

# Scale the features
x_scaled = scale.fit_transform(x)

# Split the scaled data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, train_size=0.75)

modelLinear = LinearRegression()

# Perform cross-validation
cv_scores = cross_val_score(modelLinear, x_train, y_train, cv=10)  # Using 10-fold cross-validation

# Fit the model on the training data
modelLinear.fit(x_train, y_train)

# Retrieve feature names
feature_names = x.columns.tolist()

# Retrieve coefficients of the features
coefficients = modelLinear.coef_

# Compute R-squared scores
R2_train_score_lr = modelLinear.score(x_train, y_train)
R2_test_score_lr = modelLinear.score(x_test, y_test)

# Compute mean cross-validation score
mean_cv_score = np.mean(cv_scores)

print("Feature Names:", feature_names)
print("Coefficients:", coefficients)
print("The train R-squared score for LR model is {:.4f}".format(R2_train_score_lr))
print("The test R-squared score for LR model is {:.4f}".format(R2_test_score_lr))
print("Mean Cross-validated R-squared score for LR model is {:.4f}".format(mean_cv_score))

# Predictions on the test set
y_pred = modelLinear.predict(x_test)
# Scatter plot of predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.title('Actual vs Predicted Selling Price // Linear')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.xlim(0,20)
plt.ylim(0,20)
plt.grid(True)
plt.show()

##Ridge

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# define Ridge model
ridge_model = RidgeCV(alphas=arange(0.01, 1000, 5), cv=cv)  # Adjust the alpha range as needed

# fit model
ridge_model.fit(x_train, y_train)

# Train and test score for ridge regression
R2_train_score_ridge = ridge_model.score(x_train, y_train)
R2_test_score_ridge = ridge_model.score(x_test, y_test)

# Summarize chosen configuration
print('Ridge best alpha (lambda): %f' % ridge_model.alpha_)
print("The train score for ridge model is {:.4f}".format(R2_train_score_ridge))
print("The test score for ridge model is {:.4f}".format(R2_test_score_ridge))
# Retrieve coefficients after fitting with the best alpha
print("Ridge coefficients:", ridge_model.coef_)

# Predictions on the test set
y_pred = ridge_model.predict(x_test)
# Scatter plot of predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.title('Actual vs Predicted Selling Price // Ridge')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.xlim(0,20)
plt.ylim(0,20)
plt.grid(True)
plt.show()

y_pred = ridge_model.predict(x_test)
plt.figure(figsize=(8, 6))
plt.plot(y_test.values, label='Actual', color='red')  # Plotting y_test as red line
plt.plot(y_pred, label='Predicted', color='blue')    # Plotting y_pred as blue line
plt.xlabel('Sample Index')
plt.ylabel('Selling Price')
plt.title('Ridge Regression Predictions vs. Actual Prices')
plt.legend()
plt.show()

##Lasso

# define Lasso model
lasso_model = LassoCV(alphas=arange(0, 10, 0.1), cv=cv, n_jobs=-1, max_iter=1000)

# fit model
lasso_model.fit(x_train, y_train)

# Train and test score for Lasso regression
R2_train_score_ls = lasso_model.score(x_train, y_train)
R2_test_score_ls = lasso_model.score(x_test, y_test)

# Summarize chosen configuration
print('Lasso best alpha (lambda): %f' % lasso_model.alpha_)
print("The train score for Lasso model is {:.4f}".format(R2_train_score_ls))
print("The test score for Lasso model is {:.4f}".format(R2_test_score_ls))
print("Lasso coefficients:", lasso_model.coef_)

# Predictions on the test set
y_pred = lasso_model.predict(x_test)
# Scatter plot of predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.title('Actual vs Predicted Selling Price // Lasso')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.xlim(0,20)
plt.ylim(0,20)
plt.grid(True)
plt.show()

y_pred = lasso_model.predict(x_test)
plt.figure(figsize=(8, 6))
plt.plot(y_test.values, label='Actual', color='red')  # Plotting y_test as red line
plt.plot(y_pred, label='Predicted', color='blue')    # Plotting y_pred as blue line
plt.xlabel('Sample Index')
plt.ylabel('Selling Price')
plt.title('Lasso Regression Predictions vs. Actual Prices')
plt.legend()
plt.show()


##Elastic Net

# Define Elastic Net model
ratios = arange(0, 1, 0.01)
alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]  # Removed 0.0 from alphas list
ElasticNetModel = ElasticNetCV(l1_ratio=ratios, alphas=alphas, cv=cv, n_jobs=-1)

# Fit model
ElasticNetModel.fit(x_train, y_train)

# Train and test score for Elastic Net regression
R2_train_score_en = ElasticNetModel.score(x_train, y_train)
R2_test_score_en = ElasticNetModel.score(x_test, y_test)

# Retrieve best alpha and l1_ratio
best_alpha_en = ElasticNetModel.alpha_
best_l1_ratio_en = ElasticNetModel.l1_ratio_

# Summarize chosen configuration
print('ElasticNet best alpha (lambda): %f' % best_alpha_en)
print('ElasticNet best l1_ratio (alpha): %f' % best_l1_ratio_en)
print("The train score for ElasticNet model is {:.4f}".format(R2_train_score_en))
print("The test score for ElasticNet model is {:.4f}".format(R2_test_score_en))

# Predictions on the test set
y_pred = ElasticNetModel.predict(x_test)
# Scatter plot of predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.title('Actual vs Predicted Selling Price // Elastic Net')
plt.xlabel('Actual Selling Price')
plt.ylabel('Predicted Selling Price')
plt.xlim(0,20)
plt.ylim(0,20)
plt.grid(True)
plt.show()



###Exercise 2

##For all data

#Load Data
data = pd.read_csv('diabetes.csv')
data.head()

# Select Variables for x
x = data.iloc[:,0:8]
x.head()
# Select Outcome as target
y = data.iloc[:,8]
y.head()

# standardize features
scale = StandardScaler()
x = scale.fit_transform(x)

# Create a 75% random split of data for training/testing
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75,stratify=y)

##Logistic Regression

# Initialize Logistic Regression model
log_reg_model = LogisticRegression()

# Fit the model
log_reg_model.fit(x_train, y_train)

# Retrieve coefficients of the features
coefficients_log = log_reg_model.coef_

# Evaluate model performance on training and testing set
R2_train_score_log = log_reg_model.score(x_train, y_train)
R2_test_score_log = log_reg_model.score(x_test, y_test)

print("Train score for Logistic Regression: {:.4f}".format(R2_train_score_log))
print("Train score for Logistic Regression: {:.4f}".format(R2_test_score_log))
print("Coefficients:", coefficients_log)

# Get predicted probabilities
y_pred_prob = log_reg_model.predict_proba(x_test)[:, 1]

# Calculate fpr, tpr, thresholds and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


##Support Vector Machines

# Identify best hyperparameters by exhaustive grid search
# Defining parameter ranges
param_grid = {
    'C': [10 ** i for i in range(-5, 6)],  # Values of C from 10^-5 to 10^5
    'gamma': [10 ** i for i in range(-3, 4)],  # Values of gamma (sigma) from 10^-3 to 10^3
    'kernel': ['linear', 'rbf']
}  

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, cv=5) 
  
# fitting the model for grid search 
grid.fit(x_train, y_train) 

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
best_model = grid.best_estimator_
print(best_model) 

grid_predictions = grid.predict(x_test) 
  
# print classification report 
print(metrics.classification_report(y_train, grid.best_estimator_.predict(x_train),
  target_names=["0", "1"]))

print(metrics.confusion_matrix(y_test, grid_predictions)) 
print(metrics.classification_report(y_test, grid_predictions,target_names=['0', '1']))



##For the Variables Glucose and BMI

# Select Age and Estimated Salary as features (predictors)
x2 = data.iloc[:, [1, 5]]
x2.head()
# Select Item Purchased as target
y.head()
# standardize features
x2 = scale.fit_transform(x2)

# Create a 75% random split of data for training/testing
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, train_size=0.75,stratify=y)


#Logistic Regression

# Initialize Logistic Regression model
log_reg_model1 = LogisticRegression()
# Fit the model
log_reg_model1.fit(x2_train, y_train)

# Retrieve coefficients of the features
coefficients_log1 = log_reg_model1.coef_

# Evaluate model performance on training and testing set
R2_train_score_log1 = log_reg_model1.score(x2_train, y_train)
R2_test_score_log1 = log_reg_model1.score(x2_test, y_test)

print("Train score for Logistic Regression: {:.4f}".format(R2_train_score_log1))
print("Train score for Logistic Regression: {:.4f}".format(R2_test_score_log1))
print("Coefficients:", coefficients_log1)

# Get predicted probabilities
y_pred_prob1 = log_reg_model1.predict_proba(x2_test)[:, 1]

# Calculate fpr, tpr, thresholds and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob1)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()




##Support Vector Machines

# Identify best hyperparameters by exhaustive grid search
grid2 = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, cv=5) 
# fitting the model for grid search 
grid2.fit(x2_train, y_train) 

# print best parameter after tuning 
print(grid2.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
best_model2 = grid2.best_estimator_
print(best_model2) 

grid_predictions2 = grid2.predict(x2_test) 
  
# print classification report 
print(metrics.classification_report(y_train, grid2.best_estimator_.predict(x2_train),
  target_names=["0", "1"]))

print(metrics.confusion_matrix(y_test, grid_predictions2)) 
print(metrics.classification_report(y_test, grid_predictions2,target_names=['0', '1']))

# Define ranges for creating the meshgrid
x_min, x_max = x2[:, 0].min() - 1, x2[:, 0].max() + 1
y_min, y_max = x2[:, 1].min() - 1, x2[:, 1].max() + 1
h = 0.02  # Step size in the mesh

# Create a meshgrid
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the class for each point in the meshgrid
Z = best_model2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision boundary and data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x2[:, 0], x2[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary')
plt.show()

#Plotting for C=1 and C=1000

# Accessing the best parameters
best_params = grid2.best_params_

# Extracting individual best parameters
best_gamma = best_params['gamma']
best_kernel = best_params['kernel']

#For C=1

SVMclassifier1 = SVC(C=1, kernel=best_kernel, gamma= best_gamma)
SVMclassifier1.fit(x2_train, y_train)

# Define ranges for creating the meshgrid
x_min, x_max = x2[:, 0].min() - 1, x2[:, 0].max() + 1
y_min, y_max = x2[:, 1].min() - 1, x2[:, 1].max() + 1
h = 0.02  # Step size in the mesh

# Create a meshgrid
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the class for each point in the meshgrid
Z = SVMclassifier1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision boundary and data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x2[:, 0], x2[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary C=1')
plt.show()

#For C=1000

SVMclassifier2 = SVC(C=1000, kernel=best_kernel, gamma= best_gamma)
SVMclassifier2.fit(x2_train, y_train)

# Define ranges for creating the meshgrid
x_min, x_max = x2[:, 0].min() - 1, x2[:, 0].max() + 1
y_min, y_max = x2[:, 1].min() - 1, x2[:, 1].max() + 1
h = 0.02  # Step size in the mesh

# Create a meshgrid
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict the class for each point in the meshgrid
Z = SVMclassifier2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision boundary and data points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(x2[:, 0], x2[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary C=1000')
plt.show()

