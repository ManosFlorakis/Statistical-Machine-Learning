###STATISTICAL MACHINE LEARNING ASSIGNMENT II

#Necessary Libraries
#!pip install sklearn
#!pip install matplotlib
#!pip install pandas
#!pip install seaborn
#!pip install tensorflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load Data
data = pd.read_csv('diabetes.csv')

##Random Forest

#i) Random Forest for all variables

# Standardize the data
scale = StandardScaler()
columns_to_scale = data.columns[:-1]  # Exclude the 'Outcome' column
data[columns_to_scale] = scale.fit_transform(data[columns_to_scale])

# Split data into features (x) and target variable (y)
x = data.iloc[:, :-1]
y = data['Outcome']

# Split the scaled data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=12345)

# Initialize results list
results = []

# Define the number of trees
n_trees = [500, 1000, 2000]

# Loop through different numbers of trees
for n_tree in n_trees:
    # Create RandomForestClassifier with the specified number of trees
    rf_classifier = RandomForestClassifier(random_state=123, n_estimators=n_tree)
    
    # Perform cross-validation
    cv_scores = cross_val_score(rf_classifier, x_train, y_train, cv=5)
    
    # Fit the model on the training set
    rf_classifier.fit(x_train, y_train)
    
    # Predictions on the testing dataset
    predictions_train = rf_classifier.predict(x_train)
    
    # Calculate testing accuracy
    train_accuracy = accuracy_score(y_train, predictions_train)
    
    # Predictions on the testing dataset
    predictions_test = rf_classifier.predict(x_test)
    
    # Calculate testing accuracy
    test_accuracy = accuracy_score(y_test, predictions_test)
    
    # Append results to the list
    results.append([n_tree,cv_scores.mean(),test_accuracy])

    # Visualize feature importance
    feature_importances = pd.Series(rf_classifier.feature_importances_, index=x_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(f"Visualizing Important Features - {n_tree} Trees")
    plt.show()

# Convert the list to a DataFrame
results_df = pd.DataFrame(results, columns=['Number of Trees','Training Accuracy','Testing Accuracy'])

# Print the results
print(results_df)



#ii) Random Forest for “Glucose” & “BMI”

# Load Data
data = pd.read_csv('diabetes.csv')

# Select a subset of columns
subset_columns = ["Glucose", "BMI"]

# Standardize the subset data
scale = StandardScaler()
data[subset_columns] = scale.fit_transform(data[subset_columns])

# Use only the subset columns
x_subset = data[subset_columns]
y = data['Outcome']

# Split the scaled subset data into training and testing sets
x_train_subset, x_test_subset, y_train, y_test = train_test_split(x_subset, y, train_size=0.8, random_state=12345)

# Initialize results list
results_subset = []

# Define the number of trees
n_trees = [500, 1000, 2000]

# Loop through different numbers of trees
for n_tree in n_trees:
    # Create RandomForestClassifier with the specified number of trees
    rf_classifier_subset = RandomForestClassifier(random_state=123, n_estimators=n_tree)
    
    # Perform cross-validation on the subset
    cv_scores_subset = cross_val_score(rf_classifier_subset, x_train_subset, y_train, cv=5)
    
    # Fit the model on the training set (subset)
    rf_classifier_subset.fit(x_train_subset, y_train)
    
    # Predictions on the testing dataset (subset)
    predictions_train_subset = rf_classifier_subset.predict(x_train_subset)
    
    # Calculate testing accuracy (subset)
    train_accuracy_subset = accuracy_score(y_train, predictions_train_subset)
    
    # Predictions on the testing dataset (subset)
    predictions_test_subset = rf_classifier_subset.predict(x_test_subset)
    
    # Calculate testing accuracy (subset)
    test_accuracy_subset = accuracy_score(y_test, predictions_test_subset)
    
    # Append results to the list
    results_subset.append([n_tree,cv_scores_subset.mean(),test_accuracy_subset])

    # Visualize feature importance for the subset
    feature_importances_subset = pd.Series(rf_classifier_subset.feature_importances_, index=x_train_subset.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances_subset, y=feature_importances_subset.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(f"Visualizing Important Features - {n_tree} Trees (Subset)")
    plt.show()

# Convert the list to a DataFrame for the subset
results_df_subset = pd.DataFrame(results_subset, columns=['Number of Trees', 'Training Accuracy', 'Testing Accuracy'])

# Print the results for the subset
print(results_df_subset)




##Feedforward Neural Network

# Load Data
data = pd.read_csv('diabetes.csv')

# Convert outcome to a numeric array and explicitly cast to int64
y = np.array(data['Outcome'], dtype=np.int64)

# Normalize data (variables only)
scale = StandardScaler()
X = scale.fit_transform(data.iloc[:, :8])

# Create training data as 80% of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Build and train the neural network model
def build_and_train_model(neurons, learning_rate):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(8,)),
        Dense(neurons, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        x=X_train,
        y=y_train,
        epochs=300,
        batch_size=100,
        validation_split=0.2
    )

    # Evaluate the model on the test set
    test_metrics = model.evaluate(x=X_test, y=y_test)
    train_metrics = model.evaluate(x=X_train, y=y_train)

    # Plot training history
    plot_training_history(neurons, learning_rate, history)

    return neurons, learning_rate, train_metrics[1], test_metrics[1]

# Function to plot training history
def plot_training_history(neurons, learning_rate, history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - Neurons: {neurons}, LR: {learning_rate}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - Neurons: {neurons}, LR: {learning_rate}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Matrix for results
results = []

# Different parameters
neurons_list = [4, 8, 16]
learning_rates = [0.001, 0.01]

# Running the models and extracting results
for neurons in neurons_list:
    for lr in learning_rates:
        result = build_and_train_model(neurons, lr)
        results.append(result)

# Display a summary table of results
summary_table = pd.DataFrame(results, columns=['Neurons', 'Learning Rate', 'Train Accuracy', 'Test Accuracy'])
print(summary_table)





#ii) FNN for “Glucose” & “BMI”

# Load Data
data = pd.read_csv('diabetes.csv')

# Change types of variables to numeric and factor only
data['Outcome'] = pd.Categorical(data['Outcome']).codes

# Normalize data (variables only) and check the results
scale = StandardScaler()
data[['Glucose', 'BMI']] = scale.fit_transform(data[['Glucose', 'BMI']])

# Convert outcome to a numeric array and explicitly cast to int64
y = np.array(data['Outcome'], dtype=np.int64)

# Create a reproducible random sampling
np.random.seed(123)
random_sample = np.random.rand(len(data)) < 0.8
train_data = data[random_sample]
test_data = data[~random_sample]

# Build and train the neural network model
def build_and_train_model(neurons, learning_rate):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_shape=(2,)))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        x=np.array(train_data[['Glucose', 'BMI']]),
        y=y[train_data.index],
        epochs=300,
        batch_size=100,
        validation_split=0.2
    )

    # Evaluate the model on the test set
    test_metrics = model.evaluate(
        x=np.array(test_data[['Glucose', 'BMI']]),
        y=y[test_data.index]
    )

    # Evaluate the model on the training set
    train_metrics = model.evaluate(
        x=np.array(train_data[['Glucose', 'BMI']]),
        y=y[train_data.index]
    )

    # Plot training history
    plot_training_history(neurons, learning_rate, history)

    return [neurons, learning_rate, train_metrics[1], test_metrics[1]]

# Function to plot training history
def plot_training_history(neurons, learning_rate, history):
    plt.figure(figsize=(12, 6))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - Neurons: {neurons}, LR: {learning_rate}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - Neurons: {neurons}, LR: {learning_rate}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Matrix for results
results = np.empty((6, 4))
results[:] = np.nan
colnames = ["Neurons", "Learning Rate", "Train Accuracy", "Test Accuracy"]

# Different parameters
neurons_list = [4, 8, 16]
learning_rates = [0.001, 0.01]

# Running the models and extracting results
index = 0
for neurons in neurons_list:
    for lr in learning_rates:
        result = build_and_train_model(neurons, lr)
        results[index, :] = result
        index += 1

# Display a summary table of results
summary_table = pd.DataFrame(results, columns=colnames)
print(summary_table)






