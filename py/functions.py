#library imports
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#helper functions 
#keeping notebooks clean with files here
#all functions were created with help from scikit-learn tutorials online and vscode suggestions,
#occasionally other sources such as chatgpt used to correct or give suggestions
#vscode suggested function parameters and random states = 42 , kept the same for reproducibility

#loading data function
def load_data(path): #put path in at time of use since it changes based on file location
    df = pd.read_csv(path)
    return df #returns a dataframe of the file

#function to split data from the dataframe into target and variables
def split_data(df, feature_cols, target_col): #features for prediction, target gets predicted
    x = df[feature_cols] #features used to predict - numerical values
    y = df[target_col] #target to be predicted - species 
    return x,y #returns the features and the target

#function to split data into training and testing sets
def split_train_test(x, y, test_size=0.2, random_state=42): #80-20 split so 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test #returns the split training and testing sets

#functions for question 1 - random forest 

#function to train a random forest classifier
def train_random_forest(x_train, y_train, n_estimators=100, random_state=42): #setting 100 trees to train
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state) #setting classifier
    clf.fit(x_train, y_train) #actually training the random forest on the data
    return clf #returns the trained classifier

#function to evaluate the model
def evaluate_model(clf, x_test, y_test): 
    y_pred = clf.predict(x_test) #predicting species based on test features, x
    accuracy = accuracy_score(y_test, y_pred) #using the into to calculate accuracy
    report = classification_report(y_test, y_pred) #reporting predicted classification
    return accuracy, report #returns accuracy and classification report

#function to evaluate feature importance
def feature_importance(clf, feature_cols):
    importances = clf.feature_importances_ #getting feature importance from the trained classifier
    feature_importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}) #creating a dataframe for better visualization
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False) #sorting by importance
    return feature_importance_df #returns the feature importance dataframe

#functions for question 2 - neural network

#function for feature scaling, standardising feature data to ensure no bias when training neural network
def scale_features(x_train, x_test): #scaling feature data to
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler() #initializing the scaler
    x_train_scaled = scaler.fit_transform(x_train) #fitting and transforming training data
    x_test_scaled = scaler.transform(x_test) #transforming test data
    return x_train_scaled, x_test_scaled #returns scaled training and testing features

#function to create and train a neural network - chatgpt, sklearn website used to make this
#activation and solver - relu and adam common for hidden layers
#relu allows non linearity to train the network for complex patterns
#adam is an optimization algorithm that adjusts weights efficiently
def train_neural_network(x_train_scaled, y_train, hidden_layers=(16, 16), epochs = 1000, random_state=42):
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation='relu', solver='adam', max_iter=epochs, random_state=random_state) #setting up the neural network
    mlp.fit(x_train_scaled, y_train) #training the neural network
    return mlp #returns the trained neural network
    
#funtion to evaluate the performance of the neural network
def evaluate_neural_network(mlp, x_test_scaled, y_test): #evaluating model based on test data
    y_pred = mlp.predict(x_test_scaled) #predicting species based on features from test dataset
    accuracy = accuracy_score(y_test, y_pred) #calculating accuracy
    report = classification_report(y_test, y_pred) #report of classification created
    return accuracy, report #returns accuracy and classification report