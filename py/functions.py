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
#all functions were created with help from scikit-learn tutorials online and vscode suggestions
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