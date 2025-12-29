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
#occasionally other sources such as chatgpt used to correct or give suggestions - mentioned in comments
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

#function for question 3 - imbalancing classes

#function to imbalance dataset by reducing samples of specific classes 
#vscode suggestion with some modifications from chatgpt when it wasnt working correctly
def imbalance_classes(x, y, classes_to_reduce, reduction_fraction=0.5, random_state=42): #reduce specific classes by a fraction
    import numpy as np
    np.random.seed(random_state) #setting random seed 42 for reproducibility 
    indices_to_keep = [] #storing samples to keep
    for cls in y.unique(): #iterating through classes in target
        cls_indices = y[y == cls].index #getting locations of samples for the class
        if cls in classes_to_reduce: #checking if class needs to be reduced
            n_to_keep = int(len(cls_indices) * (1 - reduction_fraction)) #calculating number of samples to keep
            kept_indices = np.random.choice(cls_indices, n_to_keep, replace=False) #randomly selecting samples to keep
        else: #if class not being reduced
            kept_indices = cls_indices #keep the samples the same
        indices_to_keep.extend(kept_indices) #adding samples to the list
    indices_to_keep = sorted(indices_to_keep) #sorting them back to the correct order - chatgpt suggestion
    x_imbalanced = x.loc[indices_to_keep] #creating imbalanced feature set
    y_imbalanced = y.loc[indices_to_keep] #creating imbalanced target set
    return x_imbalanced, y_imbalanced #returns the imbalanced features and target
#if data not organised it caused issues for analysis
