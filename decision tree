import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

#loading the csv file as a dataframe using pandas
wine = pd.read_csv('wine.csv')

#determining the columns in the csv file and printing them
original_headers = list(wine.columns.values)
print(original_headers)

#splitting the columns from the csv file into class and features
Class = wine.quality
Features = wine.drop('quality', axis=1)

#preprocessing: making Features(data) in the range of -1 to 1
Features = preprocessing.scale(Features)

#splitting the data into training set and test set
train_feature, test_feature, train_class, test_class = train_test_split(Features,Class,train_size=0.75,test_size=0.25)

#defining the random forest decision tree and its various parameters
tree = RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=100,max_features='sqrt',oob_score=True)

#fitting the decision tree model
tree.fit(train_feature, train_class)

#calculating and printing the training set accuracy score
Training_accuracy=tree.score(train_feature, train_class)
print("\nTraining set score: {:.3f}".format(Training_accuracy))

#calculating and printing the test set accuracy score
Test_accuracy=tree.score(test_feature, test_class)
print("\nTest set accuracy score: {:.3f}".format(Test_accuracy))

#After being fitted, the decision tree model can then be used to predict the class of samples
prediction = tree.predict(test_feature)
print("\nConfusion matrix:")

#printing the confusion matrix
print(pd.crosstab(test_class, prediction, rownames=['True'], colnames=['Predicted'], margins=True))

#Applying 10-fold stratified cross-validation
skf = StratifiedKFold(n_splits=10,random_state=None, shuffle=True)
Fold_accuracy = []
for train_index, test_index in skf.split(Features, Class):
    #defining the train and test data
    train_feature, test_feature = Features[train_index], Features[test_index]
    train_class, test_class = Class[train_index], Class[test_index]
    #fitting the decision tree model for the 10-fold stratified cross_validation
    tree.fit(train_feature, train_class)
    #Obtaining the accuracy score for the 10-fold stratified cross_validation
    accuracy = tree.score(test_feature, test_class)
    #appending the accuracy scores into a list
    Fold_accuracy.append(accuracy)
    
print("\n")
#printing the list of accuracy scores at each fold
print(Fold_accuracy)

#calculating and printing the Average accuracy score of the 10 folds
Average = np.mean(Fold_accuracy)
print("\nAverage cross-validation score: {:.2f}".format(Average))

#finding the improved accuracy between the decision tree and the 10 fold stratified cross-validation
Improved_accuracy = Average - Test_accuracy
print("\nImproved Accuracy is {:.2f}".format(Improved_accuracy))
