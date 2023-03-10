from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils import toNumpyArray

# You may add more classifier methods replicating this function
def applyNaiveBayes(X_train, y_train, X_test):
    '''
    Task: Given some features train a Naive Bayes classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = MultinomialNB()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


def applySVC(X_train, y_train, X_test):
    '''
    Task: Given some features train a SVC classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = SVC()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


def applyRandomForest(X_train, y_train, X_test):
    '''
    Task: Given some features train a Random Forest classifier
          and return its predictions over a test set
    Input; X_train -> Train features
           y_train -> Train_labels
           X_test -> Test features 
    Output: y_predict -> Predictions over the test set
    '''
    trainArray = toNumpyArray(X_train)
    testArray = toNumpyArray(X_test)
    
    clf = RandomForestClassifier()
    clf.fit(trainArray, y_train)
    y_predict = clf.predict(testArray)
    return y_predict


def applyKNN(X_train, y_train, X_test):
      '''
      Task: Given some features train a KNN classifier
              and return its predictions over a test set
      Input; X_train -> Train features
               y_train -> Train_labels
               X_test -> Test features 
      Output: y_predict -> Predictions over the test set
      '''
      trainArray = toNumpyArray(X_train)
      testArray = toNumpyArray(X_test)
      
      clf = KNeighborsClassifier()
      clf.fit(trainArray, y_train)
      y_predict = clf.predict(testArray)
      return y_predict