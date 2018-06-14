import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

# importing the data
iris_data = np.loadtxt("iris.data", delimiter=",")
np.random.seed(10)
np.random.shuffle(iris_data)
print(iris_data.shape)

# splitting into testing and training data-sets
train_X = iris_data[0::2, 0:4]
train_y = np.int32(iris_data[0::2, 4])
test_X = iris_data[1::2, 0:4]
test_y = np.int32(iris_data[1::2, 4])

# training a SVM classifier
clf = svm.SVC()
clf.fit(train_X, train_y)

kf = KFold(n_splits=10)
CV_scores_SVM = cross_val_score(clf, train_X, train_y, cv=kf)

# training a Random Forest Classifier
clf = RandomForestClassifier()
clf.fit(train_X, train_y)

kf = KFold(n_splits=10)
CV_scores_RF = cross_val_score(clf, train_X, train_y, cv=kf)


# my implementation of cross validation
accuracy_scores_RF = np.array([])
accuracy_scores_SVM = np.array([])
for i in range(10):
    CV_test_X = iris_data[i::10, 0:4]
    CV_test_y = iris_data[i::10, 4]
    CV_train_X = np.array([])
    CV_train_y = np.array([])
    for j in range(10):
        if j == i:
            continue
        if len(CV_train_X) == 0:
            CV_train_X = iris_data[j::10, 0:4]
            CV_train_y = iris_data[j::10, 4]
        else:
            CV_train_X = np.concatenate((CV_train_X, iris_data[j::10, 0:4]), axis=0)
            CV_train_y = np.concatenate((CV_train_y, iris_data[j::10, 4]), axis=0)
    clf = svm.SVC()
    clf.fit(CV_train_X, CV_train_y)
    CV_predicted_y = clf.predict(CV_test_X)
    accuracy_scores_SVM = np.append(accuracy_scores_SVM, np.mean(CV_test_y == CV_predicted_y))

    clf = RandomForestClassifier()
    clf.fit(CV_train_X, CV_train_y)
    CV_predicted_y = clf.predict(CV_test_X)
    accuracy_scores_RF = np.append(accuracy_scores_RF, np.mean(CV_test_y == CV_predicted_y))

print("Cross validated scores (using SVM) on native implementation : ", np.mean(CV_scores_SVM))
print("Cross validated scores (using RF) on native implementation : ", np.mean(CV_scores_SVM))
print("Cross Validated scores (using SVM) on my implementation : ", np.mean(accuracy_scores_SVM))
print("Cross Validated scores (using RF) on my implementation : ", np.mean(accuracy_scores_RF))
