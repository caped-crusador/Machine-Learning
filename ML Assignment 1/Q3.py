import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

# load the data
train_data = np.loadtxt("wine.train", delimiter=",")
np.random.seed(10)
np.random.shuffle(train_data, )
test_data = np.loadtxt("wine.test", delimiter=",")
# X_train, X_test, y_train, y_test = train_test_split(data[:, 1:13], data[:, 0], test_size=0.3, random_state=10)
X_train, X_test, y_train = train_data[:, 1:13], test_data[:, 1:13], train_data[:, 0]

# standardising the data
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# training the classifier and making predictions
clf = RandomForestClassifier(n_estimators=200, max_features=2, max_depth=100, random_state=10)
clf.fit(X=X_train, y=y_train)
predicted_labels = clf.predict(X_test)
np.savetxt("predicted_labels.csv", predicted_labels, delimiter=",")
print(predicted_labels)


