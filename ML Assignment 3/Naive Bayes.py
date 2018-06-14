from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import string
import math

print("importing***")
categories=['alt.atheism', 'comp.graphics', 'misc.forsale', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset="train", remove=('headers', 'footers', 'quotes'), categories=categories)
print(np.unique(newsgroups_train.target_names))
# newsgroups_test = fetch_20newsgroups(subset="test", remove=('headers', 'footers', 'quotes'))

train_data = pd.DataFrame({"x": newsgroups_train.data,
                           "y": newsgroups_train.target})
#
# test_data = pd.DataFrame({"x": newsgroups_test.data,
#                           "y": newsgroups_test.target})


# cleaning the data
def clean_data(x):
    print("**cleaning data")
    x = np.array(x)
    for i in range(len(x)):
        # remove punctuations
        x[i] = x[i].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        # remove numbers
        x[i] = x[i].translate(str.maketrans("", "", string.digits))
        # convert to lower case
        x[i] = x[i].lower()
    return x


x = clean_data(train_data["x"])
# x_test = clean_data(test_data["x"])

print("transforming*****")
# constructing feature vector
vectorizer = CountVectorizer(stop_words="english", binary=True)
X = vectorizer.fit_transform(x)

# X_test = vectorizer.fit_transform(x_test)
x = X.toarray()
# x_test = X_test.toarray()

# conversion to Binomial feature sets
# x[x > 1] = 1

# splitting into train and test
x_train = np.split(x, [int(len(x)*0.7), len(x)])[0]
x_test = np.split(x, [int(len(x)*0.7), len(x)])[1]

y = np.array(train_data["y"])
y_train = np.split(y,[int(len(y)*0.7), len(y)])[0]
y_test = np.split(y,[int(len(y)*0.7), len(y)])[1]


# calculating priors and likelihoods for each word per document class
priors = np.array([])          # p(C=k)
word_likelihoods = np.array([])
for i in np.unique(train_data["y"]):
    print("learning******")
    priors = np.append(priors, len(y_train[y_train == i])/len(y_train))
    if i == 0:
        word_likelihoods = np.append(word_likelihoods,
                                     (np.sum(x_train[y_train == i], axis=0)+1)/(len(y_train[y_train == i])+2))
    elif i == 1:
        word_likelihoods = np.append([word_likelihoods],
                                     [(np.sum(x_train[y_train == i], axis=0)+1)/(len(y_train[y_train == i])+2)],
                                     axis=0)
    else:
        word_likelihoods = np.append(word_likelihoods,
                                     [(np.sum(x_train[y_train == i], axis=0)+1)/(len(y_train[y_train == i])+2)],
                                     axis=0)


# predict
predicted_class = np.array([])
for j in range(len(x_test)):
    print(j)
    pred_probabs = np.array([])
    for i in np.unique(train_data["y"]):
        posteriors = ((word_likelihoods[i] * x_test[j]) +
                             ((1 - word_likelihoods[i]) * (1 - x_test[j])))
        posteriors = np.prod(posteriors) * priors[i]
        pred_probabs = np.append(pred_probabs, posteriors)

    predicted_class = np.append(predicted_class, np.argmax(pred_probabs))
print(predicted_class)
cfm = confusion_matrix(predicted_class, y_test)
print(accuracy_score(predicted_class, y_test))
pd.DataFrame(cfm, columns=['alt.atheism', 'comp.graphics', 'misc.forsale', 'sci.space'],
             index = ['alt.atheism', 'comp.graphics', 'misc.forsale', 'sci.space'])

