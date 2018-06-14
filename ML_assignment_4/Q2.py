import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X_patt1 = pd.read_csv("pattern1.txt", sep=" ", header=None)
X_patt1 = np.array(X_patt1)
X = np.array(np.split(X_patt1, 26))
X = X.reshape(26, 156)
X = np.column_stack([X, np.ones(26)])

Y = np.array([list('{0:07b}'.format(65))])
for i in range(ord("B"), ord("Z")+1):
    Y = np.vstack((Y, list('{0:07b}'.format(i))))
Y = Y.astype(float)


def NN(X, Y, Ntrials):
    # model specs
    Ni = 157
    Nh = 26
    No = 7
    # Parameter and array initialization
    wh = np.random.randn(Nh, Ni)
    d_wh = np.zeros(wh.shape)
    wo = np.random.randn(No, Nh)
    d_wo = np.zeros(wo.shape)

    error = np.array([])

    for trial in range(Ntrials):
        h = 1/(1+np.exp(-np.dot(wh, X.T)))        # hidden activation fro all patterns
        y = 1/(1+np.exp(-np.dot(wo, h)))        # output for all patterns

        delta_output = y*(1-y)*(Y-y.T).T
        delta_hidden = h*(1-h)*(np.dot(wo.T, delta_output))  # delta backprop

        # updating weights and computing error
        d_wo = 0.9*d_wo + np.dot(delta_output, h.T)
        wo = wo + 0.1*d_wo

        d_wh = 0.9*d_wh + np.dot(delta_hidden, X)
        wh = wh + 0.1*d_wh

        error = np.append(error, np.sum(np.abs(Y-y.T)))
    return error, wh, wo


error, wh, wo = NN(X, Y, 1000)
plt.plot(error)
plt.xlabel("iterations")
plt.ylabel("Absolute error")
plt.show()


def predict_single(X_test, wh, wo):
    h_t = 1 / (1 + np.exp(-np.dot(wh, X_test)))  # hidden activation fro all patterns
    y_t = 1 / (1 + np.exp(-np.dot(wo, h_t)))

    y_t[y_t > 0.5] = 1
    y_t[y_t < 0.5] = 0
    y_t = y_t.astype(int).astype(str)
    predicted = chr(int("".join(y_t), 2))
    return predicted


predict_single(X[1], wh, wo)

# introducing noise
# X_test


def add_noise_single(X_test, noise_percent):
    for i in np.random.choice(156, int(156*(noise_percent/100))):
        if X_test[i] == 0:
            X_test[i] = 1
        else:
            X_test[i] = 0
    return X_test


predictions = np.array([])
for i in range(100):
    X_test = np.copy(X[1])
    add_noise_single(X_test, 10)
    predict_single(X_test, wh, wo)
    predictions = np.append(predictions, predict_single(X_test, wh, wo))


# ----------------------------pattern 2----------------------------------------------------

X_patt2 = pd.read_csv("pattern2.txt", sep=" ", header=None)
X_patt2 = X_patt2.dropna("columns")
X_patt2.columns = np.arange(13)
X_patt2 = np.array(X_patt2)
X_test = np.append(X_patt2.reshape((1, 156)), 1)


X_test = X_test[0:-1].reshape(12, 13)

# def predict(X_test, wh, wo):
#     h_t = 1/(1+np.exp(-np.dot(wh, X_test)))        # hidden activation fro all patterns
#     y_t = 1/(1+np.exp(-np.dot(wo, h_t)))
#
#     y_t[y_t > 0.5] = 1
#     y_t[y_t < 0.5] = 0
#     y_t = y_t.astype(int).astype(str)
#     predicted = chr(int("".join(y_t), 2))
#     print(predicted)



# patt3 = np.array([])
# for i in range(100):
#     error, wh, wo = NN(X, Y, 1000)
#     patt3 = np.append(patt3, predict(X_test, wh, wo))


def add_noise(X, noise_percent):
    x_noisy = np.copy(X)
    # add noisy examples in training set
    noise_points = int(156 * (noise_percent / 100))
    for i in np.random.choice(156, noise_points):
        for j in np.random.choice(26, noise_points):
            if x_noisy[j][i] == 0:
                x_noisy[j][i] = 1
            else:
                x_noisy[j][i] = 0
    return x_noisy

X_noisy = add_noise(X, noise_percent=10)

def predict(X_test, wh, wo):
    h_t = 1 / (1 + np.exp(-np.dot(wh, X_test.T)))  # hidden activation fro all patterns
    y_t = 1 / (1 + np.exp(-np.dot(wo, h_t)))

    y_t[y_t > 0.5] = 1
    y_t[y_t < 0.5] = 0
    y_t = y_t.astype(int).astype(str)
    y_t = y_t.T
    predicted = np.array([])
    for i in y_t:
        predicted = np.append(predicted, chr(int("".join(i), 2)))
    return predicted


predicted = predict(X_noisy, wh, wo)

true_labels = np.array([list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")])

predicted == true_labels
accuracy = (np.count_nonzero(predicted == true_labels)/26)*100