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


def NN(X, Y, Ntrials, Nh=26):
    # model specs
    Ni = 157
    No = 7
    # Parameter and array initialization
    wh = np.random.randn(Nh, Ni)
    d_wh = np.zeros(wh.shape)
    wo = np.random.randn(No, Nh)
    d_wo = np.zeros(wo.shape)

    error = np.array([])

    for trial in range(Ntrials):
        h = 1/(1+np.exp(-np.dot(wh, X.T)))        # hidden activation fro all patterns
        # print("h in iteration :", trial, ": ",h)
        y = 1/(1+np.exp(-np.dot(wo, h)))        # output for all patterns
        # print("y in iteration :", trial, ": ",y)

        delta_output = y*(1-y)*(Y-y.T).T
        # print("delta_output in iteration :", trial, ": ",delta_output)
        delta_hidden = h*(1-h)*(np.dot(wo.T, delta_output))  # delta backprop
        # print("delta_hidden in iteration :", trial, ": ",delta_hidden)


        # updating weights and computing error
        d_wo = 0.9*d_wo + np.dot(delta_output, h.T)
        # print("d_wo in iteration :", trial, ": ",d_wo)
        wo = wo + 0.1*d_wo
        # print("wo in iteration :", trial, ": ",wo)


        d_wh = 0.9*d_wh + np.dot(delta_hidden, X)
        # print("d_wh in iteration :", trial, ": ",d_wh)
        wh = wh + 0.1*d_wh
        # print("wh in iteration :", trial, ": ",wh)


        error = np.append(error, np.sum(np.abs(Y-y.T)))
        # print("error in iteration :", trial, ": ",error)
    return error, wh, wo


error, wh, wo = NN(X, Y, 10000)
plt.plot(error)
plt.xlabel("iterations")
plt.ylabel("Absolute error")
plt.show()

# =====================++++++++++++++++++++++++++++++++++++++++++++++++