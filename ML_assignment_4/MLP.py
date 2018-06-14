import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
X = X.T
Y = np.array([1, 0, 1, 1])

# model specifications
Ni = 3
No = 1

# parameter and array initialization
Ntrials = 1000
w = np.random.randn(No, Ni)
dwo = np.zeros(w.shape)
error = np.array([])

for trial in range(Ntrials):
    y = 1/(1+np.exp(-np.dot(w, X)))             # output for all pattern
    delta_output = y*(1-y)*(Y-y)        # delta output

    # update weights with momentum
    dwo = 0.9*dwo + np.dot(delta_output, X.T)
    w = w + 0.1*dwo

    error = np.append(error, np.sum(np.abs(Y-y)))
plt.plot(error)
plt.show()


# ==============================================================================================
# multi layer

# model specs
Ni = 3
Nh = 4
No = 1

# Parameter and array initialization
Ntrials = 1000
wh = np.random.randn(Nh, Ni)
d_wh = np.zeros(wh.shape)
wo = np.random.randn(No, Nh)
d_wo = np.zeros(wo.shape)

error = np.zeros([])

for trial in range(Ntrials):
    h = 1/(1+np.exp(-np.dot(wh, X)))        # hidden activation fro all aptterns
    y = 1/(1+np.exp(-np.dot(wo, h)))        # output for all patterns

    delta_output = y*(1-y)*(Y-y)
    delta_hidden = h*(1-h)*(np.dot(wo.T, delta_output))  # delta backprop

    # updating weights and computing error
    d_wo = 0.9*d_wo + np.dot(delta_output, h.T)
    wo = wo + 0.1*d_wo

    d_wh = 0.9*d_wh + np.dot(delta_hidden, X.T)
    wh = wh + 0.1*d_wh

    error = np.append(error, np.sum(np.abs(Y-y)))
plt.plot(error)
plt.xlabel("iterations")
plt.ylabel("Absolute error")
plt.show()




