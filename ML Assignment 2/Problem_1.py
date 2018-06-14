import pandas as pd
from pylab import *
import numpy as np

data = pd.read_csv("houses.csv")

data.describe()

Y = data["price (grands)"]
X = data.iloc[:, 1:16]
X_1 = data.sqft_living

# def linear_regression(predictor_vals, target_vals, learning_rate, iterations):


def linear_regression(predictor_vals, target_vals, lr, repetitions):
    target_vals = array(target_vals)
    if shape(predictor_vals) == (21613, 15):
        W = pd.DataFrame(columns=["w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7",
                                  "w8", "w9", "w10", "w11", "w12", "w13", "w14", "w15"])
        W.loc[0] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        L = array([])
        for i in range(len(predictor_vals.columns)):
            predictor_vals.iloc[:, i] = (predictor_vals.iloc[:, i] - min(predictor_vals.iloc[:, i])) / \
                                        (max(predictor_vals.iloc[:, i]) - min(predictor_vals.iloc[:, i]))
        target_vals = (target_vals - min(target_vals)) / (max(target_vals) - min(target_vals))
        lr = 0.000009
        for i in range(repetitions):
            w_update = array([])
            for j in range(16):
                if j == 0:
                    y = W.iloc[-1, j]
                    continue
                y += W.iloc[-1, j] * predictor_vals.iloc[:, j - 1]
            for j in range(16):
                if j == 0:
                    w_update = append(w_update, W.iloc[-1, j] - lr * sum(y - target_vals))
                    continue
                w_update = append(w_update, W.iloc[-1, j] - lr * sum((y - target_vals) * predictor_vals.iloc[:, j - 1]))
            W.loc[i + 1] = w_update
            L = append(L, sum((y - target_vals) ** 2))
        loglog(L)
        xlabel("Iterations")
        ylabel("Loss")
        show()
        return W, L
    else:
        predictor_vals = array(predictor_vals)

        # normalizing the data
        predictor_vals = (predictor_vals - min(predictor_vals)) / (max(predictor_vals) - min(predictor_vals))
        target_vals = (target_vals - min(target_vals)) / (max(target_vals) - min(target_vals))
        # lr = 0.00008
        w1 = array([-1])
        w2 = array([-1])
        L = array([])
        predictor_vals = array(predictor_vals)
        target_vals = array(target_vals)

        for i in range(repetitions):
            y = w1[-1] * predictor_vals + w2[-1]
            w1 = append(w1, w1[-1] - lr * sum((y - target_vals) * predictor_vals))
            w2 = append(w2, w2[-1] - lr * sum(y - target_vals))
            L = append(L, sum((y - target_vals) ** 2))
        plot(predictor_vals, target_vals, ".")
        plot(predictor_vals, y)
        xlabel("Size (sqft)")
        ylabel("Price ($)")
        savefig("priceVSsize.png")
        show()
        loglog(L)
        xlabel("Iterations")
        ylabel("Loss")
        savefig("LossVSiterations.png")
        show()
        return w1, w2, L


def predict(x, w=0, param1=0, param2=0):
    y = array([])
    if x.shape == (21613,):
        x = array(x)
        for i in range(len(x)):
            y = append(y, param1[-1]*x[i] + param2[-1])
        return y
    else:
        for i in range(len(x)):
            for j in range(16):
                if j == 0:
                    y = w.iloc[-1, j]
                    continue
                y += w.iloc[-1, j] * x.iloc[:, j - 1]
        return y


# for learning rate = 0.00008
weights = pd.DataFrame(columns=["w0", "w1"])
p1, p2, loss = linear_regression(predictor_vals=X_1, target_vals=Y, lr=0.000001, repetitions=10000)
y_1 = predict(param1=p1, param2=p2, x=X_1)
plot(X_1, Y, ".", color="Orange")
plot(X_1, y_1)
xlabel("Size (sqft)")
ylabel("Price ($)")
figtext(0.2, 0.8, "Learning rate = 0.00008")
show()


# For X
parameters, loss2 = linear_regression(predictor_vals=X, target_vals=Y, lr=0.000009, repetitions=1000)
y = predict(w=parameters, x=X)
loglog(loss, label="X_1")
loglog(loss2, label="X")
xlabel("Iterations")
ylabel("Loss")
legend(loc="best", shadow=True)
savefig("LossVSiterations.png")
show()

# for learning rate = 0.00009
p1, p2, loss = linear_regression(predictor_vals=X_1, target_vals=Y, lr= 0.00009, repetitions=10000)
y = predict(param1=p1, param2=p2, x=X_1)
plot(X_1, Y, ".", color="Orange")
plot(X_1, y)
xlabel("Size (sqft)")
ylabel("Price ($)")
figtext(0.2, 0.8, "Learning rate = 0.00009")
show()

# actual Vs Predicted for X
Y_norm = (Y - min(Y)) / (max(Y) - min(Y))
plot(y, ".", alpha=0.3, label="Predicted")
plot(Y_norm, ".", alpha=0.3, label="Actual")
xlabel("Examples")
ylabel("Value")
legend()
show()

# for learning rate = 0.00005
p1, p2, loss = linear_regression(predictor_vals=X_1, target_vals=Y, lr= 0.00005, repetitions=10000)
y = predict(param1=p1, param2=p2, x=X_1)
plot(X_1, Y, ".", color="Orange")
plot(X_1, y)
xlabel("Size (sqft)")
ylabel("Price ($)")
figtext(0.2, 0.8, "Learning rate = 0.00005")
show()


