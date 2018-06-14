import pandas as pd
from pylab import *

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
            print(i)
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
            print(i)
            for j in range(16):
                if j == 0:
                    y = w.iloc[-1, j]
                    continue
                y += w.iloc[-1, j] * x.iloc[:, j - 1]
        return y


# for learning rate = 0.00008
# weights = pd.DataFrame(columns=["w0", "w1"])
p1, p2, loss = linear_regression(predictor_vals=X_1, target_vals=Y, lr=0.00008, repetitions=10000)
y_1 = predict(param1=p1, param2=p2, x=X_1)
plot(X_1, Y, ".", color="Orange")
plot(X_1, y_1)
xlabel("Size (sqft)")
ylabel("Price ($)")
figtext(0.2, 0.8, "Learning rate = 0.00008")
show()

# ----------------------------Ridge--------------------------------------------------


def linear_regression_Ridge(predictor_vals, target_vals, lr, repetitions, lam):
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
            print(i)
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
            L = append(L, sum((y - target_vals) ** 2)+lam*sum(W.iloc[-1]**2))
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
            L = append(L, sum((y - target_vals) ** 2)+lam*sum([w1**2, w2**2]))
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
            print(i)
        return y
    else:
        for i in range(len(x)):
            print(i)
            temp1 = w.iloc[-1]
            temp2 = x.iloc[i]
            for j in range(16):
                if j == 0:
                    y = temp1[j]
                    continue
                y += w.iloc[-1, j] * temp2[i, j - 1]
        return y


# for learning rate = 0.00008
# weights = pd.DataFrame(columns=["w0", "w1"])
p1_ridge, p2_ridge, loss_ridge = linear_regression_Ridge(predictor_vals=X_1, target_vals=Y, lr=0.00008, repetitions=10000, lam=0)
y_1_ridge = predict(param1=p1_ridge, param2=p2_ridge, x=X_1)
plot(X_1, Y, ".", color="Orange")
plot(X_1, y_1_ridge)
xlabel("Size (sqft)")
ylabel("Price ($)")
figtext(0.2, 0.8, "Learning rate = 0.00008")
show()

# ------------------------------Ridge with momentum--------------------------------


def linear_regression_Ridge_momentum(predictor_vals, target_vals, lr, repetitions, lam, beta):
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
            print(i)
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
                if i == 0:
                    w_update = append(w_update, W.iloc[-1, j] - lr * sum((y - target_vals) * predictor_vals.iloc[:, j - 1]))
                else:
                    w_update = append(w_update,
                                      W.iloc[-1, j] - lr * sum((y - target_vals) * predictor_vals.iloc[:, j - 1]) +
                                      beta * (W.iloc[-1, j] - W.iloc[-2, j]))
            W.loc[i + 1] = w_update
            L = append(L, sum((y - target_vals) ** 2)+lam*sum(W.iloc[-1]**2))
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
            if i == 0:
                w1 = append(w1, (w1[-1] - lr * sum((y - target_vals) * predictor_vals)))
                w2 = append(w2, w2[-1] - lr * sum(y - target_vals))
            else:
                w1 = append(w1, (w1[-1] - lr * sum((y - target_vals) * predictor_vals))+beta*(w1[-1]-w1[-2]))
                w2 = append(w2, w2[-1] - lr * sum(y - target_vals)+beta*(w2[-1]-w2[-1]))
            L = append(L, sum((y - target_vals) ** 2)+lam*sum([w1**2, w2**2]))
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
            print(i)
            for j in range(16):
                if j == 0:
                    y = w.iloc[-1, j]
                    continue
                y += w.iloc[-1, j] * x.iloc[:, j - 1]
        return y


# for learning rate = 0.00008
# weights = pd.DataFrame(columns=["w0", "w1"])
p1_momentum, p2_momentum, loss_momentum = linear_regression_Ridge_momentum(predictor_vals=X_1, target_vals=Y, lr=0.000008, repetitions=10000, lam=0, beta=0.91)
y_1_momentum = predict(param1=p1, param2=p2, x=X_1)
plot(X_1, Y, ".", color="Orange")
plot(X_1, y_1_momentum)
xlabel("Size (sqft)")
ylabel("Price ($)")
figtext(0.2, 0.8, "Learning rate = 0.00008")
show()


# ax = plt.figure().gca(projection='3d')
plot(X_1, Y, ".")
plot(X_1, y_1, label="flat")
# plot(X_1, y_1_ridge, label="ridge")
# plot(X_1, y_1_momentum, label="momentum")
xlabel("Size (sqft)")
ylabel("Price ($)")
legend()
show()


# loglog(loss, label="flat")
loglog(loss_ridge, label="Ridge")
loglog(loss_momentum, label="Ridge+momentum")
legend()
show()
