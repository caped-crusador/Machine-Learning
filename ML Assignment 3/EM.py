import pandas as pd
from pylab import *
from scipy.stats import norm
from sklearn.metrics import silhouette_score, silhouette_samples

# loading the data
data = np.loadtxt("iris.data", delimiter=",")
sepal_data = data[:, 0:2]
sepal_data = np.insert(sepal_data, 2, data[:, 4], axis=1)
sepal_data = pd.DataFrame(data=sepal_data)

# initial plot
scatter(sepal_data[0], sepal_data[1], c=sepal_data[2], alpha=0.7)
xlabel("sepal length")
ylabel("sepal width")
legend()
show()


def prob(datapoint, mu, sig, lam):
    p = log(lam)
    for i in range(len(datapoint)):
        p += log(norm.pdf(datapoint[i], mu[i], sig[i][i]))
    return p


def EM(clusters=2, iterations=10):
    np.random.seed(5)
    sepal_data_copy = sepal_data.copy()
    sepal_data_copy = np.array(sepal_data_copy)

    # defining initial parameters
    mu = sepal_data_copy[np.random.choice(len(sepal_data_copy), clusters, False), 0:2]
    sigmas = [np.eye(2)] * clusters
    likelihoods = [1. / clusters] * clusters

    # print("initial mu's : ", mu)
    # print("initial sigmas : ", sigmas)
    # print("initial likelihoods : ", likelihoods)
    for k in range(iterations):
        # print(k)
        # log_lls = np.array([])
        for i in range(len(sepal_data_copy)):
            cl_lls = np.array([])
            for j in range(clusters):
                cl_lls = np.append(cl_lls, prob([sepal_data_copy[i, 0], sepal_data_copy[i, 1]],
                                                mu[j], sigmas[j], likelihoods[j]))
            print("probabilities : iteration", k, " datapoint ",i, " ", cl_lls, " : max ", argmax(cl_lls))
            sepal_data_copy[i][2] = cl_lls.argmax()
        # print("probabilities after iteration 1 : ",  cl_lls)


        scatter(sepal_data_copy[:, 0], sepal_data_copy[:, 1], c=sepal_data_copy[:, 2], alpha=0.9)
        xlabel("sepal length")
        ylabel("sepal width")
        show()


        point_fractions = np.array([])
        for j in range(clusters):
            point_fractions = np.append(point_fractions,
                                        len(sepal_data_copy[sepal_data_copy[:, 2] == j])/len(sepal_data_copy) )
        print(" new likelihoods : iteration ", k, " ", point_fractions)
        likelihoods = point_fractions

        for j in range(clusters):
            mu[j] = np.array([sepal_data_copy[sepal_data_copy[:, 2] == j][:, 0].mean(),
                              sepal_data_copy[sepal_data_copy[:, 2] == j][:, 1].mean()])
            sigmas[j] = np.diag([sepal_data_copy[sepal_data_copy[:, 2] == j][:, 0].std(),
                              sepal_data_copy[sepal_data_copy[:, 2] == j][:, 1].std()])
        print("mu after iteration ", k, " ", mu)
        print("sigma sfter iteration", k, " ", sigmas)
    return sepal_data_copy, likelihoods

sepal_data_copy, likelihoods = EM(clusters=3, iterations=10)
print(likelihoods)
scatter(sepal_data_copy[:,0], sepal_data_copy[:,1], c=sepal_data_copy[:, 2], alpha=0.9)
xlabel("sepal length")
ylabel("sepal width")
show()


# source : Sklearn
def evaluate(clusters):
    y_init = 10
    datapoint_Sil_Vals = silhouette_samples(sepal_data_copy[:, 0:2], sepal_data_copy[:, 2])
    for i in range(clusters):
        cluster_sil_vals = np.sort(datapoint_Sil_Vals[sepal_data_copy[:, 2] == i])
        cluster_size = len(cluster_sil_vals)
        axis_y = y_init + cluster_size

        color = cm.spectral(float(i) / 2)
        fill_betweenx(np.arange(y_init, axis_y),
                      0, cluster_sil_vals, facecolor=color,
                      edgecolor=color, alpha=0.8)
        # Label the silhouette plot
        text(-0.05, y_init + 0.5 * cluster_size, str(i))
        y_init = axis_y + 10
    show()
# ---------------------------------------------------------------------------

evaluate(clusters=3)

sepal_data_copy, likelihoods = EM(clusters=2, iterations=10)
evaluate(clusters=2)

sepal_data_copy, likelihoods = EM(clusters=4, iterations=10)
evaluate(clusters=4)
