import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab


N = []
for i in range(20):
    d1 = np.random.randint(low=1, high=7)
    d2 = np.random.randint(low=1, high=7)
    N.append(d1+d2)
print(N)

n, bins, patches = plt.hist(N, facecolor='#2e86de')
plt.title("Histogram of 2 Dice rolled(20 times)")
plt.ylabel("Frequency")
plt.xlabel("Numbers")
plt.show()

# -----------------L-------------------------

N = []
for i in range(1000):
    d1 = np.random.randint(low=1, high=7)
    d2 = np.random.randint(low=1, high=7)
    N.append(d1+d2)
print(N)

n, bins, patches = plt.hist(N, facecolor='#2e86de')
plt.title("Histogram of 2 Dice rolled(1000 times)")
plt.ylabel("Frequency")
plt.xlabel("Numbers")
plt.show()

# -------------------------------------------
# -------------------------------------------

N = []
for i in range(20):
    d1 = np.random.choice(a=[1, 0, 3, 4, 5, 6])
    d2 = np.random.randint(low=1, high=7)
    N.append(d1+d2)
print(N)

n, bins, patches = plt.hist(N, facecolor='#2e86de')
plt.title("Histogram of 2 Dice(1 Die flawed) rolled(20 times)")
plt.ylabel("Frequency")
plt.xlabel("Numbers")
plt.show()

# -------------------------------------------

N = []
for i in range(1000):
    d1 = np.random.choice(a=[1, 0, 3, 4, 5, 6])
    d2 = np.random.randint(low=1, high=7)
    N.append(d1+d2)
print(N)

n, bins, patches = plt.hist(N, facecolor='#2e86de')
plt.title("Histogram of 2 Dice(1 Die flawed) rolled(1000 times)")
plt.ylabel("Frequency")
plt.xlabel("Numbers")
plt.show()

# -------------------- 7 dices ------------------

N = []
for i in range(20):
    d1 = np.random.randint(low=1, high=7)
    d2 = np.random.randint(low=1, high=7)
    d3 = np.random.randint(low=1, high=7)
    d4 = np.random.randint(low=1, high=7)
    d5 = np.random.randint(low=1, high=7)
    d6 = np.random.randint(low=1, high=7)
    d7 = np.random.randint(low=1, high=7)
    N.append(d1+d2+d3+d4+d5+d6+d7)
print(N)

n, bins, patches = plt.hist(N, normed=1, facecolor='#2e86de')
# mu is the mean of data and sigma is one standard deviation
mu, sigma = norm.fit(N)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, "r--", linewidth=2)
plt.title("Histogram of 7 Dice rolled(20 times)\n mean=%.3f, sigma=%.3f" % (mu, sigma))
plt.ylabel("Probability")
plt.xlabel("Numbers")
plt.show()

# -----------------L-------------------------

N = []
for i in range(1000):
    d1 = np.random.randint(low=1, high=7)
    d2 = np.random.randint(low=1, high=7)
    d3 = np.random.randint(low=1, high=7)
    d4 = np.random.randint(low=1, high=7)
    d5 = np.random.randint(low=1, high=7)
    d6 = np.random.randint(low=1, high=7)
    d7 = np.random.randint(low=1, high=7)
    N.append(d1+d2+d3+d4+d5+d6+d7)
print(N)

n, bins, patches = plt.hist(N, normed=1, facecolor='#2e86de')
# mu is the mean of data and sigma is one standard deviation
mu, sigma = norm.fit(N)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, "r--", linewidth=2)
plt.title("Histogram of 7 Dice rolled(1000 times)\n mean=%.3f, sigma=%.3f" % (mu, sigma))
plt.ylabel("Probability")
plt.xlabel("Numbers")
plt.show()

# -------------------------------------------
# -------------------------------------------

N = []
for i in range(20):
    d1 = np.random.choice(a=[1, 0, 3, 4, 5, 6])
    d2 = np.random.randint(low=1, high=7)
    d3 = np.random.randint(low=1, high=7)
    d4 = np.random.randint(low=1, high=7)
    d5 = np.random.randint(low=1, high=7)
    d6 = np.random.randint(low=1, high=7)
    d7 = np.random.randint(low=1, high=7)
    N.append(d1+d2+d3+d4+d5+d6+d7)
print(N)

n, bins, patches = plt.hist(N, normed=1, facecolor='#2e86de')
# mu is the mean of data and sigma is one standard deviation
mu, sigma = norm.fit(N)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, "r--", linewidth=2)
plt.title("Histogram of 7 Dice (1 Die flawed) rolled(20 times)\n mean=%.3f, sigma=%.3f" % (mu, sigma))
plt.ylabel("Probability")
plt.xlabel("Numbers")
plt.show()

# -------------------------------------------

N = []
for i in range(1000):
    d1 = np.random.choice(a=[1, 0, 3, 4, 5, 6])
    d2 = np.random.randint(low=1, high=7)
    d3 = np.random.randint(low=1, high=7)
    d4 = np.random.randint(low=1, high=7)
    d5 = np.random.randint(low=1, high=7)
    d6 = np.random.randint(low=1, high=7)
    d7 = np.random.randint(low=1, high=7)
    N.append(d1+d2+d3+d4+d5+d6+d7)
print(N)

n, bins, patches = plt.hist(N, normed=1, facecolor='#2e86de')
# mu is the mean of data and sigma is one standard deviation
mu, sigma = norm.fit(N)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, "r--", linewidth=1)
plt.title("Histogram of 7 Dice (1 Die flawed) rolled(1000 times)\n mean=%.3f, sigma=%.3f" % (mu, sigma))
plt.ylabel("Probability")
plt.xlabel("Numbers")
plt.show()
