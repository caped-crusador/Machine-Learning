from pylab import *
from mpl_toolkits.mplot3d import Axes3D

hsize = array([937, 1150, 1170, 1290, 1275, 1410, 1550, 1730, 1910])
price = array([187, 222, 330, 310, 290, 440, 600, 550, 600])

# normalize the data for better results
hsize = (hsize-min(hsize))/(max(hsize)-min(hsize))
price = (price-min(price))/(max(price)-min(price))

# code for linear regression
w1 = array([-1])
w2 = array([-1])
w3 = array([-1])
w4 = array([-1])
w5 = array([-1])
w6 = array([-1])
w7 = array([-1])
L = array([])
alpha = 0.11
for i in range(999):
    y = w3[-1]*hsize**2 + w2[-1]*hsize + w1[-1] + w4[-1]*hsize**3 + w5[-1]*hsize**4 + w6[-1] * hsize ** 5 + w7[-1]*hsize**6

    w7 = append(w7, w7[-1] - alpha * sum((y - price) * hsize ** 6))
    w6 = append(w6, w6[-1] - alpha * sum((y - price) * hsize ** 5))
    w5 = append(w5, w5[-1] - alpha * sum((y - price) * hsize ** 4))
    w4 = append(w4, w4[-1] - alpha * sum((y - price) * hsize ** 3))
    w3 = append(w3, w3[-1]-alpha*sum((y-price)*hsize**2))
    w2 = append(w2, w2[-1]-alpha*sum((y-price)*hsize))
    w1 = append(w1, w1[-1]-alpha*sum(y-price))
    L = append(L, sum((y-price)**2))
plot(hsize, price, "*")
plot(hsize, y)
xlabel("Size (sqft)")
ylabel("Price ($)")
show()
loglog(L)
xlabel("Iterations")
ylabel("Loss")
show()

# --------------plot the loss surface and learning path---------------------

ax = plt.figure().gca(projection='3d')

# Make data.
a0 = np.arange(-2, 3, 0.1)
b0 = np.arange(-2, 2, 0.1)
a0, b0 = np.meshgrid(a0, b0)

L = np.zeros(a0.shape)
for i in range(np.size(hsize)):
    L = L + (a0 * hsize[i] + b0 - price[i]) ** 2

# Plot the surface.

surf = ax.plot_surface(a0, b0, np.log(L), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# Customize the z axis.
ax.yaxis.set_ticks(np.arange(-3, 3, 1))
ax.view_init(90, 0)
ax.plot(w1, w2, 'k.-')
ax.grid(False)
plt.gca().set_aspect('equal', adjustable='box')

plt.show()

# ----------------------------------------------------------

