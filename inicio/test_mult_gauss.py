from matplotlib import pyplot as plt
import matplotlib
import numpy as np
mean = [0, 0]
cov = [[1, 0], [0, 1]]
x, y = np.random.multivariate_normal(mean, cov, 1000000).T
dist = np.sqrt(x ** 2 + y ** 2)
cmap = matplotlib.cm.hot(dist)
plt.scatter(x, y, c=dist, linewidth='0', cmap=matplotlib.cm.hot); plt.axis('equal'); plt.show() # noqa

print x.std(), y.std()
