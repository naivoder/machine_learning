
import numpy as np
import matplotlib.pyplot as plt

def broadcastable(*arrays) -> bool:
    try:
        np.broadcast(*arrays)
        return True
    except ValueError:
        return False

def assign_labels(data, centroids) -> np.ndarray:
    return np.argmin(np.linalg.norm(data - centroids, axis=2), axis=0)

def plot(data, centroids, bounds, labels, title=None):
    # target colors
    c1, c2 = ['#bc13fe', '#be0119']
    fig, ax = plt.subplots(figsize=(5, 5))
    # turn (x, y) pairs into list(x), list(y) with transpose
    ax.scatter(*data.T, c=np.where(labels, c2, c1), alpha=0.3, s=80)
    ax.scatter(*centroids.T, c=[c1, c2], marker='s', alpha=0.9, s=95, edgecolor='black')
    ax.set_ylim(*bounds)
    ax.set_xlim(*bounds)
    if title:
        ax.set_title(title)
    plt.show()

def shift_centroids(data, labels):
    a_group = data[np.where(labels==0)]
    a_center = np.mean(a_group, axis=0)
    b_group = data[np.where(labels==1)]
    b_center = np.mean(b_group, axis=0)
    centroids = np.array([a_center, b_center])
    if not broadcastable(data, centroids):
        centroids = centroids[:, None]
    return centroids

def setup():
    # generate random data distribution
    blob = np.repeat([[3, 3], [10, 10]], [5, 5], axis=0)
    blob = blob + np.random.randn(*blob.shape) * np.random.randint(1, 10)
    rands = np.array([[np.random.randint(1, 12), np.random.randint(1, 12)] for x in range(30)])
    data = np.concatenate((blob, rands), axis=0)
    bounds = lower, upper = np.trunc([data.min() * 0.9, data.max() * 1.1])
    # initialize centroids (k = 2)
    cents = np.array([[3, 3], [10, 10]])
    if not broadcastable(blob, cents):
        cents = cents[:, None]
    return data, cents, bounds

def get_iters():
    n = input('How many iterations? ')
    try:
        n = int(n)
    except ValueError:
        print('Oops! Please enter a positive integer...')
        n = get_iters()
    return n

if __name__=='__main__':
    repeat = get_iters()
    data, cents, bounds = setup()
    for iteration in range(repeat):
        # assign labels
        labels = assign_labels(data, cents)
        # visualize model state
        plot(data, cents, bounds, labels, title='K Means Clustering: Iteration {:d}'.format(iteration + 1))
        # adjust centroids
        cents = shift_centroids(data, labels)
        # assign labels
        labels = assign_labels(data, cents)
        # visualize model state
        plot(data, cents, bounds, labels, title='K Means Clustering: New Centroid Locations')
