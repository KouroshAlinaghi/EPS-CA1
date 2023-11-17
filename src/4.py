import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import poisson, binom, norm

green = '#57cc99'
red = '#e56b6f'
yellow = '#ffca3a'
purple = '#6a4c93'

line_thickness = 3

def draw_norm(mu, sigma, l, r):
    x = np.arange(l, r, 0.01)
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))
    plt.plot(x, y, color=green, label="X ~ N({}, {:.2f})".format(mu, sigma), linewidth=line_thickness)

def draw_pois(lamb, l, r):
    x = np.arange(l, r, 1)
    y = poisson.pmf(x, lamb)
    plt.plot(x, y, color=red, label="X ~ Poi({})".format(lamb), linewidth=line_thickness)

def draw_bin(n, p, l, r):
    x = np.arange(l, r)
    plt.plot(x, binom.pmf(x, n, p), color=yellow, ms=8, label='X ~ Bin({}, {})'.format(n, p), linestyle='--', linewidth=line_thickness)

def draw_random_bin(n, p, size):
    data = np.random.binomial(n, p, size)
    num_bins = 50
    counts, bins = np.histogram(data, bins=num_bins)
    bins = bins[:-1] + (bins[1] - bins[0])/2
    probs = counts/float(counts.sum())
    plt.bar(bins, probs, 1.0/num_bins, color=blue, label="Randomized Bin({}, {})".format(n, p))

n = 7072
p = 0.45
q = 1 - p
mu = n * p
sigma = (n * p * q)**0.5
l = int(mu - 3 * sigma)
r = int(mu + 3 * sigma)
m = 30000

draw_norm(mu, sigma, l, r)
draw_pois(mu, l, r)
draw_bin(n, p, l, r)

data = np.random.binomial(n, p, m)
plt.hist(data, bins=np.arange(l, r) - 0.5, density=True, color=purple, label="Randomized Bin({}, {})".format(n, p), alpha=0.8)

plt.title("Binomail prediction with Normal and Poisson distributions")
plt.xlabel("x")
plt.ylabel("P{X = x}")
plt.legend()
plt.grid(True)
plt.show()
