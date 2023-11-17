import numpy as np
import matplotlib.pyplot as plt 

green = '#57cc99'
blue = '#22577a'
red = '#e56b6f'
yellow = '#ffca3a'
purple = '#6a4c93'

line_thickness = 3

l, r = 20, 80 + 1
repetition = 2000

size = (r - l) * repetition

physics = []
for x in range(l, r):
    for t in range(repetition):
        physics.append(x)

plt.hist(physics, 101, density=True, color=green, label="Phy ~ U({}, {})".format(l, r - 1), alpha=0.5, edgecolor='black')

lamb = 8
ap = np.random.exponential(lamb, size=size)
plt.hist(ap, 101, density=True, color=blue, label="AP ~ Exp({})".format(lamb), alpha=0.5, edgecolor='black')

lamb = 40
dm = np.random.poisson(lamb, size=size)
plt.hist(dm, 101, density=True, color=red, label="DM ~ Poi({})".format(lamb), alpha=0.5, edgecolor='black')

plt.title("Poisson, Exponential and Uniform distributions")
plt.xlabel("x")
plt.ylabel("P{X = x}")
plt.legend()
plt.grid(True)
plt.show()

mu = 90.125
sigma = 18.8
x = np.arange(40, 160, 0.01)
y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))
plt.plot(x, y, color=purple, label="X ~ N({}, {:.2f})".format(mu, sigma), linewidth=line_thickness)
plt.hist(np.sum([physics, ap, dm], axis=0), 100, density=True, color=yellow, label="Phy + AP + DM", alpha=0.5, edgecolor='black')

plt.title("Sum of Poisson, Exponential and Uniform distributions")
plt.xlabel("x")
plt.ylabel("P{X = x}")
plt.legend()
plt.grid(True)
plt.show()
