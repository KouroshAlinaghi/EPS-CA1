import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import poisson, norm

green = '#57cc99'
blue = '#22577a'
red = '#e56b6f'

line_thickness = 3

n = 250
p = 0.008
q = 1 - p
mu = n * p
sigma = (n * p * q)**0.5
m = 5000

width = 0.2

x_values = np.arange(0, 8)

binomial_samples = np.random.binomial(n, p, m)
binomial_probs = [np.sum(binomial_samples == x) / m for x in x_values]
poisson_pmf = poisson.pmf(x_values, mu)
normal_pdf = norm.pdf(x_values, mu, sigma)

fig, ax = plt.subplots()

ax.bar(x_values - width, poisson_pmf, width, label='X ~ Pois({})'.format(mu), color=blue, alpha=0.7, edgecolor='black')
ax.bar(x_values, binomial_probs, width, label='X ~ Bin({}, {})'.format(n, p), color=green, alpha=0.7, edgecolor='black')
ax.bar(x_values + width, normal_pdf, width, label='X ~ N({}, {:.2f})'.format(mu, sigma), color=red, alpha=0.7, edgecolor='black')

ax.set_xticks(x_values)
ax.set_xlabel('x')
ax.set_ylabel('P{X = x}')
ax.set_title("Binomail preidiction with Normal and Poisson distributions")
ax.legend()
plt.grid(True)
plt.show()
