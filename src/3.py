import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm

green = '#57cc99'
blue = '#22577a'
red = '#e56b6f'
yellow = '#ffca3a'

line_thickness = 3

def norm_cdf_bs(mu, sigma, cdf):
    l, r = 0, mu + 2000
    while (r - l > 0.001):
        mid = (l + r) / 2
        if (norm.cdf((mid - mu) / sigma) > cdf):
            r = mid
        else:
            l = mid
    return (r + l) / 2

mu = 80
sigma = 12
min_grade = norm_cdf_bs(mu, sigma, 0.9)

x = np.arange(mu - 3 * sigma, mu + 3 * sigma, 0.01)
y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))
plt.plot(x, y, color="green", label="X ~ N({}, {})".format(mu, sigma), linewidth=line_thickness)    
plt.fill_between(x, y, where=(x < min_grade), color=yellow, alpha=0.3, label="F(x) = 0.9, x = {:.2f}".format(min_grade))

second_forth_start = norm_cdf_bs(mu, sigma, 0.25)
third_forth_end = norm_cdf_bs(mu, sigma, 0.75)
plt.fill_between(x, y, where=(x < third_forth_end) & (x > second_forth_start), color=red, alpha=0.3, label="0.25 < F(x) < 0.75, {:.2f} < x < {:.2f}".format(second_forth_start, third_forth_end))

plt.fill_between(x, y, where=(x < 90) & (x > 80), color=blue, alpha=0.3, label="P{{80 < X < 90}} = {:.5f}".format(norm.cdf((90 - mu) / sigma) - norm.cdf((80 - mu) / sigma)))

plt.title("Normal Distribution")
plt.xlabel("x")
plt.ylabel("P{X = x}")
plt.legend()
plt.grid(True)
plt.show()
