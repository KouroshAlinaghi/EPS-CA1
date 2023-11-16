import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import poisson, binom, norm
import math

def clear_plt():
    plt.clf()
    plt.cla()

def binomial(n, p, test_times):
    return np.reshape(np.array(np.random.choice(2, n * test_times, p=[1 - p, p])), (test_times, n))

def draw_bin_exp(action, theory):
    plt.title("Expectation")
    plt.xlabel("p")
    plt.ylabel("E[X]")
    plt.plot(range(101), action, color="green", label="E[X] in action")
    plt.plot(range(101), theory, color="red", label="E[X] in theory (npq)", linestyle='--')
    plt.legend()
    plt.show()

def draw_bin_var(action, theory):
    plt.title("Variance")
    plt.xlabel("p")
    plt.ylabel("var[X]")
    plt.plot(range(101), action, color="green", label="var[X] in action")
    plt.plot(range(101), theory, color="red", label="var[X] in theory (np)", linestyle='--')
    plt.legend()
    plt.show()

def Q1():
    n = 500
    m = 5000
    exp_in_action = []
    exp_in_theory = []
    var_in_action = []
    var_in_theory = []
    for p in range(101):
        p /= 100
        test = binomial(n, p, m)
        exp_in_action.append(np.sum(test) / m)
        exp_in_theory.append(n * p)
        var_in_action.append(np.var(np.sum(test, axis=1)))
        var_in_theory.append(n * p * (1 - p))

    draw_bin_exp(exp_in_action, exp_in_theory)
    clear_plt()
    draw_bin_var(var_in_action, var_in_theory)

def draw_norm(mu, sigma, only_positive = False):
    x = np.arange(0 if only_positive else mu - 3 * sigma, mu + 3 * sigma, 0.01)
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))
    plt.plot(x, y, color="green", label="X ~ N({}, {:.2f})".format(mu, sigma))    

def draw_pois(lamb, l = 0, r = 9):
    x = np.arange(l, r, 1)
    y = poisson.pmf(x, lamb)
    plt.plot(x, y, color="red", label="X ~ Poi({})".format(lamb))

def draw_bin(n, p):
    x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
    plt.plot(x, binom.pmf(x, n, p), color="orange", ms=8, label='X ~ Bin({}, {})'.format(n, p), linestyle='--')

def draw_random_bin(n, p, size):
    data = np.random.binomial(n, p, size)
    num_bins = 50
    counts, bins = np.histogram(data, bins=num_bins)
    bins = bins[:-1] + (bins[1] - bins[0])/2
    probs = counts/float(counts.sum())
    plt.bar(bins, probs, 1.0/num_bins, color="blue", label="Randomized Bin({}, {})".format(n, p))

def Q2():
    n = 250
    p = 0.008
    m = 5000
    draw_norm(n * p, math.sqrt(n * p * (1 - p)), only_positive=True)
    draw_pois(n * p)
    draw_bin(n, p)
    draw_random_bin(n, p, m)

    plt.title("Binomail preidiction with Normal and Poisson distributions")
    plt.xlabel("x")
    plt.ylabel("P(X = x)")
    plt.legend()
    plt.show()

def norm_cdf_bs(mu, sigma, cdf):
    l, r = 0, mu + 2000
    while (r - l > 0.5):
        mid = (l + r) / 2
        if (norm.cdf((mid - mu) / sigma) > cdf):
            r = mid
        else:
            l = mid
    return (r + l) / 2

def Q3():
    mu = 80
    sigma = 12
    min_grade = norm_cdf_bs(mu, sigma, 0.9)

    x = np.arange(mu - 3 * sigma, mu + 3 * sigma, 0.01)
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))
    plt.plot(x, y, color="green", label="X ~ N({}, {})".format(mu, sigma))    
    plt.fill_between(x, y, where=(x < min_grade), color='green', alpha=0.3, label="F(x) = 0.9, x = {:.2f}".format(min_grade))

    second_forth_start = norm_cdf_bs(mu, sigma, 0.25)
    third_forth_end = norm_cdf_bs(mu, sigma, 0.75)
    plt.fill_between(x, y, where=(x < third_forth_end) & (x > second_forth_start), color='red', alpha=0.3, label="0.25 < F(x) < 0.75, {:.2f} < x < {:.2f}".format(second_forth_start, third_forth_end))

    plt.fill_between(x, y, where=(x < 90) & (x > 80), color='blue', alpha=0.3, label="P(80 < X < 90) = {:.5f}".format(norm.cdf((90 - mu) / sigma) - norm.cdf((80 - mu) / sigma)))

    plt.title("Normal Distribution")
    plt.xlabel("x")
    plt.ylabel("P(X = x)")
    plt.legend()
    plt.show()

def Q3_bonus():
    size = 20000

    l, r = 0, 20 + 1
    physics = np.random.randint(l, r, size)
    plt.hist(physics, 21, density=True, color="green", label="Phy ~ U({}, {})".format(l, r - 1), alpha=0.3)

    lamb = 5
    ap = np.random.exponential(lamb, size=size)
    plt.hist(ap, 21, density=True, color="blue", label="AP ~ Exp({})".format(lamb), alpha=0.3)

    lamb = 15
    dm = np.random.poisson(lamb, size=size)
    plt.hist(dm, 21, density=True, color="red", label="DM ~ Poi({})".format(lamb), alpha=0.3)

    plt.hist(np.sum([physics, ap, dm], axis=0), 20, density=True, color="yellow", label="Phy + AP + DM", alpha=0.3)

    plt.title("Sum of Poisson, Exponential and Uniform distributions")
    plt.xlabel("x")
    plt.ylabel("P(X = x)")
    plt.legend()
    plt.show()

def Q4():
    n = 7072
    p = 0.45
    m = 40000
    draw_norm(n * p, math.sqrt(n * p * (1 - p)))
    draw_pois(n * p, l = 3056, r = 3308)
    draw_bin(n, p)

    data = np.random.binomial(n, p, m)
    plt.hist(data, bins=np.arange(n * p - 150, n * p + 150) - 0.5, density=True, color='blue', label="Randomized Bin({}, {})".format(n, p), alpha=0.7)

    plt.title("Binomail prediction with Normal and Poisson distributions")
    plt.xlabel("x")
    plt.ylabel("P(X = x)")
    plt.legend()
    plt.show()

import time

def main():
    std = time.time()
    Q1()
    print(time.time - std)
    clear_plt()
    Q2()
    clear_plt()
    Q3()
    clear_plt()
    Q3_bonus()
    Q4()

if __name__ == "__main__":
    main()
