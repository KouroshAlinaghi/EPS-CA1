import numpy as np
import matplotlib.pyplot as plt 

green = '#57cc99'
red = '#e56b6f'

line_thickness = 3

def binomial(n, p, test_times):
    return np.sum(np.reshape(np.array(np.random.choice(2, n * test_times, p=[1 - p, p])), (test_times, n)), axis=1)

n = 500
m = 5000
exp_in_action = []
exp_in_theory = []
var_in_action = []
var_in_theory = []
for p in range(101):
    p /= 100
    q = 1 - p
    sample = binomial(n, p, m)
    exp_in_action.append(np.mean(sample))
    exp_in_theory.append(n * p)
    var_in_action.append(np.var(sample))
    var_in_theory.append(n * p * q)

plt.title("Expectation")
x_values = range(101)
plt.xlabel("p (%)")
plt.ylabel("E[X]")
plt.plot(x_values, exp_in_action, color=green, label="E[X] in action", linewidth=line_thickness)
plt.plot(x_values, exp_in_theory, color=red, label="E[X] in theory (npq)", linestyle='--', linewidth=line_thickness)
plt.legend()
plt.grid(True)
plt.show()

plt.title("Variance")
plt.xlabel("p (%)")
plt.ylabel("var[X]")
plt.plot(x_values, var_in_action, color=green, label="var[X] in action", linewidth=line_thickness)
plt.plot(x_values, var_in_theory, color=red, label="var[X] in theory (np)", linestyle='--', linewidth=line_thickness)
plt.legend()
plt.grid(True)
plt.show()
