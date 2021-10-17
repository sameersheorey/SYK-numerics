# Calculate the large N two point function by numerically solving the SD equations
# and compare to the analytical solution in the conformal limit (deep IR)

from matplotlib import pyplot as plt
from SYK_functions import *

# Set variables for run
t0 = -100  # -t0 is the inverse temperature β.
dt = 0.01  # Time step - must divide t0 exactly.
J_squared = 1  # Variance of couplings is given by 'J_squared * 3!/N^3'
q = 4  # Even integer ≥4. Number of fermions involved in random interactions in SYK model
iteration_length = 500


# Define initial guess for two point function: usually taken to be free theory two point function
def G_input(x):
    return 1/2 * np.sign(x)


t, G = G_SD(t0, dt, G_input, q, iteration_length)[0:2]

plt.plot(t, G_input(t), color='orange', label="initial guess")
plt.scatter(t, G.real, color='purple', s=1, label="G(τ) from SD")

# Calculate large N IR solution for comparison
b = -t0
a = ((1/(J_squared*np.pi))*(1/2-1/q)*np.tan(np.pi*1/q))**(1/q)

x = np.arange(1.1, b-1, 0.1)
y = a*(np.pi/(b*np.sin((np.pi*x)/b)))**(2/q)

plt.plot(x, y, '--', label="IR solution")
plt.legend()
plt.show()
