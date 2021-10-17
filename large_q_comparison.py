# Calculate the large N two point function for different values of q
# and compare to the large q analytical solution

from matplotlib import pyplot as plt
from SYK_functions import *
from scipy.optimize import *

# Set variables for run
t0 = -100  # -t0 is the inverse temperature β.
dt = 0.01  # Time step - must divide t0 exactly.
J_squared = 1  # Variance of couplings is given by 'J_squared * 3!/N^3'
q_range = np.arange(8, 30, 2)  # Even integers ≥4. Number of fermions involved in random interactions in SYK model
iteration_length = 500


# Define initial guess for two point function: usually taken to be free theory two point function
def G_input(x):
    return 1/2 * np.sign(x)


for q in q_range:
    t, G = G_SD(t0, dt, G_input, q, iteration_length)[0:2]

    plt.plot(t, G_input(t), color='orange')
    plt.scatter(t, G.real, color='purple', s=1)

    # Calculate large q analytical solution for comparison
    # See Maldacena Stanford, Comments on the SYK model section 2.4

    JJ = np.sqrt(q) / (2 ** ((q - 1) / 2))  # 'Curly' J in (2.16) of Maldacena, Stanford

    # nu in (2.19) of Maldacena, Stanford is defined implicitly and so must be solved for numerically
    def f(x):
        return -t0 * JJ - np.pi * x / (np.cos(np.pi * x / 2))


    sol = root_scalar(f, x0=0.5, bracket=[0, 1])
    nu = sol.root

    ex = (np.cos(np.pi * nu / 2) / np.cos(np.pi * nu * ((1 / 2) - (abs(t) / (-t0))))) ** (2 / (q - 1))
    G_large_q = 1 / 2 * np.sign(t) * ex

    plt.plot(t, G_large_q, '--', color='red')
