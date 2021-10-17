# Calculate the disorder averaged two point function for finite N

from SYK_functions import *
import numpy as np
from sympy import *
import scipy.linalg as la
from matplotlib import pyplot as plt
import time

start_all = time.time()

# Set variables for run
N = 14  # Number Majorana fermions (N must be even and at least 4)
J_squared = 1  # Variance of couplings is given by 'J_squared * 3!/N^3'
b = 10  # Inverse temperature
points = 101  # Number of points to plot
realisations = 20  # Number of realisations of model to average over

# Create fermions and Hamiltonian
psi = create_majorana_fermions(N)
H = create_Hamiltonian(psi, N, realisations, J_squared)

# Find the eigenvalues of H
evalues = np.array([la.eig(H[i])[0].real for i in range(len(H))])

# Calculate the partition function
minus_beta = np.array([np.exp(-b*evalues[i]) for i in range(len(evalues))])
Z_beta = np.array([sum(minus_beta[i]) for i in range(len(minus_beta))])

# Calculate the Euclidean two point function
# Find diagonalisation matrix for H
P = np.array([la.eig(H[i])[1] for i in range(len(H))])  # the first column is the first eigenvector
Pinv = np.array([la.inv(la.eig(H[i])[1]) for i in range(len(H))])

# Apply P & Pinv to psi, so that we can use diagonalised H in trace of two point function
# This significantly speeds up calculation
for i in range(1, N+1):
    # print(i)
    psi[i] = np.array([Pinv[realisation] @ psi[i] @ P[realisation] for realisation in range(len(P))])

# For each realisation of H, calculate two point function for ith fermion: <psi_i psi_i>
def G(i, tau):
    valid = list(range(1, N+1))
    if i not in valid:
        raise ValueError("results: i must be one of %r." % valid)
    plus_tau = np.array([np.exp(tau*evalues[realisation]) for realisation in range(len(evalues))])
    minus_tau = np.array([np.exp(-tau*evalues[realisation]) for realisation in range(len(evalues))])
    return(np.array([1/Z_beta[realisation] *
                     np.sum((psi[i][realisation] * minus_tau[realisation]) *
                            (psi[i][realisation] * minus_beta[realisation] * plus_tau[realisation]).T)
                     for realisation in range(len(minus_beta))]))


times = np.linspace(0, b, points)

corr = dict()
avcorr = 0

# Average over realisations and average over each fermion to calculate: 1/N * sum_{i} <psi_i psi_i>_{J}
for i in range(1, N+1):
    # print(i)
    corr[i] = np.array([np.mean(G(i, tau).real) for tau in times])  # average over realisations
    avcorr = avcorr + corr[i]

two_point = dict()
two_point[N] = avcorr/N

# Calculate large N IR solution for comparison
a = ((1/(J_squared*np.pi))*(1/2-1/4)*np.tan(np.pi*1/4))**(1/4)
x = np.arange(1.1, b-1, 0.1)
y = a*np.sqrt(np.pi/(b*np.sin((np.pi*x)/b)))

# Plot graphs
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, y, '--', label="Large N IR solution, " + "β=" + str(b))
ax1.plot(times, two_point[N], color='red', label="N=" + str(N) + ", β=" + str(b))
ax2.hist(np.concatenate(evalues), bins=50, label="N="+str(N) + "evalues")

ax1.legend()
ax2.legend(loc='best')
ax1.set_yticks(np.arange(0, 0.52, 0.05))

ax1.set_xlabel("\u03C4", fontsize=12)
ax1.set_ylabel("G(\u03C4)", fontsize=12)

plt.show()

end_all = time.time()
print(end_all-start_all)
