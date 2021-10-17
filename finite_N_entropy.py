# Calculate the energy and entropy of the disorder averaged SYK model at finite N

from SYK_functions import *
import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
import time

# Set variables for run
N = 14  # Number Majorana fermions (N must be even and at least 4)
temp = 1/np.linspace(1, 100, 1000)  # Range of temperatures
realisations = 20  # Number of realisations of model to average over
J_squared = 1  # Variance of couplings is given by 'J_squared * 3!/N^3'

start_all = time.time()

# Create fermions and Hamiltonian
psi = create_majorana_fermions(N)
H = create_Hamiltonian(psi, N, realisations, J_squared)

# Find the eigenvalues of H
evalues = np.array([la.eig(H[i])[0].real for i in range(len(H))])

# Calculate energy and entropy
EbyN = list()  # <E>/N
SbyN = list()  # S/N

for t in temp:
    # print(t)
    minus_beta = np.array([np.exp(-1/t*evalues[i]) for i in range(len(evalues))])
    Z_beta = np.array([sum(minus_beta[i]) for i in range(len(minus_beta))])
    diff = -np.array([sum(evalues[i] * minus_beta[i]) for i in range(len(evalues))])  # dZ/dbeta
    E = np.mean(-diff * 1/Z_beta)
    EbyN.append(E/N)
    S = np.mean(1/t*(-diff * 1/Z_beta)+np.log(Z_beta))
    SbyN.append(S/N)

# Plot graphs
Energy = dict()
Entropy = dict()
Energy[N] = EbyN
Entropy[N] = SbyN

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(temp, Energy[N], color='blue', label="N="+str(N))
ax2.plot(temp, Entropy[N], color='blue', label="N="+str(N))

ax1.set_xlabel("Temp.")
ax2.set_xlabel("Temp.")

ax1.set_title("<E>/N")
ax2.set_title("<S>/N")

plt.show()

end_all = time.time()
print(end_all-start_all)
