# Create Majorana Fermions and SYK Hamiltonian

import numpy as np
from functools import reduce
from scipy.fft import fft, fftfreq, ifft, rfft, irfft, fftshift, ifftshift

def create_majorana_fermions(N):
    """Create Majorana Fermions.

    Creates Majorana Fermions - a set of N Hermitian matrices psi_i, i=1,..N
    obeying anti-commutation relations {psi_i,psi_j} = δ_{ij}

    Args:
        N: An integer denoting number of Majorana fermions.

    Returns:
        A dictionary containing matrix representations of Majorana fermions.
    """
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])

    psi = dict()

    for i in range(1, N+1):
        if (i % 2) == 1:
            matlist = [Z] * int((i-1)/2)
            matlist.append(X)
            matlist = matlist + [I] * int((N/2 - (i+1)/2))
            psi[i] = 1/np.sqrt(2)*reduce(np.kron, matlist)
        else:
            matlist = [Z] * int((i - 2) / 2)
            matlist.append(Y)
            matlist = matlist + [I] * int((N/2 - i/2))
            psi[i] = 1/np.sqrt(2)*reduce(np.kron, matlist)
    return psi


def create_Hamiltonian(psi, N, realisations, J_squared=1):
    """Create Majorana Fermions.

    Creates multiple realisations of the SYK Hamiltonian

    Args:
        psi: A dictionary containing matrix representations of Majorana fermions.
        N: An integer denoting number of Majorana fermions.
        realisations: An integer denoting the number of realisations of the Hamiltonian to be created
        J_squared: Variance of couplings is given by 'J_squared * 3!/N^3'. Set to 1 by default.

    Returns:
        An array containing an SYK Hamiltonian for each realisation of the model.
    """

    H = 0
    J = dict()
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            for k in range(j+1, N+1):
                for l in range(k+1, N+1):
                    # print([i, j, k, l])
                    J[i, j, k, l] = np.random.normal(loc=0, scale=np.sqrt(J_squared*np.math.factorial(3))/(N**(3/2)),
                                                     size=realisations)
                    M = psi[i] @ psi[j] @ psi[k] @ psi[l]
                    H = H + np.array([element * M for element in J[i, j, k, l]])
    return H


def G_SD(t0, dt, G_input, q, iteration_length, J_squared=1):
    """Compute large N two point function from Schwinger Dyson equations.

    Solves the Schwinger Dyson equations numerically with an iterative algorithm and using the FFT
    to switch between coordinate and Fourier space,.

    Args:
        t0: A float. -t0 is the inverse temperature β of the model and sets the limits of the time interval
        on which G(t) is evaluated.
        dt: A float that sets the time step.
        G_input: A function used as the initial guess for G.
        q: Even integer ≥2. Number of fermions involved in random interactions in SYK model.
        iteration_length: Integer denoting many iterations to carry out.
        J_squared: Variance of couplings is given by 'J_squared * 3!/N^3'. Set to 1 by default.

    Returns:
        t: Array containing time points on which functions are evaluated on.
        G: Array containing large N two point function.
        w: Array containing points in frequency space.
        S: Array containing sigma field.
        Sf: Array containing fourier transform of sigma field.
    """

    t = np.arange(t0, -t0, dt)  # define time points on which G(t) is evaluated

    # initialize G and sigma fields
    G = G_input(t)
    S = J_squared*(G**(q-1)) + (1**-10)*1j

    # Compute Fourier transform by scipy's FFT function
    Gf = fft(G)
    Gf[::2] = 0
    # frequency normalization factor is 2*np.pi/dt
    w = fftfreq(G.size)*2*np.pi/dt

    w = -w  # convention of sign in exponential of definition of Fourier transform
    #In order to get a discretisation of the continuous Fourier transform
    #we need to multiply g by a phase factor

    phase = 0.5 * dt * np.exp((-complex(0, 1) * t0) * w)

    Gf = Gf * phase

    # Compute Fourier transform by scipy's FFT function
    Sf = fft(S)
    Sf = Sf * phase

    a = 0.5

    for k in range(1, iteration_length):
        Gf_adjustment = np.reciprocal(-1j * w[1::2] - Sf[1::2])
        Gf_adjustment -= Gf[1::2]
        Gf_adjustment *= a
        Gf[1::2] += Gf_adjustment
        diff_new = np.sum(np.abs(Gf_adjustment))
        if k > 1:
            if diff_new > diff:
                a = 0.5 * a
        diff = diff_new

        G = ifft(Gf / phase, t.size)

        S = J_squared * (G ** (q-1))
        Sf = fft(S) * phase

    return t, G, w, S, Sf
