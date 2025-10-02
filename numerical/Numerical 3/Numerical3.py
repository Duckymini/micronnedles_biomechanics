import numpy as np
import matplotlib.pyplot as plt

def compute_C3_matrix(diffusion_coefficients, initial_concentration, layer_lengths, needle_length, duration, N=100, M=1000000):
    """
    Compute the concentration matrix for a system with multiple layers, each having its own diffusion coefficient and length.
    
    Parameters:
        diffusion_coefficients (np.array): Diffusion coefficients for each layer.
        initial_concentration (float): Initial concentration in the needle region.
        layer_lengths (np.array): Lengths of each layer.
        needle_length (float): Length of the region initially filled with drug.
        duration (float): Total simulation time.
        N (int): Number of spatial discretization points.
        M (int): Number of temporal discretization points.
        
    Returns:
        C (np.array): Matrix representing concentration over time and space.
        C_initial (np.array): Vector representing the concentration over space at t = 0
    """

    total_length = np.sum(layer_lengths)
    dx = total_length / N
    dt = duration / M
    
    # Compute layer indices based on layer lengths and spatial discretization
    layer_indices = compute_layers_indices(layer_lengths, dx, N)
    
    # Compute alpha values for each layer (related to diffusion coefficients)
    alphas = compute_alphas(diffusion_coefficients, dx, dt)

    print('\u03B1s= ', alphas)

    # Compute the alpha_vector containing the right alpha for each indices
    alpha_vector = compute_alphas_vector(layer_indices, alphas, N)
    
    x_set = np.linspace(0, total_length, N + 1) #utile ?
    t_set = np.linspace(0, duration, M + 1) #utile ?

    # Concentration matrix initialization
    C = np.zeros((M + 1, N + 1))
    
    # Concentration initialization (1)
    needle_indices = int(needle_length / total_length * N)
    C[0, :needle_indices] = initial_concentration

    # Store initial concentration for reference
    C_initial = C[0, :]

    for m in range(M):
        for n in range(1, N):
            C[m + 1, n] = C[m, n] + alpha_vector[n] * (C[m, n + 1] - 2 * C[m, n] + C[m, n - 1])
        
        # No flow at x = 0 (2)
        C[m + 1, 0] = C[m + 1, 1]

        # Nul concentration at the end (3)
        C[m + 1, -1] = 0
        
        #Continuity of concentration and flux (4) and (5)
        for k in range(1, len(layer_lengths) - 1):
            C[m+1, k+1] = C[m+1, k]
            C[m+1, k+2] = C[m+1, k]

    return C, C_initial

def compute_layers_indices(layer_lengths, dx, N):
    """
    Compute the indices corresponding to the boundaries between layers.
    
    Parameters:
        layer_lengths (np.array): Lengths of each layer.
        dx (float): Spatial discretization step.
        
    Returns:
        layer_indices (np.array): Array of indices representing the boundaries between layers.
    """

    layer_indices = np.array([])
    current_lenght = 0

    for layer_length in layer_lengths:
        layer_indice = int(current_lenght // dx)
        layer_indices = np.append(layer_indices, layer_indice)
        current_lenght += layer_length
    
    layer_indices = np.append(layer_indices, N)
    return layer_indices.astype(int)

def compute_alphas(diffusion_coefficients, dx, dt):
    """
    Compute the alpha values for each layer, which are used in the finite difference scheme.
    
    Parameters:
        diffusion_coefficients (np.array): Diffusion coefficients for each layer.
        dx (float): Spatial discretization step.
        dt (float): Temporal discretization step.
        
    Returns:
        alphas (np.array): Array of alpha values for each layer.
    """
    return np.array(diffusion_coefficients) * dt / (dx**2)

def compute_alphas_vector(layer_indices, alphas, N):

    alpha_vector = np.zeros(N + 1)

    for n in range(N + 1):
        for k in range(len(layer_indices)):
            if n < layer_indices[k]:
                alpha_vector[n] = alphas[k-1]
                break

    alpha_vector[-1] = alphas[-1]
    return alpha_vector




