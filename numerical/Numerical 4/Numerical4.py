import numpy as np

def compute_layer_indices(positions, dx, N):
    """
    Compute the indices corresponding to the boundaries between layers.
    
    Parameters:
        positions (np.array): Positions of layer interfaces.
        dx (float): Spatial discretization step.
        N (int): Number of spatial discretization points.
        
    Returns:
        indices (np.array): Array of indices representing the boundaries between layers.
    """
    return (positions / dx).astype(int)

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
    alphas = np.array([])
    for diffusion_coefficient in diffusion_coefficients:
        alpha = (diffusion_coefficient * dt) / (dx**2)
        alphas = np.append(alphas, alpha)

    return alphas

# def compute_alphas_vector(layer_indices, alphas, N):
#     """
#     Compute the vector of alpha values for each spatial point based on layer indices.

#     Parameters:
#         layer_indices (np.array): Indices of layer interfaces.
#         alphas (np.array): Alpha values for each layer.
#         N (int): Number of spatial points.

#     Returns:
#         alpha_vector (np.array): Vector of alpha values of size (N+1).
#     """
#     alpha_vector = np.zeros(N + 1)
    
#     if len(layer_indices) == 0:

#         alpha_vector[:] = alphas[0]
#         return alpha_vector
    

#     current_layer = 0
#     for n in range(N + 1):
#         while current_layer < len(layer_indices) and n >= layer_indices[current_layer]:
#             current_layer += 1
        
#         alpha_vector[n] = alphas[current_layer]

#     return alpha_vector


def compute_alphas_vector(layer_indices, alphas, N):
    """
    Compute the vector of alpha values for each spatial point based on layer indices.

    Parameters:
        layer_indices (np.array): Indices of layer interfaces.
        alphas (np.array): Alpha values for each layer.
        N (int): Number of spatial points.

    Returns:
        alpha_vector (np.array): Vector of alpha values of size (N+1).
    """
    alpha_vector = np.zeros(N + 1)
    
    if len(layer_indices) == 0:

        alpha_vector[:] = alphas[0]
        return alpha_vector
    

    current_layer = 0
    for n in range(N + 1):
        while current_layer < len(layer_indices) and n >= layer_indices[current_layer]:
            current_layer += 1
        
        alpha_vector[n] = alphas[current_layer]

    return alpha_vector

# def compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx=50, Ny=50, M=50000):
#     """
#     Compute the concentration matrix for a 2D diffusion problem with multiple layers.

#     This is the first version, it considers a an initial concentration as the drug injected.
    
#     Parameters:
#         diffusion_coeffs_x (np.array): Diffusion coefficients in the x-direction.
#         diffusion_coeffs_y (np.array): Diffusion coefficients in the y-direction.
#         initial_concentration (float): Initial drug concentration in the defined region.
#         layer_x_positions (np.array): Positions of layer interfaces in x-direction.
#         layer_y_positions (np.array): Positions of layer interfaces in y-direction.
#         Lmax (float): Maximum length in x-direction.
#         Hmax (float): Maximum height in y-direction.
#         w1, w2 (float): Range of y-values where the initial concentration is applied.
#         duration (float): Total simulation time.
#         Nx, Ny (int): Number of spatial discretization points.
#         M (int): Number of temporal discretization points.
        
#     Returns:
#         C (np.array): 3D array representing concentration over time and space.
#     """

#     # Compute d's
#     dx = Lmax / Nx
#     dy = Wmax / Ny
#     dt = duration / M

#     # Compute layer indices
#     layer_x_indices = compute_layer_indices(layer_x_positions, dx, Nx)
#     layer_y_indices = compute_layer_indices(layer_y_positions, dy, Ny)

#     print(layer_x_indices)

#     # Compute alphas 
#     alphas_vector_x = compute_alphas_vector(layer_x_indices, compute_alphas(diffusion_coeffs_x, dx, dt), Nx)
#     alphas_vector_y = compute_alphas_vector(layer_y_indices, compute_alphas(diffusion_coeffs_y, dy, dt), Ny)

#     print("Max alpha_x:", np.max(alphas_vector_x))
#     print("Max alpha_y:", np.max(alphas_vector_y))
#     print("Sum:", np.max(alphas_vector_x) + np.max(alphas_vector_y))
#     print("Number of iterations that will be needed :", M)

#     if max(alphas_vector_x) + max(alphas_vector_y) > 1:
#         print("The scheme is unstable")
#         return


#     # Initialize concentration matrix (time, x, y)
#     C = np.zeros((M + 1, Nx + 1, Ny + 1))
    
#     # Initial condition
#     for i in range(Nx + 1):
#         for j in range(Ny + 1):
#             if i * dx <= needle_lenght and w1 <= j * dy <= w2:
#                 C[0, i, j] = initial_concentration
    
#     # Time-stepping loop
#     for m in range(M):
#         for i in range(1, Nx):
#             for j in range(1, Ny):
#                 C[m + 1, i, j] = C[m, i, j] + alphas_vector_x[i] * (C[m, i + 1, j] - 2 * C[m, i, j] + C[m, i - 1, j]) + alphas_vector_y[j] * (C[m, i, j + 1] - 2 * C[m, i, j] + C[m, i, j - 1])
        
#         # Boundary conditions
#         C[m + 1, 0, :] = C[m + 1, 1, :] # Flux nul à x=0
#         C[m + 1, :, 0] = C[m + 1, :, 1] # Flux nul à y=0
#         C[m + 1, :, -1] = C[m + 1, :, -2] # Flux nul à y=Wmax
#         C[m + 1, -1, :] = 0  # Concentration nulle à x=Lmax
        
#         # Interface conditions for continuity of concentration and flux
#         for k in layer_x_indices:
#             for j in range(Ny + 1):
#                 D_left = diffusion_coeffs_x[np.searchsorted(layer_x_positions, k) - 1]
#                 D_right = diffusion_coeffs_x[np.searchsorted(layer_x_positions, k)]
#                 C[m+1, k, j] = (D_left * C[m+1, k-1, j] + D_right * C[m+1, k+1, j]) / (D_left + D_right)

#         if m % 1000 == 0 :
#             print(m)

#     return C



def compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_length, Lmax, Wmax, w1, w2, duration, Nx=50, Ny=50, M=50000):
    """
    Compute the concentration matrix for a 2D diffusion problem with multiple layers.
    
    Parameters:
        diffusion_coeffs_x (np.array): Diffusion coefficients in the x-direction.
        diffusion_coeffs_y (np.array): Diffusion coefficients in the y-direction.
        initial_concentration (float): Initial drug concentration in the defined region.
        layer_x_positions (np.array): Positions of layer interfaces in x-direction.
        layer_y_positions (np.array): Positions of layer interfaces in y-direction.
        Lmax (float): Maximum length in x-direction.
        Wmax (float): Maximum width in y-direction.
        w1, w2 (float): Range of y-values where the initial concentration is applied.
        duration (float): Total simulation time.
        Nx, Ny (int): Number of spatial discretization points.
        M (int): Number of temporal discretization points.
        
    Returns:
        C (np.array): 3D array representing concentration over time and space.
    """

    # Discrétisation de l'espace et du temps
    dx = Lmax / Nx
    dy = Wmax / Ny
    dt = duration / M

    # Indices des interfaces des couches
    layer_x_indices = compute_layer_indices(layer_x_positions, dx, Nx)
    layer_y_indices = compute_layer_indices(layer_y_positions, dy, Ny)
    # Calcul des coefficients de diffusion normalisés
    alphas_vector_x = compute_alphas_vector(layer_x_indices, compute_alphas(diffusion_coeffs_x, dx, dt), Nx)
    alphas_vector_y = compute_alphas_vector(layer_y_indices, compute_alphas(diffusion_coeffs_y, dy, dt), Ny)

    # Vérification de la stabilité
    print(np.max(alphas_vector_x) + np.max(alphas_vector_x))
    if np.max(alphas_vector_x) + np.max(alphas_vector_y) > 0.5:
        print("Le schéma est instable, veuillez ajuster les paramètres.")
        return None

    # Initialisation de la matrice de concentration
    C = np.zeros((M + 1, Nx + 1, Ny + 1))
    
    # Condition initiale
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if i * dx <= needle_length and w1 <= j * dy <= w2:
                C[0, i, j] = initial_concentration
    
    # Boucle temporelle
    for m in range(M):
        for i in range(1, Nx):
            for j in range(1, Ny):
                C[m + 1, i, j] = C[m, i, j] + alphas_vector_x[i] * (C[m, i + 1, j] - 2 * C[m, i, j] + C[m, i - 1, j]) + alphas_vector_y[j] * (C[m, i, j + 1] - 2 * C[m, i, j] + C[m, i, j - 1])
        
        # Conditions aux bords (flux nul sauf à x = Lmax où la concentration est nulle)
        C[m + 1, 0, :] = C[m + 1, 1, :]  # Flux nul à x=0
        C[m + 1, :, 0] = C[m + 1, :, 1]  # Flux nul à y=0
        C[m + 1, :, -1] = C[m + 1, :, -2]  # Flux nul à y=Wmax
        C[m + 1, -1, :] = 0  # Concentration nulle à x=Lmax
        
        for k in layer_x_indices:
            for j in range(Ny + 1):
                x_k = k * dx
                D_left_idx = np.searchsorted(layer_x_positions, x_k) - 1
                D_right_idx = D_left_idx + 1
                D_left = diffusion_coeffs_x[D_left_idx]
                D_right = diffusion_coeffs_x[D_right_idx]

                C[m+1, k, j] = (D_left * C[m+1, k-1, j] + D_right * C[m+1, k+1, j]) / (D_left + D_right)


        if m % 1000 == 0:
            print(f"Iteration {m}/{M}")

    return C


def compute_C2_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_length, Lmax, Wmax, w1, w2, duration, needle_flux, t_start, t_duration, Nx=50, Ny=50, M=50000):
    """
    Compute the concentration matrix for a 2D diffusion problem with multiple layers.

    This is the second version, it considers a constant flux of drug injected as injection.
    
    Parameters:
        diffusion_coeffs_x (np.array): Diffusion coefficients in the x-direction.
        diffusion_coeffs_y (np.array): Diffusion coefficients in the y-direction.
        initial_concentration (float): Initial drug concentration in the defined region.
        layer_x_positions (np.array): Positions of layer interfaces in x-direction.
        layer_y_positions (np.array): Positions of layer interfaces in y-direction.
        Lmax (float): Maximum length in x-direction.
        Hmax (float): Maximum height in y-direction.
        w1, w2 (float): Range of y-values where the initial concentration is applied.
        duration (float): Total simulation time.
        Nx, Ny (int): Number of spatial discretization points.
        M (int): Number of temporal discretization points.
        
    Returns:
        C (np.array): 3D array representing concentration over time and space.
    """

    # Compute d's
    dx = Lmax / Nx
    dy = Wmax / Ny
    dt = duration / M

    # Compute layer indices
    layer_x_indices = compute_layer_indices(layer_x_positions, dx, Nx)
    layer_y_indices = compute_layer_indices(layer_y_positions, dy, Ny)

    # Compute alphas 
    alphas_vector_x = compute_alphas_vector(layer_x_indices, compute_alphas(diffusion_coeffs_x, dx, dt), Nx)
    alphas_vector_y = compute_alphas_vector(layer_y_indices, compute_alphas(diffusion_coeffs_y, dy, dt), Ny)


    print("Max alpha_x:", np.max(alphas_vector_x))
    print("Max alpha_y:", np.max(alphas_vector_y))
    print("Sum:", np.max(alphas_vector_x) + np.max(alphas_vector_y))
    print("Number of iterations that will be needed :", M)

    # Initialize concentration matrix (time, x, y)
    C = np.zeros((M + 1, Nx + 1, Ny + 1))
    
    # Initial condition
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            if i * dx <= needle_length and w1 <= j * dy <= w2:
                C[0, i, j] = initial_concentration

    # Compute the index of the start and duration of the injection
    t_start_idx = int(t_start / dt)
    t_end_idx = int((t_start + t_duration) / dt)
    L_idx = int(round(needle_length / dx))

    # Time-stepping loop
    for m in range(M):
        for i in range(1, Nx):
            for j in range(1, Ny):
                C[m + 1, i, j] = C[m, i, j] + alphas_vector_x[i] * (C[m, i + 1, j] - 2 * C[m, i, j] + C[m, i - 1, j]) + alphas_vector_y[j] * (C[m, i, j + 1] - 2 * C[m, i, j] + C[m, i, j - 1])
        
        # Boundary conditions
        C[m + 1, 0, :] = C[m + 1, 1, :] # Flux nul à x=0
        C[m + 1, :, 0] = C[m + 1, :, 1] # Flux nul à y=0
        C[m + 1, :, -1] = C[m + 1, :, -2] # Flux nul à y=Wmax
        C[m + 1, -1, :] = 0  # Concentration nulle à x=Lmax

        # Drug injection in the skin 
        if t_start_idx <= m <= t_end_idx:
            for j in range(Ny + 1):
                if round(w1 / dy) <= j <= round(w2 / dy):
                    D_left = diffusion_coeffs_x[0]
                    C[m + 1, L_idx, j] = C[m + 1, L_idx + 1, j] + (needle_flux * dx / D_left)


        # Interface conditions for continuity of concentration and flux
        for k in layer_x_indices:
            for j in range(Ny + 1):
                x_k = k * dx
                D_left_idx = np.searchsorted(layer_x_positions, x_k) - 1
                D_right_idx = D_left_idx + 1
                D_left = diffusion_coeffs_x[D_left_idx]
                D_right = diffusion_coeffs_x[D_right_idx]

                C[m+1, k, j] = (D_left * C[m+1, k-1, j] + D_right * C[m+1, k+1, j]) / (D_left + D_right)

        if m % 1000 == 0 :
            print(m)
    
    return C



def compute_initial_quantity(C_matrix, dx, dy):
    return np.sum(C_matrix[0, :, :]) * dx * dy

def compute_final_quantity(C_matrix, dx, dy):
    return np.sum(C_matrix[-1, :, :]) * dx * dy

def compute_i_quantity(C_matrix, dx, dy, i):
    return np.sum(C_matrix[i, :, :]) * dx * dy

def compute_final_delivered_quantity(C_matrix, dx, dy):
    return compute_initial_quantity(C_matrix, dx, dy) - compute_final_quantity(C_matrix, dx, dy)

def compute_i_delivered_quantity(C_matrix, dx, dy, i):
    return compute_initial_quantity(C_matrix, dx, dy) - compute_i_quantity(C_matrix, dx, dy, i)


def compute_initial_quantity_x(C_matrix, dy, x):
    return np.sum(C_matrix[0, x, :]) * dy

def compute_final_x(C_matrix,dy, x):
    return np.sum(C_matrix[-1, x, :]) * dy

def compute_final_delivered_quantity_x(C_matrix, dy, x):
    return compute_initial_quantity_x(C_matrix, dy, x) - compute_final_x(C_matrix, dy, x)

def compute_concentration_x(C_matrix, dy, x):
    return np.sum(C_matrix[-1, x, :]) * dy