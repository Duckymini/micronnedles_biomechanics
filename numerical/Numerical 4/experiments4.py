import matplotlib.pyplot as plt
import numpy as np
import Numerical4 as num
import time 
from matplotlib.animation import FuncAnimation

def plot_diffusion_result(C, Lmax, Wmax, Nx, Ny, name='figure'):
    """
    Plot the final concentration distribution as a heatmap with x as ordinate and y as abscissa.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(C[-1, :, :], extent=[0, Wmax, 0, Lmax], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Concentration')
    plt.xlabel('y position')
    plt.ylabel('x position')
    plt.title('Final concentration distribution')
    plt.savefig('numerical4_diffusion_result')
    plt.show()

def plot_diffusion_result_i(C, Lmax, Wmax, Nx, Ny, i, name='figure', name2='Final concentration distribution'):
    """
    Plot the final concentration distribution as a heatmap with x as ordinate and y as abscissa.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(C[i, :, :], extent=[0, Wmax, 0, Lmax], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Concentration')
    plt.xlabel('y position')
    plt.ylabel('x position')
    plt.title(name2)
    plt.savefig(name)
    plt.show()

def plot_diffusion_video(C, Lmax, Wmax, Nx, Ny, name='figure', duration=10, frame_skip=50):
    """
    Plot the diffusion process as a video, skipping frames to speed up rendering.
    
    Parameters:
        C (np.array): 3D array of concentration values over time and space.
        Lmax (float): Maximum length in x-direction.
        Wmax (float): Maximum width in y-direction.
        Nx, Ny (int): Number of spatial discretization points.
        name (str): The name of the output file for the video.
        duration (float): Total duration of the video in seconds.
        frame_skip (int): Number of time steps to skip between frames.
    """
    frames = C[::frame_skip]  # Select only some frames to reduce processing time
    num_frames = frames.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(frames[0], extent=[0, Wmax, 0, Lmax], origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=np.max(C))
    ax.set_xlabel('y position')
    ax.set_ylabel('x position')
    ax.set_title('Diffusion Process')
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label('Concentration')

    def update(frame):
        im.set_array(frames[frame])
        im.set_clim(vmin=0, vmax=np.max(frames[frame]))  # Adjust color scale dynamically
        ax.set_title(f'Diffusion Process - Time Step {frame * frame_skip}')
        return [im]
    
    anim = FuncAnimation(fig, update, frames=num_frames, interval=(duration * 1000) / num_frames, blit=True)
    anim.save(name, writer='ffmpeg', fps=num_frames / duration, extra_args=['-vcodec', 'libx264'])
    plt.show()

def plot_initial(C, Lmax, Wmax, Nx, Ny, name='figure'):
    """
    Plot the final concentration distribution as a heatmap with x as ordinate and y as abscissa.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(C[0, :, :], extent=[0, Wmax, 0, Lmax], origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Concentration')
    plt.xlabel('y position')
    plt.ylabel('x position')
    plt.title('Final concentration distribution')
    plt.savefig('numerical4_initial')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_quantity_delivered1(C_matrix, M, dx, dy, duration, name='figure'):
    """
    Plots the delivered quantity as a function of time, with a horizontal red line indicating the initial quantity.

    Parameters:
        C_matrix (list of np.array): List of 3D matrices representing concentration over time and space.
        M (list of int): List of temporal discretization steps for each case.
        dx (float): Spatial discretization step in the x-direction.
        dy (float): Spatial discretization step in the y-direction.
        duration (float): Total simulation time.
        name (str): Name of the saved figure.
    """
    plt.figure(figsize=(8, 6))

    linestyles = [':', '-.', '--', '-']
    labels = [r'$D = 1 \times 10^{-7}$', r'$D = 5 \times 10^{-8}$', r'$D = 1 \times 10^{-8}$', r'$D = 1 \times 10^{-9}$']

    for i in range(len(M)):
        delivered_quantities = []
        dt = duration / M[i]  # Calcul du pas de temps spécifique
        time_values = np.linspace(0, duration, M[i] + 1)  # Axe des temps

        for m in range(M[i] + 1):
            delivered_quantity = num.compute_i_delivered_quantity(C_matrix[i], dx, dy, m)
            delivered_quantities.append(delivered_quantity)

        plt.plot(time_values, delivered_quantities, label=labels[i], color='black', linestyle=linestyles[i])

    # Tracer la ligne de la quantité initiale
    # initial_quantity = num.compute_initial_quantity(C_matrix[3], dx, dy)
    # plt.axhline(y=initial_quantity, color='red', linestyle='-', label='Total quantity injected')
    plt.xlim(0, duration)
    plt.xlabel(r'$t$ $(h)$')
    plt.ylabel(r'$Delivered$ $Quantity$ $(kg)$')

    plt.legend(loc='best').set_draggable(True)
    plt.savefig(name)
    plt.show()

def plot_quantity_delivered2(C_matrix, M, dx, dy, duration, name='figure'):
    """
    Plots the delivered quantity as a function of time, with a horizontal red line indicating the initial quantity.

    Parameters:
        C_matrix (list of np.array): List of 3D matrices representing concentration over time and space.
        M (list of int): List of temporal discretization steps for each case.
        dx (float): Spatial discretization step in the x-direction.
        dy (float): Spatial discretization step in the y-direction.
        duration (float): Total simulation time.
        name (str): Name of the saved figure.
    """
    plt.figure(figsize=(8, 6))

    linestyles = [':', '-.', '--', '-']
    labels = [r'$L = 6 \times 10^{-4}$', r'$L = 6 \times 10^{-4}$', r'$L = 7 \times 10^{-4}$', r'$L = 8 \times 10^{-4}$']

    for i in range(len(M)):
        delivered_quantities = []
        dt = duration / M[i]  # Calcul du pas de temps spécifique
        time_values = np.linspace(0, duration, M[i] + 1)  # Axe des temps

        for m in range(M[i] + 1):
            delivered_quantity = num.compute_i_delivered_quantity(C_matrix[i], dx, dy, m)
            delivered_quantities.append(delivered_quantity)

        plt.plot(time_values, delivered_quantities, label=labels[i], color='black', linestyle=linestyles[i])

    # Tracer la ligne de la quantité initiale
    # initial_quantity = num.compute_initial_quantity(C_matrix[3], dx, dy)
    # plt.axhline(y=initial_quantity, color='red', linestyle='-', label='Total quantity injected')
    plt.xlim(0, duration)
    plt.xlabel(r'$t$ $(h)$')
    plt.ylabel(r'$Delivered$ $Quantity$ $(kg)$')

    plt.legend(loc='best').set_draggable(True)
    plt.savefig(name)
    plt.show()

def plot_quantity_delivered3(C_matrix, M, dx, dy, duration, name='figure'):
    """
    Plots the delivered quantity as a function of time, with a horizontal red line indicating the initial quantity.

    Parameters:
        C_matrix (list of np.array): List of 3D matrices representing concentration over time and space.
        M (list of int): List of temporal discretization steps for each case.
        dx (float): Spatial discretization step in the x-direction.
        dy (float): Spatial discretization step in the y-direction.
        duration (float): Total simulation time.
        name (str): Name of the saved figure.
    """
    plt.figure(figsize=(8, 6))

    linestyles = [':', '-.', '--', '-']
    labels = [r'$C_0 = 13.3$', r'$C_0 = 14.3$', r'$C_0 = = 15.3$', r'$C_0 = 16.3$']

    for i in range(len(M)):
        delivered_quantities = []
        dt = duration / M[i]  # Calcul du pas de temps spécifique
        time_values = np.linspace(0, duration, M[i] + 1)  # Axe des temps

        for m in range(M[i] + 1):
            delivered_quantity = num.compute_i_delivered_quantity(C_matrix[i], dx, dy, m)
            delivered_quantities.append(delivered_quantity)

        plt.plot(time_values, delivered_quantities, label=labels[i], color='black', linestyle=linestyles[i])

    # Tracer la ligne de la quantité initiale
    # initial_quantity = num.compute_initial_quantity(C_matrix[3], dx, dy)
    # plt.axhline(y=initial_quantity, color='red', linestyle='-', label='Total quantity injected')
    plt.xlim(0, duration)
    plt.xlabel(r'$t$ $(h)$')
    plt.ylabel(r'$Delivered$ $Quantity$ $(kg)$')

    plt.legend(loc='best').set_draggable(True)
    plt.savefig(name)
    plt.show()



def plot_concentration_x(C_matrix, Nx, dy, name='figure'):
    """
    Plots the quantity as a function of x.
    
    Parameters:
        C_matrix (np.array): 3D matrix representing the concentration over time and space.
        dx (float): Spatial discretization step in the x-direction.
        dy (float): Spatial discretization step in the y-direction.
        M (int): The number of temporal discretization steps.
    """
    
    quantity_x = []
    
    for i in range(Nx+1):
        concentration = num.compute_concentration_x(C_matrix, dy, i)
        quantity_x.append(concentration)

    plt.figure(figsize=(8, 6))
    plt.plot(range(Nx + 1), quantity_x, label='Concentration', color='blue')
    plt.xlabel('x')
    plt.ylabel('Concentration')
    plt.title('Concentration vs x')
    plt.legend()
    plt.savefig(name)
    plt.show()
    

def save_C_matrix(C, name='C_matrix'):
    """
    Save the concentration matrix to a file.
    
    Parameters:
        C (np.array): 3D matrix representing the concentration over time and space.
        name (str): The name of the file to save the matrix to.
    """
    np.savez(name, C)

Nx=100
Ny=100
dx=1e-5
dy = 1e-5
duration = 24

M1, M2, M3, M4 = 50000, 50000, 50000, 50000
data = np.load('C_C13.3.npz')
C1 = data["arr_0"]
data = np.load('C_C14.3.npz')
C2 = data["arr_0"]
data = np.load('C_C15.3.npz')
C3 = data["arr_0"]
data = np.load('C_C16.3.npz')
C4 = data["arr_0"]
C = [C1, C2, C3, C4]
M = [M1, M2, M3, M4]
plot_quantity_delivered3(C, M, dx, dy, duration, 'figure_8')

# M1, M2, M3, M4 = 100000, 50000, 10000, 10000
# data = np.load('C_D1e-7.npz')
# C1 = data["arr_0"]
# data = np.load('C_D5e-8.npz')
# C2 = data["arr_0"]
# data = np.load('C_D1e-8.npz')
# C3 = data["arr_0"]
# data = np.load('C_D5e-9.npz')
# C4 = data["arr_0"]
# C = [C1, C2, C3, C4]
# M = [M1, M2, M3, M4]
# plot_quantity_delivered1(C, M, dx, dy, duration, 'figure_7')

# M1, M2, M3, M4 = 50000, 50000, 50000, 50000
# data = np.load('C_L5e-4.npz')
# C1 = data["arr_0"]
# data = np.load('C_L6e-4.npz')
# C2 = data["arr_0"]
# data = np.load('C_L7e-4.npz')
# C3 = data["arr_0"]
# data = np.load('C_L8e-4.npz')
# C4 = data["arr_0"]
# C = [C1, C2, C3, C4]
# M = [M1, M2, M3, M4]
# plot_quantity_delivered2(C, M, dx, dy, duration, 'figure_8')

# plot_diffusion_result_i(C2, 1e-3, 1e-3, Nx, Ny, 0, 'figure_6a', 't = 0')
# plot_diffusion_result_i(C2, 1e-3, 1e-3, Nx, Ny, 250, 'figure_6b', 't = 2')
# plot_diffusion_result_i(C2, 1e-3, 1e-3, Nx, Ny, 500, 'figure_6c', 't = 4')
# plot_diffusion_result_i(C2, 1e-3, 1e-3, Nx, Ny, 1000, 'figure_6d', 't = 6')
# plot_diffusion_result_i(C2, 1e-3, 1e-3, Nx, Ny, 2000, 'figure_6e', 't = 8')
# plot_diffusion_result_i(C2, 1e-3, 1e-3, Nx, Ny, 10000, 'figure_6f', 't = 10')





# start_time = time.time()

# Nx, Ny = 100, 100
# Lmax, Wmax = 1e-3, 1e-3
# needle_width = 1e-5
# w1, w2 = Wmax/2 - needle_width/2, Wmax/2 + needle_width/2
# duration = 24 #in hours
# layer_x_positions = np.array([])
# layer_y_positions = np.array([])
# dx = Lmax / Nx
# dy = Wmax / Ny
# video_duration = 10 #in seconds
# needle_flux = 1 
# t_start = 0.001 * 3600
# t_duration = 0.09 * 3600

# #1
# print('Experiment 1')
# M = 100000
# diffusion_coeffs_x = np.array([1e-7]) #in m^2/s
# diffusion_coeffs_y = np.array([1e-7])
# needle_lenght = 7e-4 #in m
# initial_concentration = 15.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_D1e-7')
# print('Experiment 1.1 done')

# M = 50000
# diffusion_coeffs_x = np.array([5e-8]) #in m^2/s
# diffusion_coeffs_y = np.array([5e-8])
# needle_lenght = 7e-4 #in m
# initial_concentration = 15.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_D5e-8')
# print('Experiment 1.2 done')

# M = 10000
# diffusion_coeffs_x = np.array([1e-8]) #in m^2/s
# diffusion_coeffs_y = np.array([1e-8])
# needle_lenght = 7e-4 #in m
# initial_concentration = 15.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_D1e-8')
# print('Experiment 1.3 done')

# M = 10000
# diffusion_coeffs_x = np.array([5e-9]) #in m^2/s
# diffusion_coeffs_y = np.array([5e-9])
# needle_lenght = 7e-4 #in m
# initial_concentration = 15.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_D5e-9')
# print('Experiment 1.4 done')

# #2
# print('Experiment 2')
# M = 50000
# diffusion_coeffs_x = np.array([5e-8]) #in m^2/s
# diffusion_coeffs_y = np.array([5e-8])
# needle_lenght = 5e-4 #in m
# initial_concentration = 15.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_L5e-4')
# print('Experiment 2.1 done')

# M = 50000
# diffusion_coeffs_x = np.array([5e-8]) #in m^2/s
# diffusion_coeffs_y = np.array([5e-8])
# needle_lenght = 6e-4 #in m
# initial_concentration = 15.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_L6e-4')
# print('Experiment 2.2 done')

# M = 50000
# diffusion_coeffs_x = np.array([5e-8]) #in m^2/s
# diffusion_coeffs_y = np.array([5e-8])
# needle_lenght = 7e-4 #in m
# initial_concentration = 15.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_L7e-4')
# print('Experiment 2.3 done')

# M = 50000
# diffusion_coeffs_x = np.array([5e-8]) #in m^2/s
# diffusion_coeffs_y = np.array([5e-8])
# needle_lenght = 8e-4 #in m
# initial_concentration = 15.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_L8e-4')
# print('Experiment 2.4 done')

# #3
# print('Experiment 3')
# M = 50000
# diffusion_coeffs_x = np.array([5e-8]) #in m^2/s
# diffusion_coeffs_y = np.array([5e-8])
# needle_lenght = 7e-4 #in m
# initial_concentration = 13.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_C13.3')
# print('Experiment 3.1 done')

# M = 50000
# diffusion_coeffs_x = np.array([5e-8]) #in m^2/s
# diffusion_coeffs_y = np.array([5e-8])
# needle_lenght = 7e-4 #in m
# initial_concentration = 14.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_C14.3')
# print('Experiment 3.2 done')

# M = 50000
# diffusion_coeffs_x = np.array([5e-8]) #in m^2/s
# diffusion_coeffs_y = np.array([5e-8])
# needle_lenght = 7e-4 #in m
# initial_concentration = 15.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_C15.3')
# print('Experiment 3.3 done')

# M = 50000
# diffusion_coeffs_x = np.array([5e-8]) #in m^2/s
# diffusion_coeffs_y = np.array([5e-8])
# needle_lenght = 7e-4 #in m
# initial_concentration = 16.3 #in mg/ml = kg/m^3

# C = num.compute_C_matrix_2D(diffusion_coeffs_x, diffusion_coeffs_y, initial_concentration, layer_x_positions, layer_y_positions, needle_lenght, Lmax, Wmax, w1, w2, duration, Nx, Ny, M)
# save_C_matrix(C, 'C_C16.3')
# print('Experiment 3.4 done')


# C = np.load('C_matrix.npz')['arr_0']
# plot_diffusion_result(C, Lmax, Wmax, Nx, Ny, 'diffusion 1')
# plot_initial(C, Lmax, Wmax, Nx, Ny, 'initial 1')
# plot_quantity_delivered(C, dx, dy, M, 'delivered 1')
# plot_diffusion_video(C, Lmax, Wmax, Nx, Ny, 'video_1.mp4', video_duration, 40)

# plot_diffusion_result(C2, Lmax, Wmax, Nx, Ny, 'diffusion 2')
# plot_initial(C2, Lmax, Wmax, Nx, Ny, 'initial 2')
# plot_quantity_delivered(C2, dx, dy, M, 'delivered 2')
# plot_diffusion_video(C2, Lmax, Wmax, Nx, Ny, 'video_2.mp4', video_duration)

# end_time = time.time()

# print('Total time to compute : ', end_time - start_time)
