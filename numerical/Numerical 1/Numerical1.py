import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def compute_C1_matrix(diffusion_coefficient, initial_concentration, needle_lenght, duration, N=100, M=1000000):

    '''
    Goal : Compute the C1 matrix representing the values of the concentration function for a discrete set of x (position) and t (time)
    The C1 matrix is based on the initial conditions of the modelisation 1.
    Parameters : diffusion_coefficient, initial_concentration, needle_lenght, duration, N and M (the size of C1)
    Return : The C1 matrix (np.array)
    '''

    #Initialization of usefull components
    dx = needle_lenght / N                              # Spatial step
    dt = duration / M                                   # Time step
    alpha = (diffusion_coefficient * dt) / (dx**2)      # Alpha coefficient

    print('\u03B1 = ', alpha)

    x_set = np.linspace(0, needle_lenght, N+1)          # Spatial grid
    t_set = np.linspace(0, duration, M+1)               # Time grid

    C = np.zeros((M+1, N+1))                            # C matrix initialization (M lines of time and N columns of position)

    C[0, :] = initial_concentration                     # Introduction of the first initial condtions ( C(x,0) = C0 )

    for m in range(M):
        for n in range(1, N): 
            C[m+1, n] = C[m, n] + alpha * (C[m, n+1] - 2*C[m, n] + C[m, n-1])
        C[m+1, 0] = C[m+1, 1]
        C[m, -1] = 0

    return C

def experiment_0(diffusion_coefficient, initial_concentration, needle_lenght, duration, N=100, M=100000):
    print('Starting experiment 0...')

    C = compute_C1_matrix(diffusion_coefficient, initial_concentration, needle_lenght, duration, N, M)

    x = np.linspace(0, needle_lenght, N+1)

    plt.figure()
    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.plot(x, C[-1, :], label=f't = {duration /60}')
    plt.xlim(0, needle_lenght+0.0001)
    plt.legend()
    plt.grid()
    plt.savefig('numerical1_experiment0')
    print('Experiment 0 successful, plot saved')

def experiment_1(diffusion_coefficient, initial_concentration, needle_lenght, duration_values, N=100, M=50000):
    print('Starting experiment 1...')
    x = np.linspace(0, needle_lenght, N+1)

    plt.figure()

    for duration in duration_values:
        C = compute_C1_matrix(diffusion_coefficient, initial_concentration, needle_lenght, duration, N, M)
        plt.plot(x, C[-1, :], label=f't = {duration / 60}')

    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.xlim(0, needle_lenght+0.0001)
    plt.ylim(0, initial_concentration+(initial_concentration/10))
    plt.legend()
    plt.grid()
    plt.savefig('numerical1_experiment1')
    print('Experiment 1 successful, plot saved')

def experiment_2(diffusion_coefficient_values, initial_concentration, needle_lenght, duration, N=100, M=20000):
    print('Starting experiment 2...')

    x = np.linspace(0, needle_lenght, N+1)

    plt.figure()

    C = compute_C1_matrix(diffusion_coefficient_values[3], initial_concentration, needle_lenght, duration, N, M)
    plt.plot(x, C[-1, :], label=r'$D = 5 \times 10^{-9}$', color='black', linestyle='-')

    C = compute_C1_matrix(diffusion_coefficient_values[2], initial_concentration, needle_lenght, duration, N, M)
    plt.plot(x, C[-1, :], label=r'$D = 1 \times 10^{-8}$', color='black', linestyle='--')

    C = compute_C1_matrix(diffusion_coefficient_values[1], initial_concentration, needle_lenght, duration, N, M)
    plt.plot(x, C[-1, :], label=r'$D = 5 \times 10^{-8}$', color='black', linestyle='-.')

    C = compute_C1_matrix(diffusion_coefficient_values[0], initial_concentration, needle_lenght, duration, N, M)
    plt.plot(x, C[-1, :], label= r'$D = 1 \times 10^{-7}$', color='black', linestyle=':')

    plt.xlabel('x (cm)')
    plt.ylabel('Drug concentration (mg/ml)')
    plt.xlim(0, needle_lenght+0.0001)
    plt.ylim(0, initial_concentration+(initial_concentration/10))
    plt.legend()
    plt.grid()
    plt.savefig('figure_2')
    print('Experiment 2 successful, plot saved')

def experiment_3(diffusion_coefficient, initial_concentration_values, needle_lenght, duration, N=100, M=20000):
    print('Starting experiment 3...')

    x = np.linspace(0, needle_lenght, N+1)

    plt.figure()

    C = compute_C1_matrix(diffusion_coefficient, initial_concentration_values[0], needle_lenght, duration, N, M)
    plt.plot(x, C[-1, :], label=r'$C_0 = 13.8$', color='black', linestyle='-')

    C = compute_C1_matrix(diffusion_coefficient, initial_concentration_values[1], needle_lenght, duration, N, M)
    plt.plot(x, C[-1, :], label=r'$C_0 = 14.8$', color='black', linestyle='--')

    C = compute_C1_matrix(diffusion_coefficient, initial_concentration_values[2], needle_lenght, duration, N, M)
    plt.plot(x, C[-1, :], label=r'$C_0 = 15.8$', color='black', linestyle='-.')

    C = compute_C1_matrix(diffusion_coefficient, initial_concentration_values[3], needle_lenght, duration, N, M)
    plt.plot(x, C[-1, :], label=r'$C_0 = 16.8$', color='black', linestyle=':')

    plt.xlabel('x (cm)')
    plt.ylabel('Drug concentration (mg/ml)')
    plt.xlim(0, needle_lenght+0.0001)
    plt.ylim(0, 10)
    plt.legend()
    plt.grid()
    plt.savefig('figure_3')
    print('Experiment 3 successful, plot saved')

def experiment_4(diffusion_coefficient, initial_concentration, needle_lenght_values, duration, N=100, M=20000):
    print('Starting experiment 4...')

    plt.figure()

    C = compute_C1_matrix(diffusion_coefficient, initial_concentration, needle_lenght_values[0], duration, N, M)
    x = np.linspace(0, needle_lenght_values[0], N+1)
    plt.plot(x, C[-1, :], label=r'$L = 0.01$', color='black', linestyle='-')

    C = compute_C1_matrix(diffusion_coefficient, initial_concentration, needle_lenght_values[1], duration, N, M)
    x = np.linspace(0, needle_lenght_values[1], N+1)
    plt.plot(x, C[-1, :], label=r'$L = 0.015$', color='black', linestyle='--')

    C = compute_C1_matrix(diffusion_coefficient, initial_concentration, needle_lenght_values[2], duration, N, M)
    x = np.linspace(0, needle_lenght_values[2], N+1)
    plt.plot(x, C[-1, :], label=r'$L = 0.02$', color='black', linestyle='-.')

    C = compute_C1_matrix(diffusion_coefficient, initial_concentration, needle_lenght_values[3], duration, N, M)
    x = np.linspace(0, needle_lenght_values[3], N+1)
    plt.plot(x, C[-1, :], label=r'$L = 0.025$', color='black', linestyle=':')

    plt.xlabel('x (cm)')
    plt.ylabel('Drug concentration (mg/ml)')
    plt.xlim(0, max(needle_lenght_values))
    plt.ylim(0, 15)
    plt.legend()
    plt.grid()
    plt.savefig('figure_4')
    print('Experiment 4 successful, plot saved')

def experiment_5(diffusion_coefficient_values, initial_concentration, needle_lenght, duration, N=100, M=100000):
    print('Starting experiment 5...')

    plt.figure()

    t = np.linspace(0, duration, 500)

    C = compute_C1_matrix(diffusion_coefficient_values[0], initial_concentration, needle_lenght, duration, N, M)
    C_0_t = C[:, 0]
    sampled_C_0_t = np.interp(t, np.linspace(0, duration, M + 1), C_0_t)
    plt.plot(t / 60, (initial_concentration - sampled_C_0_t) / initial_concentration, label=r'$D = 5 \times 10^{-9}$', color='black', linestyle='-')

    C = compute_C1_matrix(diffusion_coefficient_values[1], initial_concentration, needle_lenght, duration, N, M)
    C_0_t = C[:, 0]
    sampled_C_0_t = np.interp(t, np.linspace(0, duration, M + 1), C_0_t)
    plt.plot(t / 60, (initial_concentration - sampled_C_0_t) / initial_concentration, label=r'$D = 1 \times 10^{-8}$', color='black', linestyle='--')

    C = compute_C1_matrix(diffusion_coefficient_values[2], initial_concentration, needle_lenght, duration, N, M)
    C_0_t = C[:, 0]
    sampled_C_0_t = np.interp(t, np.linspace(0, duration, M + 1), C_0_t)
    plt.plot(t / 60, (initial_concentration - sampled_C_0_t) / initial_concentration, label=r'$D = 5 \times 10^{-8}$', color='black', linestyle='-.')

    C = compute_C1_matrix(diffusion_coefficient_values[3], initial_concentration, needle_lenght, duration, N, M)
    C_0_t = C[:, 0]
    sampled_C_0_t = np.interp(t, np.linspace(0, duration, M + 1), C_0_t)
    plt.plot(t / 60, (initial_concentration - sampled_C_0_t) / initial_concentration, label=r'$D = 1 \times 10^{-7}$', color='black', linestyle=':')

    plt.xlabel(f't (h)')
    plt.ylabel(r'$\frac{C_0 - C(0,t)}{C_0}$')
    plt.xlim(0, duration / 60)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.savefig('figure_5')
    print('Experiment 5 sucessful, plot saved')

def compute_concentration(x, time, diffusion_coefficient, initial_concentration, needle_lenght, N = 3): 

    '''
    Goal : Compute the concentration of drug in the epidermius for a given set of parameters
    Parameters : x (in cm), time (in hours), diffusion_coefficient (in cm^2.mn^-1), initial_concentration (mg.ml^-1), needle_lenght (in cm), N (the number of terms to compute)
    Return : The concentration (in mg.ml^-1)
    '''

    concentration = 0
    ratio_vector = np.array([])

    for n in range(N):
        term = (2 * ((-1)**(2+n)) * initial_concentration * 
        (np.exp(1j * (2*n+1) * np.pi * x / (2*needle_lenght)) + np.exp(-1j * (2*n+1) * np.pi * x / (2*needle_lenght))) 
        * np.exp(-diffusion_coefficient * (2*n+1)**2 * np.pi**2 * time / (4*needle_lenght**2))) / ((2*n + 1) * np.pi)

        concentration += term.real

        print(term.real)

    return concentration

def experiment_1_bis(diffusion_coefficient, initial_concentration, needle_lenght, duration_values, N=100, M=50000):
    print('Starting experiment 1...')
    x = np.linspace(0, needle_lenght, N+1)
    x_ = np.linspace(0, needle_lenght, 7)

    plt.figure()

    for duration in duration_values:
        C = compute_C1_matrix(diffusion_coefficient, initial_concentration, needle_lenght, duration, N, M)
        plt.plot(x, C[-1, :], color='black', linewidth=0.05)
        plt.plot(x_, [compute_concentration(x, duration, diffusion_coefficient, initial_concentration, needle_lenght, N) for x in x_], marker='s', markerfacecolor='none', markeredgewidth=1, color='black')
        if duration == 24 * 60:  # 24h
            plt.text(needle_lenght * 0.22, 9.1, '24h', fontsize=10, color='black', ha='center')
        elif duration == 48 * 60:  # 48h
            plt.text(needle_lenght * 0.3, 4.2, '48h', fontsize=10, color='black', ha='center')
        elif duration == 72 * 60:  # 72h
            plt.text(needle_lenght * 0.4, 1.9, '72h', fontsize=10, color='black', ha='center')

    plt.xlabel('x (cm)')
    plt.ylabel('Drug concentration (mg/ml)')
    plt.xlim(0, needle_lenght+0.0001)
    plt.ylim(0, 14) 
    numerical_solution_handle = Line2D([0], [0], color='black', linewidth=1, label='Numerical solution')
    analytical_solution_handle = Line2D([0], [0], color='black', linewidth=0.001, marker='s', markerfacecolor='none', markeredgewidth=1, markersize=6, label='Analytical solution')
    plt.legend(handles=[numerical_solution_handle, analytical_solution_handle])
    plt.grid()
    plt.savefig('figure_1')
    print('Experiment 1 successful, plot saved')


D = 5e-8
C0 = 15.8
L = 0.015
t = 24 * 60
t2 = 100 * 60

t_values = np.array([24, 48, 72]) * 60
D_values = np.array([5e-9, 1e-8, 5e-8, 1e-7])
C0_values = np.array([13.8, 14.8, 15.8, 16.8])
L_values = np.array([0.01, 0.015, 0.02, 0.025])

# experiment_0(D, C0, L, t)
# experiment_1_bis(D, C0, L, t_values)
# experiment_2(D_values, C0, L, t)
# experiment_3(D, C0_values, L, t)
# experiment_4(D, C0, L_values, t)
experiment_5(D_values, C0, L, t2)