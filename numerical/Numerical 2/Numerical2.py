import numpy as np
import matplotlib.pyplot as plt

def compute_C2_matrix(diffusion_coefficient, initial_concentration, needle_lenght, layer_lenght, duration, N=100, M=1000000):

    '''
    Goal : Compute the C1 matrix representing the values of the concentration function for a discrete set of x (position) and t (time)
    The C1 matrix is based on the initial conditions of the modelisation 1.
    Parameters : diffusion_coefficient, initial_concentration, needle_lenght, duration, N and M (the size of C1)
    Return : The C1 matrix (np.array)
    '''

    #Initialization of usefull components
    dx = layer_lenght / N
    dt = duration / M
    alpha = (diffusion_coefficient * dt) / (dx**2)

    print('\u03B1 = ', alpha)

    x_set = np.linspace(0, layer_lenght, N+1)
    t_set = np.linspace(0, duration, M+1)

    C = np.zeros((M+1, N+1))

    nb_of_dx_in_needle_lenght = int(needle_lenght // dx)
    C[0, 0:nb_of_dx_in_needle_lenght] = initial_concentration

    for m in range(M):
        for n in range(1, N):
            C[m+1, n] = C[m, n] + alpha * (C[m, n+1] - 2*C[m, n] + C[m, n-1])
        C[m+1, 0] = C[m+1, 1]
        C[m, -1] = 0

    return C

def compute_C2_initial(initial_concentration, needle_lenght, layer_lenght, N=100):
    dx = layer_lenght / N
    x_set = np.linspace(0, layer_lenght, N+1)
    C_initial = np.zeros(N+1)
    nb_of_dx_in_needle_lenght = int(needle_lenght // dx)
    C_initial[0:nb_of_dx_in_needle_lenght] = initial_concentration

    return C_initial

def experiment_0(diffusion_coefficient, initial_concentration, needle_lenght, layer_lenght, duration, N=100, M=2000):
    print('Starting experiment 0...')

    C = compute_C2_matrix(diffusion_coefficient, initial_concentration, needle_lenght, layer_lenght, duration, N, M)

    x = np.linspace(0, layer_lenght, N+1)

    plt.figure()
    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.plot(x, C[-1, :], label=f't = {duration /60}')
    plt.plot(x, compute_C2_initial(initial_concentration, needle_lenght, layer_lenght, N), label=f't = 0', color='black', linestyle='--', linewidth=0.5)
    plt.xlim(0, layer_lenght+0.0001)
    plt.legend()
    plt.grid()
    plt.savefig('numerical2_experiment0')
    print('Experiment 0 successful, plot saved')

def experiment_1(diffusion_coefficient, initial_concentration, needle_lenght, layer_lenght, duration_values, N=100, M=5000):
    print('Starting experiment 1...')
    x = np.linspace(0, layer_lenght, N+1)

    plt.figure()

    for duration in duration_values:
        C = compute_C2_matrix(diffusion_coefficient, initial_concentration, needle_lenght, layer_lenght, duration, N, M)
        plt.plot(x, C[-1, :], label=f't = {duration / 60}')

    plt.plot(x, compute_C2_initial(initial_concentration, needle_lenght, layer_lenght, N), label=f't = 0', color='black', linestyle='--', linewidth=0.5)
    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.xlim(0, layer_lenght+0.0001)
    plt.ylim(0, initial_concentration+(initial_concentration/10))
    plt.legend()
    plt.grid()
    plt.savefig('numerical2_experiment1')
    print('Experiment 1 successful, plot saved')

def experiment_2(diffusion_coefficient_values, initial_concentration, needle_lenght, layer_lenght, duration, N=100, M=4000):
    print('Starting experiment 2...')

    x = np.linspace(0, layer_lenght, N+1)

    plt.figure()

    for diffusion_coefficient in diffusion_coefficient_values:
        C = compute_C2_matrix(diffusion_coefficient, initial_concentration, needle_lenght, layer_lenght, duration, N, M)
        plt.plot(x, C[-1, :], label=f'D = {diffusion_coefficient}')

    plt.plot(x, compute_C2_initial(initial_concentration, needle_lenght, layer_lenght, N), label=f't = 0', color='black', linestyle='--', linewidth=0.5)
    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.xlim(0, layer_lenght+0.0001)
    plt.ylim(0, initial_concentration+(initial_concentration/10))
    plt.legend()
    plt.grid()
    plt.savefig('numerical2_experiment2')
    print('Experiment 2 successful, plot saved')

def experiment_3(diffusion_coefficient, initial_concentration_values, needle_lenght, layer_lenght, duration, N=100, M=2000):
    print('Starting experiment 3...')

    x = np.linspace(0, layer_lenght, N+1)

    plt.figure()

    for initial_concentration in initial_concentration_values:

        color = next(plt.gca()._get_lines.prop_cycler)['color'] #same color for both plots
        
        C = compute_C2_matrix(diffusion_coefficient, initial_concentration, needle_lenght, layer_lenght, duration, N, M)
        plt.plot(x, C[-1, :], label=fr'$C_0 = {initial_concentration}$', color=color)
        
        plt.plot(x, compute_C2_initial(initial_concentration, needle_lenght, layer_lenght, N), label=fr'$t = 0, C_0 = {initial_concentration}$', color=color, linestyle='--', linewidth=0.5)

    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.xlim(0, layer_lenght+0.0001)
    plt.ylim(0, max(initial_concentration_values) + 0.1 * max(initial_concentration_values))  # Ajuste les limites
    plt.legend(fontsize=8)
    plt.grid()
    plt.savefig('numerical2_experiment3')
    print('Experiment 3 successful, plot saved')


def experiment_4(diffusion_coefficient, initial_concentration, needle_lenght_values, layer_lenght, duration, N=100, M=2000):
    print('Starting experiment 4...')

    plt.figure()

    for L in needle_lenght_values:

        color = next(plt.gca()._get_lines.prop_cycler)['color']

        C = compute_C2_matrix(diffusion_coefficient, initial_concentration, L, layer_lenght, duration, N, M)
        x = np.linspace(0, layer_lenght, N+1)
        plt.plot(x, C[-1, :], label=f'L = {L}', color=color)

        plt.plot(x, compute_C2_initial(initial_concentration, L, layer_lenght, N), label=f't = 0, L = {L}cm', color=color, linestyle='--', linewidth=0.5)

    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.xlim(0, layer_lenght)
    plt.ylim(0, initial_concentration + (initial_concentration / 10))
    plt.legend(fontsize=8)
    plt.grid()
    plt.savefig('numerical2_experiment4')
    print('Experiment 4 successful, plot saved')

def experiment_5(diffusion_coefficient_values, initial_concentration, needle_lenght, layer_lenght, duration, N=100, M=20000):
    print('Starting experiment 5...')

    plt.figure()

    t = np.linspace(0, duration, 500)

    for diffusion_coefficient in diffusion_coefficient_values:
        C = compute_C2_matrix(diffusion_coefficient, initial_concentration, needle_lenght, layer_lenght, duration, N, M)

        C_0_t = C[:, 0]
        sampled_C_0_t = np.interp(t, np.linspace(0, duration, M + 1), C_0_t)
        plt.plot(t / 60, (initial_concentration - sampled_C_0_t) / initial_concentration, label=f'D = {diffusion_coefficient}')

    plt.title('Transdermal drug delivery')
    plt.xlabel('t (h)')
    plt.ylabel(f'C_0 - C(0,t) / C_0')
    plt.xlim(0, duration / 60)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.savefig('numerical2_experiment5')
    print('Experiment 5 sucessful, plot saved')


D = 1e-7
C0 = 15.8
L = 0.015
l = 0.1
t = 24 * 60
t2 = 100 * 60

t_values = np.array([24, 48, 72]) * 60
D_values = np.array([5e-9, 1e-8, 5e-8, 1e-7])
C0_values = np.array([13.8, 14.8, 15.8, 16.8])
L_values = np.array([0.01, 0.015, 0.02, 0.025])

experiment_0(D, C0, L, l, t)
experiment_1(D, C0, L, l, t_values)
experiment_2(D_values, C0, L, l, t)
experiment_3(D, C0_values, L, l, t)
experiment_4(D, C0, L_values, l, t)
experiment_5(D_values, C0, L, l, t2)