import numpy as np
import matplotlib.pyplot as plt
import Numerical3 as num3

def experiment_0(diffusion_coefficients, initial_concentration, needle_length, layer_lengths, duration, N=100, M=10000, name='figure'):
    print('Starting experiment 0...')

    C3, C_initial = num3.compute_C3_matrix(diffusion_coefficients, initial_concentration, layer_lengths, needle_length, duration, N, M)

    total_length = np.sum(layer_lengths)
    x = np.linspace(0, total_length, N+1)

    dx = total_length / N
    plt.figure()
    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.plot(x, C3[-1, :], label=f't = {duration /60}')
    plt.plot(x, C_initial, label=f't = 0', color='black', linestyle='--', linewidth=0.5)
    plt.xlim(0, total_length+0.0001)
    plt.legend()
    plt.grid()
    plt.savefig(name)
    print('Experiment 0 successful, plot saved')

def experiment_1(diffusion_coefficients, initial_concentration, needle_length, layer_lengths, durations, N=100, M=10000, name='figure'):

    print('Starting experiment 1...')

    total_length = np.sum(layer_lengths)
    x = np.linspace(0, total_length, N+1)

    plt.figure()

    for duration in durations:
        C3, C_initial = num3.compute_C3_matrix(diffusion_coefficients, initial_concentration, layer_lengths, needle_length, duration, N, M)
        plt.plot(x, C3[-1, :], label=f't = {duration / 60}')

    plt.plot(x, C_initial, label=f't = 0', color='black', linestyle='--', linewidth=0.5)
    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.xlim(0, total_length+0.0001)
    plt.ylim(0, initial_concentration+(initial_concentration/10))
    plt.legend()
    plt.grid()
    plt.savefig(name)
    print('Experiment 1 successful, plot saved')

def experiment_2(diffusion_coefficients_values, initial_concentration, needle_length, layer_lengths, duration, N=100, M=10000, name='figure'):
    print('Starting experiment 2...')

    x = np.linspace(0, np.sum(layer_lengths), N+1)

    plt.figure()

    for diffusion_coefficients in diffusion_coefficients_values:
        C, C_initial = num3.compute_C3_matrix(diffusion_coefficients, initial_concentration, layer_lengths, needle_length, duration, N, M)
        plt.plot(x, C[-1, :], label=f'D = {diffusion_coefficients}')

    plt.plot(x, C_initial, label=f't = 0', color='black', linestyle='--', linewidth=0.5)
    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.xlim(0, np.sum(layer_lengths)+0.0001)
    plt.ylim(0, initial_concentration+(initial_concentration/10))
    plt.legend()
    plt.grid()
    plt.savefig(name)
    print('Experiment 2 successful, plot saved')

def experiment_3(diffusion_coefficient, initial_concentration_values, needle_lenght, layer_lengths, duration, N=100, M=10000, name='figure'):
    print('Starting experiment 3...')

    x = np.linspace(0, np.sum(layer_lengths), N+1)

    plt.figure()

    for initial_concentration in initial_concentration_values:

        color = next(plt.gca()._get_lines.prop_cycler)['color'] #same color for both plots
        
        C, C_initial = num3.compute_C3_matrix(diffusion_coefficient, initial_concentration, layer_lengths, needle_lenght, duration, N, M)
        plt.plot(x, C[-1, :], label=fr'$C_0 = {initial_concentration}$', color=color)
        
        plt.plot(x, C_initial, label=fr'$t = 0, C_0 = {initial_concentration}$', color=color, linestyle='--', linewidth=0.5)

    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.xlim(0, np.sum(layer_lengths)+0.0001)
    plt.ylim(0, max(initial_concentration_values) + 0.1 * max(initial_concentration_values))  # Ajuste les limites
    plt.legend(fontsize=8)
    plt.grid()
    plt.savefig(name)
    print('Experiment 3 successful, plot saved')


def experiment_4(diffusion_coefficient, initial_concentration, needle_lenght_values, layer_lengths, duration, N=100, M=10000, name='figure'):
    print('Starting experiment 4...')

    plt.figure()

    for L in needle_lenght_values:

        color = next(plt.gca()._get_lines.prop_cycler)['color']

        C, C_initial = num3.compute_C3_matrix(diffusion_coefficient, initial_concentration, layer_lengths, L, duration, N, M)
        x = np.linspace(0, np.sum(layer_lengths), N+1)
        plt.plot(x, C[-1, :], label=f'L = {L}', color=color)

        plt.plot(x, C_initial, label=f't = 0, L = {L}cm', color=color, linestyle='--', linewidth=0.5)

    plt.title('Transdermal drug delivery')
    plt.xlabel('x (cm)')
    plt.ylabel('Concentration (mg/ml)')
    plt.xlim(0, np.sum(layer_lengths))
    plt.ylim(0, initial_concentration + (initial_concentration / 10))
    plt.legend(fontsize=8)
    plt.grid()
    plt.savefig(name)
    print('Experiment 4 successful, plot saved')

def experiment_5(diffusion_coefficient_values, initial_concentration, needle_lenght, layer_lengths, duration, N=100, M=10000, name='figure'):
    print('Starting experiment 5...')

    plt.figure()

    t = np.linspace(0, duration, 500)

    for diffusion_coefficient in diffusion_coefficient_values:
        C, C_initial = num3.compute_C3_matrix(diffusion_coefficient, initial_concentration, layer_lengths, needle_lenght, duration, N, M)

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
    plt.savefig(name)
    print('Experiment 5 sucessful, plot saved')




Ds = np.array([1e-7, 5e-8])
Ds_values = np.array([[1e-7, 1e-8], [1e-7, 5e-8], [1e-7, 1e-7], [1e-7, 5e-7]])
Ds_values2 = np.array([[1e-8, 1e-7], [5e-8, 1e-7], [1e-7, 1e-7], [5e-7, 1e-7]])

C0 = 15.8
C0_values = np.array([13.8, 14.8, 15.8, 16.8])

L = 0.015
L_values = np.array([0.01, 0.015, 0.02, 0.025])

ls = np.array([0.03, 0.04])

t = 48 * 60
ts = np.array([12, 24, 48, 72]) * 60


experiment_0(Ds, C0, L, ls, t, 100, 1500, 'numerical3_experiment0')
experiment_1(Ds, C0, L, ls, ts, 100, 2000, 'numerical3_experiment1')
experiment_2(Ds_values, C0, L, ls, t, 100, 8000, 'numerical3_experiment2')
experiment_2(Ds_values2, C0, L, ls, t, 100, 8000, 'numerical3_experiment2_2')
experiment_3(Ds, C0_values, L, ls, t, 100, 2000, 'numerical3_experiment3')
experiment_4(Ds, C0, L_values, ls, t, 100, 2000, 'numerical3_experiment4')
experiment_5(Ds_values, C0, L, ls, t, 100, 10000, 'numerical3_experiment5')
