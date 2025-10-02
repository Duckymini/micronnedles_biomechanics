import numpy as np
import matplotlib.pyplot as plt

def compute_concentration(x, time, diffusion_coefficient, initial_concentration, needle_lenght, N = 2): 

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

def experiment_0(time_, diffusion_coefficient, initial_concentration, needle_lenght, N = 2):

    '''
    Goal : Plot the Concentration vs. x for different time values.
    Parameters : time array (in hours), diffusion_coefficient (in cm^2.mn^-1), 
        initial_concentration (mg.ml^-1), needle_lenght (in cm), N (the number of terms to compute)
    Return : Void
    '''

    x_values = np.linspace(0, needle_lenght, 100)

    plt.figure(figsize=(8, 6))

    C_values = [compute_concentration(x, time, diffusion_coefficient, initial_concentration, needle_lenght, N) for x in x_values]
    plt.plot(x_values, C_values, label=f't = {time}')

    plt.title(f'C(x, t) vs. x for different t values', fontsize=14)
    plt.xlabel('x (cm)', fontsize=12)
    plt.ylabel(r'Drug concentration (mg.ml$^{-1}$)', fontsize=12)
    plt.ylim(0, initial_concentration)
    plt.xlim(0, needle_lenght + 0.0001)
    plt.legend()
    plt.grid()
    plt.savefig("analytic_experiment0")


def experiment_1(time_values, diffusion_coefficient, initial_concentration, needle_lenght, N = 2):

    '''
    Goal : Plot the Concentration vs. x for different time values.
    Parameters : time array (in hours), diffusion_coefficient (in cm^2.mn^-1), 
        initial_concentration (mg.ml^-1), needle_lenght (in cm), N (the number of terms to compute)
    Return : Void
    '''

    x_values = np.linspace(0, needle_lenght, 100)

    plt.figure(figsize=(8, 6))

    for time in time_values:
        C_values = [compute_concentration(x, time, diffusion_coefficient, initial_concentration, needle_lenght, N) for x in x_values]
        plt.plot(x_values, C_values, label=f't = {time/60}')

    plt.title(f'C(x, t) vs. x for different t values', fontsize=14)
    plt.xlabel('x (cm)', fontsize=12)
    plt.ylabel(r'Drug concentration (mg.ml$^{-1}$)', fontsize=12)
    plt.ylim(0, initial_concentration)
    plt.xlim(0, needle_lenght + 0.0001)
    plt.legend()
    plt.grid()
    plt.savefig("analytic_experiment1")


def experiment_2(time, diffusion_coefficient_values, initial_concentration, needle_lenght, N = 2):

    '''
    Goal : Plot the Concentration vs. x with different diffusion coefficients
    Parameters : x (an np.array in cm), time (in hours), diffusion_coefficient (an np.array in cm^2.mn^-1), 
        initial_concentration (mg.ml^-1), needle_lenght (in cm), N (the number of terms to compute)
    Return : Void
    '''

    x_values = np.linspace(0, needle_lenght, 100)

    plt.figure(figsize=(8, 6))

    for diffusion_coefficient in diffusion_coefficient_values:
        C_values = [compute_concentration(x, time, diffusion_coefficient, initial_concentration, needle_lenght, N) for x in x_values]
        plt.plot(x_values, C_values, label=f'D = {diffusion_coefficient}')

    plt.title(f'Concentration C(x,t) vs. x for different D values', fontsize=14)
    plt.xlabel('x (cm)', fontsize=12)
    plt.ylabel(r'Drug concentration (mg.ml$^{-1}$)', fontsize=12)
    plt.xlim(0, needle_lenght+0.0001)
    plt.legend()
    plt.grid()
    plt.savefig("analytic_experiment2")


def experiment_3(time, diffusion_coefficient, initial_concentration_values, needle_lenght, N =2 ):

    '''
    Goal : Plot the Concentration vs. x for different initial concentration values. 
    Parameters : time (in hours), diffusion_coefficient (in cm^2.mn^-1), 
        initial_concentration_values (an array in mg.ml^-1), needle_lenght (in cm), N (the number of terms to compute)
    Return : Void
    '''

    x_values = np.linspace(0, needle_lenght, 100)

    plt.figure(figsize=(8, 6))

    for initial_concentration in initial_concentration_values:
        C_values = [compute_concentration(x, time, diffusion_coefficient, initial_concentration, needle_lenght, N) for x in x_values]
        plt.plot(x_values, C_values, label=f'C0 = {initial_concentration}')

    plt.title(f'C(x, t) vs. x for different C0 values', fontsize=14)
    plt.xlabel('x (cm)', fontsize=12)
    plt.ylabel(r'Drug concentration (mg.ml$^{-1}$)', fontsize=12)
    plt.xlim(0, 0.0001 + needle_lenght)
    plt.legend()
    plt.grid()
    plt.savefig("analytic_experiment3")

def experiment_4(time, diffusion_coefficient, initial_concentration, needle_lenght_values, N = 2):

    '''
    Goal : Plot the Concentration vs. x for different values of needle lenght.
    Parameters : time (in hours), diffusion_coefficient (in cm^2.mn^-1), 
        initial_concentration (mg.ml^-1), needle_lenght_values (an array in cm), N (the number of terms to compute)
    Return : Void
    '''

    plt.figure(figsize=(8, 6))

    for needle_lenght in needle_lenght_values:
        x_values = np.linspace(0, needle_lenght, 100)
        C_values = [compute_concentration(x, time, diffusion_coefficient, C0, needle_lenght, N) for x in x_values]
        plt.plot(x_values, C_values, label=f'L = {needle_lenght}')

    plt.title(f'C(x, t) vs. x for different L values', fontsize=14)
    plt.xlabel('x (cm)', fontsize=12)
    plt.ylabel(r'Drug concentration (mg.ml$^{-1}$)', fontsize=12)
    plt.xlim(0, max(needle_lenght_values) + 0.0001)
    plt.legend()
    plt.grid()
    plt.savefig("analytic_experiment4")

time = 24 * 60
D = 5e-8
C0 = 15.8
L = 0.015
N = 2

time_values = np.array([24, 48, 72]) * 60
D_values = np.array([5*10**(-9), 1*10**(-8), 5*10**(-8), 1*10**(-7)])
C0_values = np.array([13.8, 14.8, 15.6, 16.8])
L_values = np.array([0.01, 0.015, 0.02, 0.025])

experiment_0(time, D, C0, L, N)
# experiment_1(time_values, D, C0, L, N)
# experiment_2(time, D_values, C0, L, N)
# experiment_3(time, D, C0_values, L, N)
# experiment_4(time, D, C0, L_values, N)

