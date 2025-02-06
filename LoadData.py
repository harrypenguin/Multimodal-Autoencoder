import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def loaddata(simulation_names: list):
    """
    Load data from the simulation data files (in simulation_names) and interpolate the SFHs to 139 timesteps.
    Returns [sfhs, [logmass, arcsinh(presentsfr)]] for each galaxy in the simulation data.
    """
    work_dir = '/Users/pengzehao/Desktop/Multimodal/Iyer_etal_2020_SFH_data/'
    extn = '_sfhs_psds.mat'

    combined = []
    for sim_name in simulation_names:
        sim_data = sio.loadmat(work_dir + sim_name + extn)

        # Interpolating SFHs into 139 timesteps 
        x = np.linspace(0, 14, 139) # creating general array to act as universal timestep array, with 139 timesteps
        sfhs = np.zeros((len(sim_data['smallsfhs'].T), 139)) # creating a np array of all 0s of size (number of galaxies, 139)
        xp = np.linspace(0, 14, len(sim_data['smallsfhs'].T[0])) 
        for i in range(len(sim_data['smallsfhs'].T)):
            sfhs[i] = np.interp(x, xp, sim_data['smallsfhs'].T[i])
            
        presentsfr = sfhs[:, -1]  # Accessing the last time step for redshift 0 sfr
        logmass = np.array(sim_data['logmass'].ravel())
        if sim_name in ['Simba', 'Mufasa']:
            combined = combined + [[arr, [m, np.arcsinh(s)], sim_name] for arr, m, s in zip(sfhs, logmass, presentsfr) if m > 10]
        else:
            combined = combined + [[arr, [m, np.arcsinh(s)], sim_name] for arr, m, s in zip(sfhs, logmass, presentsfr) if m > 9]
        # Setting a cut off for mass, categorized by simulation
            
    return combined

def filter_zeroes(inputHistories, mass_presentsfr, labels):
    """
    Filter out galaxies with zero SFH.
    Returns filtered inputHistories, filtered mass_presentsfr, filtered_labels.
    """
    zero_indices = np.array([i for i in range(len(inputHistories)) if np.trapz(inputHistories[i]) == 0])
    mask = np.ones(inputHistories.shape[0], dtype=bool)
    mask[zero_indices] = False

    return inputHistories[mask], mass_presentsfr[mask], labels[mask]

def count_zeroes(inputHistories, mass_presentsfr, labels):
    """
    Count the number of galaxies with zero SFH by simulation.
    """
    # count by simulation
    zero_indices = np.array([i for i in range(len(inputHistories)) if np.trapz(inputHistories[i]) == 0])
    mask = np.ones(inputHistories.shape[0], dtype=bool)
    mask[zero_indices] = False

    # print out the number of galaxies with zero SFH for each type of simulation
    print('Total number of galaxies:', len(inputHistories))
    print('Number of galaxies with zero SFH in each simulation:')
    for i, sim in enumerate(set(labels)):
        print(sim, np.sum(np.array(labels) == sim) - np.sum(np.array(labels)[mask] == sim))
    print('Total number of galaxies with zero SFH:', len(zero_indices))

def filter_data(data):
    """
    Filter the data outputted by loaddata() and return the filtered AH, present day mass & sfr, and simulation labels.
    """
    inputHistories = np.array([element[0] for element in data])
    mass_presentsfr = np.array([element[1] for element in data])
    labels = np.array([element[2] for element in data])

    for i in range(len(labels)):
        if labels[i] == 'Eagle':
            labels[i] = 0
        elif labels[i] == 'Illustris':
            labels[i] = 1
        elif labels[i] == 'IllustrisTNG':
            labels[i] = 2
        elif labels[i] == 'Mufasa':
            labels[i] = 3
        elif labels[i] == 'Simba':
            labels[i] = 4
        elif labels[i] == 'SC-Sam':
            labels[i] = 5
        elif labels[i] == 'UniverseMachine':
            labels[i] = 6
    labels = np.array(labels).astype('float32')


    filtered_SFH, filtered_mass_presentsfr, filtered_labels = filter_zeroes(inputHistories, mass_presentsfr, labels)
    filtered_AH = np.array([(sfh / (np.trapz(sfh)) * 500) for sfh in filtered_SFH]) # Normalization into AHs
    return filtered_AH, filtered_mass_presentsfr, filtered_labels