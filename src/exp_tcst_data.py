import numpy as np
import scipy.io as sio
from typing import List, Optional, Dict
from math import sin, cos, tan, asin, acos, atan2, fabs, sqrt
import logging

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

def compute_simulation_data(folder: str = '../2/') -> None:
    save_data_folder = folder
    prefix = 'batch_'

    import os
    files = [os.path.join(save_data_folder, f) for f in os.listdir(save_data_folder) if prefix in f]
    from bearing_only_solver import bgpnp, bearing_linear_solver, load_simulation_data
    import sophuspy as sp

    errors = {}
    failures = {'bgpnp': 0, 'bls': 0, 'bsdp': 0}
    times = {'bgpnp': [], 'bls': [], 'bsdp': []}
    for f in files:
        logger.info(f"Processing file: {f}")
        data = load_simulation_data(f)
        logger.debug(data["p1"])
        logger.debug(data["p2"].shape)
        logger.debug(data["Rgt"])
        logger.debug(data["tgt"])

        uvw = data["p1"]
        xyz = data["p2"]
        bearing = data["bearing"]

        try:
            # timeit
            (R1, t1, err), time = bgpnp.solve(uvw.T, xyz.T, bearing.T, True)
            times['bgpnp'].append(time)
            logger.warning(err)
        except Exception as e:  # Corrected the typo and added exception handling
            R1, t1 = np.eye(3), np.zeros(3)
            failures['bgpnp'] += 1
            logger.info(f"bgpnp An error occurred: {e}")  # Optional: Print the exception message for debugging

        bearing_angle = np.zeros((2, data["bearing"].shape[1]))
        for i in range(data["bearing"].shape[1]):
            vec = data["bearing"][:, i]
            phi = asin(vec[2])
            theta = atan2(vec[1], vec[0])
            bearing_angle[:, i] = np.array([theta, phi])

        try:
            (R2, t2), time = bearing_linear_solver.solve(uvw, xyz, bearing)
            times['bls'].append(time)
        except Exception as e:  # Corrected the typo and added exception handling
            R2, t2 = np.eye(3), np.zeros(3)
            failures['bls'] += 1
            logger.info(f"bls An error occurred: {e}")  # Optional: Print the exception message for debugging

        try:
            (R3, t3), time = bearing_linear_solver.solve_with_sdp_sdr(uvw, xyz, bearing)
            times['bsdp'].append(time)
        except Exception as e:  # Corrected the typo and added exception handling
            R3, t3 = np.eye(3), np.zeros(3)
            failures['bsdp'] += 1
            logger.info(f"bsdp An error occurred: {e}")  # Optional: Print the exception message for debugging

        logger.debug(f'R1: {R1}')
        logger.debug(f'R2: {R2}')
        logger.debug(f't1: {t1}')
        logger.debug(f't2: {t2}')
        logger.debug(f'Rgt: {data["Rgt"]}')
        logger.debug(f'tgt: {data["tgt"]}')
        logger.debug(f'Error R: {np.linalg.norm(R1 - data["Rgt"])}')
        logger.info(f'Error t: {np.linalg.norm(t1 - data["tgt"])}')

        errors['bgpnp_rot'] = errors.get('bgpnp_rot', []) + [np.linalg.norm(R1 - data["Rgt"])]
        errors['bgpnp_tra'] = errors.get('bgpnp_tra', []) + [np.linalg.norm(t1 - data["tgt"])]
        errors['bls_rot'] = errors.get('bls_rot', []) + [np.linalg.norm(R2 - data["Rgt"])]
        errors['bls_tra'] = errors.get('bls_tra', []) + [np.linalg.norm(t2 - data["tgt"])]
        errors['bsdp_rot'] = errors.get('bsdp_rot', []) + [np.linalg.norm(R3 - data["Rgt"])]
        errors['bsdp_tra'] = errors.get('bsdp_tra', []) + [np.linalg.norm(t3 - data["tgt"])]

    # save the errors as npz file
    # np.savez(save_data_folder + 'errors.npz', **errors)

    # plot the errors
    import matplotlib.pyplot as plt
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot rotation errors in the first subplot
    ax1.plot(errors['bgpnp_rot'], label='BGPnP Rotation')
    ax1.plot(errors['bls_rot'], label='BLS Rotation')
    ax1.plot(errors['bsdp_rot'], label='BSDP Rotation')
    ax1.set_title('Rotation Errors')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Error')
    ax1.legend()
    ax1.grid(True)

    # Plot translation errors in the second subplot
    ax2.plot(errors['bgpnp_tra'], label='BGPnP Translation')
    ax2.plot(errors['bls_tra'], label='BLS Translation')
    ax2.plot(errors['bsdp_tra'], label='BSDP Translation')
    ax2.set_title('Translation Errors')
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # draw another plot showing the times, use boxplot
    fig, ax = plt.subplots()
    # Prepare data for boxplot
    data = [times['bgpnp'], times['bls'], times['bsdp']]
    labels = ['BGPnP', 'BLS', 'BSDP']

    # Create the boxplot
    ax.boxplot(data, labels=labels)

    # Set title and labels
    ax.set_title('Execution Times')
    ax.set_xlabel('Method')
    ax.set_ylabel('Time (s)')
    ax.grid(True)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate simulation data for bearing localization problem.')
    parser.add_argument('--folder', type=str, default='../2/', help='The folder path where the file is located.')
    args = parser.parse_args()
    compute_simulation_data(args.folder)
