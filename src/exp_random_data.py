import numpy as np
import scipy.io as sio
from typing import List, Optional, Dict
from math import sin, cos, tan, asin, acos, atan2, fabs, sqrt
import logging
import sophuspy as sp

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

def save_simulation_data(simulation_data: List[Dict[str, np.ndarray]], folder: str, file: str) -> None:
    """
    Saves the simulation data to a file.
    Parameters:
    - simulation_data (List[Dict[str, np.ndarray]]): The simulation data containing 'p1', 'p2', 'bearing', 'Rgt', and 'tgt' arrays.
    - folder (str): The folder path where the file will be saved.
    - file (str): The name of the file.
    Returns:
    - None
    """
    import os
    os.makedirs(folder, exist_ok=True)
    for i, data in enumerate(simulation_data):
        mat_fname = os.path.join(folder, file + str(i) + '.txt')
        with open(mat_fname, 'w') as f:
            p1 = data["p1"]
            p2 = data["p2"]
            bearing = data["bearing"]
            Rgt = data["Rgt"]
            tgt = data["tgt"]
            for j in range(p1.shape[1]):
                f.write(' '.join([str(x) for x in p1[:, j]]))
                f.write(' ')
            f.write('\n')
            for j in range(p2.shape[1]):
                f.write(' '.join([str(x) for x in p2[:, j]]))
                f.write(' ')
            f.write('\n')
            for j in range(bearing.shape[1]):
                f.write(' '.join([str(x) for x in bearing[:, j]]))
                f.write(' ')
            f.write('\n')
            Rgt = Rgt.flatten().squeeze().tolist()
            tgt = tgt.flatten().squeeze().tolist()

            if type(Rgt[0]) is list:
                Rgt = Rgt[0]
            if type(tgt[0]) is list:
                tgt = tgt[0]

            for element in Rgt:
                f.write(f"{element} ")
            f.write('\n')
            for element in tgt:
                f.write(f"{element} ")
            f.write('\n')


def prepare_data(max_size: int = 100, std_noise_on_theta: float = 2.0  * np.pi / 180.0, batch_size: int = 6) -> np.ndarray:
    # prepare bearing data
    simulation_data = []

    for i in range(0, max_size):
        # batch_size = np.random.randint(6, 30)
        # generate global position of p1 and p2
        p1_global = np.random.rand(3, batch_size) * 100
        p2_relative = np.random.rand(3, batch_size) * 100
        # generate a so3
        Ropt = sp.SO3.exp(np.random.rand(3) * 2 * np.pi).matrix()
        Tgt = np.random.rand(3, 1) * 10
        # generate bearing
        p2_global = np.zeros((3, p2_relative.shape[1]))
        for j in range(p2_relative.shape[1]):
            p2_global[:, j] = (Ropt.T@(p2_relative[:, j].reshape(3,1) - Tgt)).flatten()
        bearing_computed = np.zeros((3, p2_relative.shape[1]))
        for j in range(p2_global.shape[1]):
            vec = p1_global[:,j] - p2_global[:,j]
            vec = vec / np.linalg.norm(vec)
            bearing_computed[:, j] = Ropt.dot(vec)

        bearing_angle = np.zeros((2, bearing_computed.shape[1]))
        for j in range(bearing_computed.shape[1]):
            vec = bearing_computed[:, j]
            phi = asin(vec[2]) + np.random.randn() * std_noise_on_theta * 4
            theta = atan2(vec[1], vec[0]) + np.random.randn() * std_noise_on_theta
            bearing_angle[:, j] = np.array([theta, phi])

        f1 = lambda x: np.array([cos(x[0]) * cos(x[1]), cos(x[1]) * sin(x[0]), sin(x[1])])
        bearing = [f1(bearing_angle[:, j]) for j in range(bearing_angle.shape[1])]
        bearing = np.array(bearing).T

        # assert np.allclose(bearing, bearing_computed)
        # p2_relative = p2_relative + np.random.randn(3, batch_size) * 1
        simulation_data.append({"p1": p1_global, "p2": p2_relative, "bearing": bearing, "Rgt": Ropt, "tgt": Tgt})

    return simulation_data

def gen_simulation_data(folder: str = '../3/', max_size: int = 100, std_noise_on_theta: float = 2.0, batch_size: int = 6) -> None:
    import os
    save_data_folder = os.path.join(folder, 'batch_' + str(batch_size) + '_' + '{:.2f}'.format(float(std_noise_on_theta)))
    os.makedirs(save_data_folder, exist_ok=True)
    simulation_data = prepare_data(max_size, std_noise_on_theta * np.pi / 180.0, batch_size)
    prefix = 'trial_'
    save_simulation_data(simulation_data, save_data_folder, prefix)

def compute_simulation_data(folder: str = '../3/') -> None:
    batch_size = [6, 12]
    noise_std = [0.1, 1]
    prefix = 'trial_'
    from bearing_only_solver import bgpnp, bearing_linear_solver, load_simulation_data
    from bo_epnp_solver import BO_EPnP
    import time as pytime
    import os
    from tqdm import tqdm

    if not os.path.exists(os.path.join(folder, 'results.npz')):
        res = {}

        for bs in tqdm(batch_size):
            for ns in tqdm(noise_std, leave=False):
                save_data_folder = os.path.join(folder, 'batch_' + str(bs) + '_' + '{:.2f}'.format(float(ns)))
                files = [os.path.join(save_data_folder, f) for f in os.listdir(save_data_folder) if prefix in f]
                errors = {}
                failures = {'bgpnp': 0, 'bls': 0, 'bsdp': 0, 'bo_epnp': 0}
                times = {'bgpnp': [], 'bls': [], 'bsdp': [], 'bo_epnp': []}
                for f in tqdm(files, leave=False, desc=f'bs_{bs}_ns_{ns}'):
                    data = load_simulation_data(f)
                    uvw = data["p1"]
                    xyz = data["p2"]
                    bearing = data["bearing"]
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
                        R2, t2 = np.eye(3), np.ones(3) * 100
                        failures['bls'] += 1
                        logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

                    try:
                        (R3, t3), time = bearing_linear_solver.solve_with_sdp_sdr(uvw, xyz, bearing)
                        times['bsdp'].append(time)
                    except Exception as e:  # Corrected the typo and added exception handling
                        R3, t3 = np.eye(3), np.ones(3) * 100
                        failures['bsdp'] += 1
                        logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

                    try:
                        (R1, t1, _), time = bgpnp.solve(uvw.T, xyz.T, bearing.T, True)
                        times['bgpnp'].append(time)
                    except Exception as e:  # Corrected the typo and added exception handling
                        R1, t1 = np.eye(3), np.ones(3) * 100
                        failures['bgpnp'] += 1
                        logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

                    try:
                        t_start = pytime.perf_counter()
                        R4, t4 = BO_EPnP.solve(
                            uvw,
                            xyz,
                            bearing,
                            use_wtls=False,
                            use_ml_refine=True,
                            control_point_method="pca"
                        )
                        times['bo_epnp'].append(pytime.perf_counter() - t_start)
                    except Exception as e:
                        R4, t4 = np.eye(3), np.ones(3) * 100
                        failures['bo_epnp'] += 1
                        logger.info(f"An error occurred: {e}")

                    errors['bgpnp_rot'] = errors.get('bgpnp_rot', []) + [np.linalg.norm(R1 - data["Rgt"])]
                    errors['bgpnp_tra'] = errors.get('bgpnp_tra', []) + [np.linalg.norm(t1 - data["tgt"]) / np.linalg.norm(data["tgt"])]
                    errors['bls_rot'] = errors.get('bls_rot', []) + [np.linalg.norm(R2 - data["Rgt"])]
                    errors['bls_tra'] = errors.get('bls_tra', []) + [np.linalg.norm(t2 - data["tgt"]) / np.linalg.norm(data["tgt"])]
                    errors['bsdp_rot'] = errors.get('bsdp_rot', []) + [np.linalg.norm(R3 - data["Rgt"])]
                    errors['bsdp_tra'] = errors.get('bsdp_tra', []) + [np.linalg.norm(t3 - data["tgt"]) / np.linalg.norm(data["tgt"])]
                    errors['bo_epnp_rot'] = errors.get('bo_epnp_rot', []) + [np.linalg.norm(R4 - data["Rgt"])]
                    errors['bo_epnp_tra'] = errors.get('bo_epnp_tra', []) + [
                        np.linalg.norm(np.asarray(t4).reshape(-1) - np.asarray(data["tgt"]).reshape(-1)) / np.linalg.norm(data["tgt"])
                    ]

                res[f'bs_{bs}_ns_{ns}'] = {'mean_rot_error': {'bgpnp': np.mean(errors['bgpnp_rot']),
                                                                'bls': np.mean(errors['bls_rot']),
                                                                'bsdp': np.mean(errors['bsdp_rot']),
                                                                'bo_epnp': np.mean(errors['bo_epnp_rot'])},
                                            'mean_tra_error': {'bgpnp': np.mean(errors['bgpnp_tra']),
                                                                'bls': np.mean(errors['bls_tra']),
                                                                'bsdp': np.mean(errors['bsdp_tra']),
                                                                'bo_epnp': np.mean(errors['bo_epnp_tra'])},
                                                'failures': {'bgpnp': failures['bgpnp'], 'bls': failures['bls'], 'bsdp': failures['bsdp'], 'bo_epnp': failures['bo_epnp']},
                                                'times': {'bgpnp': np.mean(times['bgpnp']), 'bls': np.mean(times['bls']), 'bsdp': np.mean(times['bsdp']), 'bo_epnp': np.mean(times['bo_epnp'])}}

        # save results as npz
        np.savez(os.path.join(folder, 'results.npz'), res=res)
    else:
        res = np.load(os.path.join(folder, 'results.npz'), allow_pickle=True)['res'].item()
        print(res.keys())

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # import plotly.graph_objects as go
    # Assuming res, batch_size, and noise_std are already defined and populated
    # Prepare data for plotting
    data = []
    for solver in ['bgpnp', 'bls', 'bsdp', 'bo_epnp']:
        for bs in batch_size:
            for ns in noise_std:
                key = f'bs_{bs}_ns_{ns}'
                rot_val = res[key]['mean_rot_error'].get(solver, np.nan)
                tra_val = res[key]['mean_tra_error'].get(solver, np.nan)
                fail_val = res[key]['failures'].get(solver, np.nan)
                time_val = res[key]['times'].get(solver, np.nan)
                data.append({
                    'solver': solver,
                    'batch_size': bs,
                    'noise_std': ns,
                    'mean_rot_error': rot_val,
                    'mean_tra_error': tra_val,
                    'failures': fail_val,
                    'runtime': time_val  # Add runtime information
                })

    # Convert data to a DataFrame
    import pandas as pd
    df = pd.DataFrame(data)

    # Set the style of the visualization
    # Set the style of the visualization
    sns.set(style="whitegrid", palette="muted")

    # Plot Mean Rotation Error
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='batch_size', y='mean_rot_error', hue='solver', style='noise_std', markers=True, dashes=False, linewidth=2.5)
    plt.title('Mean Rotation Error vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Rotation Error')
    plt.legend(title='Solver and Noise Std', loc='upper center', ncol=3)

    # Plot Mean Translation Error
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='batch_size', y='mean_tra_error', hue='solver', style='noise_std', markers=True, dashes=False, linewidth=2.5)
    plt.title('Mean Translation Error vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Translation Error')
    plt.legend(title='Solver and Noise Std', loc='upper center', ncol=3)
    plt.ylim(0, 10)  # Set the y-axis limit to see more details

    # Plot Number of Failures
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='batch_size', y='failures', hue='solver', style='noise_std', markers=True, dashes=False, linewidth=2.5)
    plt.title('Number of Failures vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Number of Failures')
    plt.legend(title='Solver and Noise Std', loc='upper center', ncol=3)

    # Plot Mean Runtime
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='batch_size', y='runtime', hue='solver', style='noise_std', markers=True, dashes=False, linewidth=2.5)
    plt.title('Mean Runtime vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Runtime (seconds)')
    plt.legend(title='Solver and Noise Std', loc='upper center', ncol=3)
    plt.show()

    # prefix = 'batch_'
    # import os
    # files = [os.path.join(save_data_folder, f) for f in os.listdir(save_data_folder) if prefix in f]
    # from bearing_only_solver import bgpnp, bearing_linear_solver, load_simulation_data
    # errors = {}
    # failures = {'bgpnp': 0, 'bls': 0, 'bsdp': 0}
    # times = {'bgpnp': [], 'bls': [], 'bsdp': []}
    # for f in files:
    #     logger.info(f"Processing file: {f}")
    #     data = load_simulation_data(f)
    #     logger.debug(data["p1"])
    #     logger.debug(data["p2"].shape)
    #     logger.debug(data["Rgt"])
    #     logger.debug(data["tgt"])

    #     uvw = data["p1"]
    #     xyz = data["p2"]
    #     bearing = data["bearing"]

    #     bearing_angle = np.zeros((2, data["bearing"].shape[1]))
    #     for i in range(data["bearing"].shape[1]):
    #         vec = data["bearing"][:, i]
    #         phi = asin(vec[2])
    #         theta = atan2(vec[1], vec[0])
    #         bearing_angle[:, i] = np.array([theta, phi])

    #     try:
    #         (R2, t2), time = bearing_linear_solver.solve(uvw, xyz, bearing)
    #         times['bls'].append(time)
    #     except Exception as e:  # Corrected the typo and added exception handling
    #         R2, t2 = np.eye(3), np.ones(3) * 100
    #         failures['bls'] += 1
    #         logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

    #     try:
    #         (R3, t3), time = bearing_linear_solver.solve_with_sdp_sdr(uvw, xyz, bearing)
    #         times['bsdp'].append(time)
    #     except Exception as e:  # Corrected the typo and added exception handling
    #         R3, t3 = np.eye(3), np.ones(3) * 100
    #         failures['bsdp'] += 1
    #         logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

    #     try:
    #         (R1, t1, _), time = bgpnp.solve(uvw.T, xyz.T, bearing.T, True)
    #         times['bgpnp'].append(time)
    #     except Exception as e:  # Corrected the typo and added exception handling
    #         R1, t1 = np.eye(3), np.ones(3) * 100
    #         failures['bgpnp'] += 1
    #         logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

    #     logger.debug(f'R1: {R1}')
    #     logger.debug(f'R2: {R2}')
    #     logger.debug(f't1: {t1}')
    #     logger.debug(f't2: {t2}')
    #     logger.debug(f'Rgt: {data["Rgt"]}')
    #     logger.debug(f'tgt: {data["tgt"]}')
    #     logger.debug(f'Error R: {np.linalg.norm(R1 - data["Rgt"])}')
    #     logger.debug(f'Error t: {np.linalg.norm(t1 - data["tgt"])}')

    #     errors['bgpnp_rot'] = errors.get('bgpnp_rot', []) + [np.linalg.norm(R1 - data["Rgt"])]
    #     errors['bgpnp_tra'] = errors.get('bgpnp_tra', []) + [np.linalg.norm(t1 - data["tgt"])]
    #     errors['bls_rot'] = errors.get('bls_rot', []) + [np.linalg.norm(R2 - data["Rgt"])]
    #     errors['bls_tra'] = errors.get('bls_tra', []) + [np.linalg.norm(t2 - data["tgt"])]
    #     errors['bsdp_rot'] = errors.get('bsdp_rot', []) + [np.linalg.norm(R3 - data["Rgt"])]
    #     errors['bsdp_tra'] = errors.get('bsdp_tra', []) + [np.linalg.norm(t3 - data["tgt"])]

    # # plot the errors
    # import matplotlib.pyplot as plt
    # # Create a figure with two subplots
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # # Plot rotation errors in the first subplot
    # ax1.plot(errors['bgpnp_rot'], label='BGPnP Rotation')
    # ax1.plot(errors['bls_rot'], label='BLS Rotation')
    # ax1.plot(errors['bsdp_rot'], label='BSDP Rotation')
    # ax1.set_title('Rotation Errors')
    # ax1.set_xlabel('Trial')
    # ax1.set_ylabel('Error')
    # ax1.legend()
    # ax1.grid(True)

    # # Plot translation errors in the second subplot
    # ax2.plot(errors['bgpnp_tra'], label='BGPnP Translation')
    # ax2.plot(errors['bls_tra'], label='BLS Translation')
    # ax2.plot(errors['bsdp_tra'], label='BSDP Translation')
    # ax2.set_title('Translation Errors')
    # ax2.set_xlabel('Trial')
    # ax2.set_ylabel('Error')
    # ax2.legend()
    # ax2.grid(True)

    # # Adjust layout to prevent overlap
    # plt.tight_layout()

    # # draw another plot showing the times, use boxplot
    # fig, ax = plt.subplots()
    # # Prepare data for boxplot
    # data = [times['bgpnp'], times['bls'], times['bsdp']]
    # labels = ['BGPnP', 'BLS', 'BSDP']

    # # Create the boxplot
    # ax.boxplot(data, labels=labels)

    # # Set title and labels
    # ax.set_title('Execution Times')
    # ax.set_xlabel('Method')
    # ax.set_ylabel('Time (s)')
    # ax.grid(True)

    # # Show the plot
    # plt.show()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate simulation data for localization problem.')
    parser.add_argument('--max_size', type=int, default=100, help='The maximum size of the data.')
    parser.add_argument('--generate', action='store_true', help='Generate simulation data.')
    parser.add_argument('--batch_size', type=int, default=6, help='The batch size of the data.')
    parser.add_argument('--folder', type=str, default='../3/', help='The folder path where the file is located.')
    parser.add_argument('--std_noise_on_theta', type=float, default=2.0, help='The standard deviation of the noise on theta.')

    args = parser.parse_args()

    if args.generate:
        gen_simulation_data(args.folder, args.max_size, args.std_noise_on_theta, args.batch_size)
    else:
        compute_simulation_data(args.folder)
