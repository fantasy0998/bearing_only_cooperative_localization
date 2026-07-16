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


def prepare_data(max_size: int = 100, std_noise_on_theta: float = 2.0  * np.pi / 180.0, batch_size: int = 6, ratio: float = 0.2) -> np.ndarray:
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

        num_outliers = int(batch_size * ratio)
        if num_outliers > 0:
            vec_outlier = np.random.rand(3, num_outliers) * 2 - 1
            vec_outlier = vec_outlier / np.linalg.norm(vec_outlier, axis=0)
            bearing[:, -num_outliers:] = vec_outlier

        # assert np.allclose(bearing, bearing_computed)
        # p2_relative = p2_relative + np.random.randn(3, batch_size) * 1
        simulation_data.append({"p1": p1_global, "p2": p2_relative, "bearing": bearing, "Rgt": Ropt, "tgt": Tgt})

    return simulation_data

def gen_simulation_data(folder: str = '../5/', max_size: int = 100, std_noise_on_theta: float = 2.0, batch_size: int = 6, ratio: float = 0.2) -> None:
    import os
    save_data_folder = os.path.join(folder, 'batch_' + str(batch_size) + '_' + '{:.2f}'.format(float(std_noise_on_theta)) + '_' + '{:.2f}'.format(float(ratio)))
    os.makedirs(save_data_folder, exist_ok=True)
    simulation_data = prepare_data(max_size, std_noise_on_theta * np.pi / 180.0, batch_size, ratio)
    prefix = 'trial_'
    save_simulation_data(simulation_data, save_data_folder, prefix)

def compute_simulation_data(folder: str = '../5/') -> None:
    batch_size = [10, 20, 30]
    noise_std = [0.01]
    ratio = [0.2]
    prefix = 'trial_'
    from bearing_only_solver import bgpnp, bearing_linear_solver, load_simulation_data
    import os
    from tqdm import tqdm

    if not os.path.exists(os.path.join(folder, 'results.npz')):
        res = {}

        for bs in tqdm(batch_size):
            for ns in tqdm(noise_std, leave=False):
                save_data_folder = os.path.join(folder, 'batch_' + str(bs) + '_' + '{:.2f}'.format(float(ns)) + '_' + '{:.2f}'.format(float(ratio[0])))
                files = [os.path.join(save_data_folder, f) for f in os.listdir(save_data_folder) if prefix in f]
                errors = {}
                failures = {'bgpnp': 0, 'bls': 0, 'bsdp': 0}
                times = {'bgpnp': [], 'bls': [], 'bsdp': []}
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
                        (R2, t2), time = bearing_linear_solver.ransac_solve(uvw, xyz, bearing)
                        times['bls'].append(time)
                    except Exception as e:  # Corrected the typo and added exception handling
                        R2, t2 = np.eye(3), np.ones(3) * 100
                        failures['bls'] += 1
                        logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

                    try:
                        (R3, t3), time = bearing_linear_solver.ransac_solve_with_sdp_sdr(uvw, xyz, bearing)
                        times['bsdp'].append(time)
                    except Exception as e:  # Corrected the typo and added exception handling
                        R3, t3 = np.eye(3), np.ones(3) * 100
                        failures['bsdp'] += 1
                        logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

                    #try:
                    (R1, t1, _), time = bgpnp.ransac_solve(uvw.T, xyz.T, bearing.T)
                    times['bgpnp'].append(time)
                    # except Exception as e:  # Corrected the typo and added exception handling
                        # R1, t1 = np.eye(3), np.ones(3) * 100
                        # failures['bgpnp'] += 1
                        # logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

                    errors['bgpnp_rot'] = errors.get('bgpnp_rot', []) + [np.linalg.norm(R1 - data["Rgt"])]
                    errors['bgpnp_tra'] = errors.get('bgpnp_tra', []) + [np.linalg.norm(t1 - data["tgt"]) / np.linalg.norm(data["tgt"])]
                    errors['bls_rot'] = errors.get('bls_rot', []) + [np.linalg.norm(R2 - data["Rgt"])]
                    errors['bls_tra'] = errors.get('bls_tra', []) + [np.linalg.norm(t2 - data["tgt"]) / np.linalg.norm(data["tgt"])]
                    errors['bsdp_rot'] = errors.get('bsdp_rot', []) + [np.linalg.norm(R3 - data["Rgt"])]
                    errors['bsdp_tra'] = errors.get('bsdp_tra', []) + [np.linalg.norm(t3 - data["tgt"]) / np.linalg.norm(data["tgt"])]

                res[f'bs_{bs}_ns_{ns}'] = {'mean_rot_error': {'bgpnp': np.mean(errors['bgpnp_rot']),
                                                                'bls': np.mean(errors['bls_rot']),
                                                                'bsdp': np.mean(errors['bsdp_rot'])},
                                            'mean_tra_error': {'bgpnp': np.mean(errors['bgpnp_tra']),
                                                                'bls': np.mean(errors['bls_tra']),
                                                                'bsdp': np.mean(errors['bsdp_tra'])},
                                                'failures': {'bgpnp': failures['bgpnp'], 'bls': failures['bls'], 'bsdp': failures['bsdp']},
                                                'times': {'bgpnp': np.mean(times['bgpnp']), 'bls': np.mean(times['bls']), 'bsdp': np.mean(times['bsdp'])}}

        # save results as npz
        np.savez(os.path.join(folder, 'results.npz'), res=res)
    else:
        res = np.load(os.path.join(folder, 'results.npz'), allow_pickle=True)['res'].item()
        print(res.keys())

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    import plotly.graph_objects as go
    # Assuming res, batch_size, and noise_std are already defined and populated
    # Prepare data for plotting
    data = []
    for solver in ['bgpnp', 'bls', 'bsdp']:
        for bs in batch_size:
            for ns in noise_std:
                key = f'bs_{bs}_ns_{ns}'
                data.append({
                    'solver': solver,
                    'batch_size': bs,
                    'noise_std': ns,
                    'mean_rot_error': res[key]['mean_rot_error'][solver],
                    'mean_tra_error': res[key]['mean_tra_error'][solver],
                    'failures': res[key]['failures'][solver],
                    'runtime': res[key]['times'][solver]  # Add runtime information
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


def compute_simulation_data2(folder: str = '../5/') -> None:
    batch_size = [10, 20, 30]
    noise_std = [0.01]
    ratio = [0.2]
    prefix = 'trial_'
    from bearing_only_solver import bgpnp, bearing_linear_solver, load_simulation_data
    import os
    from tqdm import tqdm

    for bs in tqdm(batch_size):
        for ns in tqdm(noise_std, leave=False):
            save_data_folder = os.path.join(folder, 'batch_' + str(bs) + '_' + '{:.2f}'.format(float(ns)) + '_' + '{:.2f}'.format(float(ratio[0])))
            files = [os.path.join(save_data_folder, f) for f in os.listdir(save_data_folder) if prefix in f]
            errors = {}
            failures = {'bgpnp': 0, 'bls': 0, 'bsdp': 0}
            times = {'bgpnp': [], 'bls': [], 'bsdp': []}
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
                    (R2, t2), time = bearing_linear_solver.ransac_solve(uvw, xyz, bearing)
                    times['bls'].append(time)
                except Exception as e:  # Corrected the typo and added exception handling
                    R2, t2 = np.eye(3), np.ones(3,1) * 100
                    failures['bls'] += 1
                    logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

                try:
                    (R3, t3), time = bearing_linear_solver.ransac_solve_with_sdp_sdr(uvw, xyz, bearing)
                    times['bsdp'].append(time)
                except Exception as e:  # Corrected the typo and added exception handling
                   R3, t3 = np.eye(3), np.ones(3,1) * 100
                   failures['bsdp'] += 1
                   logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

                try:
                    (R1, t1, _), time = bgpnp.ransac_solve(uvw.T, xyz.T, bearing.T)
                    times['bgpnp'].append(time)
                except Exception as e:  # Corrected the typo and added exception handling
                   R1, t1 = np.eye(3), np.ones(3, 1) * 100
                   failures['bgpnp'] += 1
                   logger.info(f"An error occurred: {e}")  # Optional: Print the exception message for debugging

                errors['bgpnp_rot'] = errors.get('bgpnp_rot', []) + [np.linalg.norm(R1 - data["Rgt"])]
                errors['bgpnp_tra'] = errors.get('bgpnp_tra', []) + [np.linalg.norm(t1 - data["tgt"]) / np.linalg.norm(data["tgt"])]
                errors['bls_rot'] = errors.get('bls_rot', []) + [np.linalg.norm(R2 - data["Rgt"])]
                errors['bls_tra'] = errors.get('bls_tra', []) + [np.linalg.norm(t2 - data["tgt"]) / np.linalg.norm(data["tgt"])]
                errors['bsdp_rot'] = errors.get('bsdp_rot', []) + [np.linalg.norm(R3 - data["Rgt"])]
                errors['bsdp_tra'] = errors.get('bsdp_tra', []) + [np.linalg.norm(t3 - data["tgt"]) / np.linalg.norm(data["tgt"])]


                logger.debug(f"Rgt: {data['Rgt']}")
                logger.debug(f"tgt: {data['tgt']}")
                logger.debug(f"R1: {R1}")
                logger.debug(f"t1: {t1}")
                logger.debug(f"R2: {R2}")
                logger.debug(f"t2: {t2}")
                logger.debug(f"R3: {R3}")
                logger.debug(f"t3: {t3}")
                logger.warning(f"Errors: {errors}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate simulation data for localization problem.')
    parser.add_argument('--max_size', type=int, default=100, help='The maximum size of the data.')
    parser.add_argument('--generate', action='store_true', help='Generate simulation data.')
    parser.add_argument('--batch_size', type=int, default=20, help='The batch size of the data.')
    parser.add_argument('--folder', type=str, default='../5/', help='The folder path where the file is located.')
    parser.add_argument('--std_noise_on_theta', type=float, default=0.1, help='The standard deviation of the noise on theta.')
    parser.add_argument('--ratio_outlier', type=float, help='The ratio of outlier.')

    args = parser.parse_args()

    if args.generate:
        gen_simulation_data(args.folder, args.max_size, args.std_noise_on_theta, args.batch_size, args.ratio_outlier)
    else:
        compute_simulation_data(args.folder)
