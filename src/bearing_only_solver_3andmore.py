import numpy as np
import os
import logging
from math import sin, cos, tan, asin, acos, atan2, fabs, sqrt
from typing import Optional, Tuple, List, Dict
import cvxpy as cp
import time
from scipy.optimize import least_squares
from bearing_only_solver import bgpnp

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

logger = logging.getLogger(__name__)

class bgpnp_generic(bgpnp):
    def __init__(self) -> None:
        super().__init__()
        pass

    @staticmethod
    def solve(p2s: List[Dict[str, np.ndarray]], bearings_relative: List[Tuple[str, str, np.ndarray]],
              sol_iter: bool = True) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

        # sequentially solve the problem for each target
        args_results = []
        args_results2 = []
        for i, bearing_pack in enumerate(bearings_relative):
            from_id, to_id, bearing = bearing_pack
            M, b, Alph, Cw = bgpnp.prepare_data(p2s[to_id].T, bearing.T, p2s[from_id].T)
            possible_dims = 4
            Km = bgpnp.kernel_noise(M, b, dimker=possible_dims)
            # R, t, err, ctl_pts = bgpnp.KernelPnP(Cw, Km, dims=4, sol_iter=sol_iter)
            args_results.append({'from_id': from_id, 'to_id': to_id, 'M': M, 'b': b, 'Alph': Alph, 'Cw': Cw, 'Km': Km})
            args_results2.append({'from_id': from_id, 'to_id': to_id,
                                    'p1': p2s[to_id], 'p2': p2s[from_id],
                                    'bearing': bearing})


        # def fun2(x, p2, bearing, alph):
        #             M = bgpnp.kron_A_N(alph, 3) # 3n x 12
        #             bearing_computed = M @ x
        #             bearing_computed = bearing_computed.reshape(-1, 3) - p2
        #             bearing_computed = bearing_computed / np.linalg.norm(bearing_computed, axis=1)[:, None]
        #             # do cross-product for each bearing
        #             cross_products = np.cross(bearing_computed, bearing)
        #             error = cross_products.reshape(-1)
        #             # error = np.linalg.norm(cross_products, axis=1)
        #             return error



        def fun3(x, *args, **kwargs):
            params1 = args[0]
            params2 = args[1]
            dims = 4
            rt_global = {}
            # todo: we need a graph to solve this problem,
            #   for now we just try a-b-c cases to validate its feasibility
            total_error = 0
            for i, result in enumerate(params1):
                from_id = result['from_id']
                to_id = result['to_id']
                M = result['M']
                b = result['b']
                Alph = result['Alph']
                Cw = result['Cw']
                Km = result['Km']
                # prepare X
                X = {}
                X['P'] = Cw.T
                X['mP'] = np.mean(X['P'], axis=1)
                X['cP'] = X['P'] - X['mP'].reshape(3, 1)
                X['norm'] = np.linalg.norm(X['cP'])
                X['nP'] = X['cP'] / X['norm']
                # prepare V
                abcdi = x[i * 4: (i + 1) * 4]
                newV = np.reshape(Km @ abcdi, (dims, -1)).T
                # procrustes solution
                R, b, mc = bgpnp.myProcrustes(X, newV)
                solV = b * newV
                solR = R
                # pack the results
                R = solR
                mV = np.mean(solV, axis=1)
                T = mV - R @ X['mP']
                # compute error
                newerr = np.linalg.norm(R.T @ newV + mc - X['P'],2)
                if to_id == 'global':
                    # total_error += newerr
                    # logger.warning(f'newerr: {newerr}')
                    rt_global[from_id] = (R, T)

                    R2, T2 = rt_global[from_id]

                    p1, p2, bearing = params2[i]['p1'], params2[i]['p2'], params2[i]['bearing']
                    # logger.warning(f'p1: {p1.shape}, p2: {p2.shape}, bearing: {bearing.shape}')
                    bearing_computed = R2 @ p1 + T2.reshape(3,1) - p2
                    bearing_computed = bearing_computed / np.linalg.norm(bearing_computed, axis=0)
                    cross_products = bearing_computed - bearing #np.cross(bearing_computed.T, bearing.T)
                    error = np.linalg.norm(cross_products, axis=0)
                    total_error += np.sum(error)
                else:
                    # logger.warning(rt_global)
                    if to_id in rt_global.keys() and from_id in rt_global.keys():
                        # R,t from global to local
                        R1, T1 = rt_global[to_id]
                        R2, T2 = rt_global[from_id]
                        R_relative = R2 @ R1.T
                        T_relative = T2 - R_relative.dot(T1)

                        p1, p2, bearing = params2[i]['p1'], params2[i]['p2'], params2[i]['bearing']
                        # logger.warning(f'p1: {p1.shape}, p2: {p2.shape}, bearing: {bearing.shape}')
                        bearing_computed = R_relative @ p1 + T_relative.reshape(3,1) - p2
                        bearing_computed = bearing_computed / np.linalg.norm(bearing_computed, axis=0)
                        cross_products = bearing_computed - bearing #np.cross(bearing_computed.T, bearing.T)
                        error = np.linalg.norm(cross_products, axis=0)
                        total_error += np.sum(error)

                        # logger.warning(f'from_id: {from_id}, to_id: {to_id}')
                        # logger.warning(f'R_relative: {R_relative}')
                        # logger.warning(f'T_relative: {T_relative}')
                        # logger.warning(f'Final solution: {R}, {T}')

                        # compute transformation caused error
                        # point_orig = Cw.T
                        # point_dst = solV
                        # point_pred = R_relative @ point_orig + T_relative.reshape(3, 1)
                        # newerr2 = np.linalg.norm(point_pred - point_dst,2)
                        # # logger.warning(f'point_dst: {point_dst}')
                        # # logger.warning(f'newerr: {newerr}')
                        # # logger.warning(f'newerr2: {newerr2}')
                        # total_error += newerr
                        # total_error += newerr2
                        # point_orig = Cw.T
                        # point_dst = solV
                        # logger.debug(f'point_orig: {point_orig}')
                        # logger.debug(f'point_dst: {point_dst}')
                        # point_pred = R @ point_orig + T.reshape(3, 1)
                        # logger.debug(f'point_pred: {point_pred}')
                        # logger.debug(f'Final solution: {R}, {T}')
                    # end if
                # end if
            # end for
            return total_error

        x0 = np.zeros(len(args_results) * 4)
        x0[3::4] = 1
        logger.warning(f'Initial guess: {x0}')
        res = least_squares(fun3, x0, args=(args_results, args_results2), verbose=2, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        x = res.x

        print(f'Final solution: {x}')

        dims = 4
        rt_global = {}
        for i, result in enumerate(args_results):
            # logger.warning(result['from_id'])
            # logger.warning(result['to_id'])
            # logger.warning(i)
            if result['to_id'] == 'global':
                M = result['M']
                b = result['b']
                Alph = result['Alph']
                Cw = result['Cw']
                Km = result['Km']
                # prepare X
                X = {}
                X['P'] = Cw.T
                X['mP'] = np.mean(X['P'], axis=1)
                X['cP'] = X['P'] - X['mP'].reshape(3, 1)
                X['norm'] = np.linalg.norm(X['cP'])
                X['nP'] = X['cP'] / X['norm']
                # prepare V
                abcdi = x[i * 4: (i + 1) * 4]
                newV = np.reshape(Km @ abcdi, (dims, -1)).T
                # procrustes solution
                R, b, mc = bgpnp.myProcrustes(X, newV)
                solV = b * newV
                solR = R
                # pack the results
                R = solR
                mV = np.mean(solV, axis=1)
                T = mV - R @ X['mP']

                rt_global[result['from_id']] = (R, T)
                # logger.warning(result['from_id'])
            # end if
        # end for
        return rt_global


def bearing_only_solver():
    batch_size = 6
    # generate global position of p1 and p2
    p1_global = np.random.rand(3, batch_size) * 100
    p2_relative = np.random.rand(3, batch_size) * 100
    p3_relative = np.random.rand(3, batch_size) * 100

    import sophuspy as sp
    # generate a so3
    Ropt2 = sp.SO3.exp(np.random.rand(3) * 2 * np.pi).matrix()
    Tgt2 = np.random.rand(3, 1) * 10
    # generate bearing
    p2_global = np.zeros((3, p2_relative.shape[1]))
    for j in range(p2_relative.shape[1]):
        p2_global[:, j] = (Ropt2.T@(p2_relative[:, j].reshape(3,1) - Tgt2)).flatten()
    bearing_21 = np.zeros((3, p2_relative.shape[1]))
    for j in range(p2_global.shape[1]):
        vec = p1_global[:,j] - p2_global[:,j]
        vec = vec / np.linalg.norm(vec)
        bearing_21[:, j] = Ropt2.dot(vec)

    # generate a so3
    Ropt3 = sp.SO3.exp(np.random.rand(3) * 2 * np.pi).matrix()
    Tgt3 = np.random.rand(3, 1) * 10
    # generate bearing
    p3_global = np.zeros((3, p3_relative.shape[1]))
    for j in range(p3_relative.shape[1]):
        p3_global[:, j] = (Ropt3.T@(p3_relative[:, j].reshape(3,1) - Tgt3)).flatten()
    bearing_31 = np.zeros((3, p3_relative.shape[1]))
    for j in range(p3_global.shape[1]):
        vec = p1_global[:,j] - p3_global[:,j]
        vec = vec / np.linalg.norm(vec)
        bearing_31[:, j] = Ropt3.dot(vec)

    bearing_32 = np.zeros((3, p3_relative.shape[1]))
    for j in range(p3_global.shape[1]):
        vec = p2_global[:,j] - p3_global[:,j]
        vec = vec / np.linalg.norm(vec)
        bearing_32[:, j] = Ropt3.dot(vec)

    logger.info(f'Ropt23: {Ropt3 @ Ropt2.T}')
    logger.info(f'top23: {(Tgt3 - Ropt3 @ Ropt2.T @ Tgt2).reshape(-1)}')

    p2s = {'global': p1_global, 'p2': p2_relative, 'p3': p3_relative}
    bearings_relative = [('p2', 'global', bearing_21), ('p3', 'global', bearing_31), ('p3', 'p2', bearing_32)]

    # solve the problem
    rt_global = bgpnp_generic.solve(p2s, bearings_relative)
    # logger.warning(rt_global)
    # check the result
    R1, T1 = rt_global['p2']
    R2, T2 = rt_global['p3']

    logger.info(f'Ropt2: {Ropt2}')
    logger.info(f'top2: {Tgt2.reshape(-1)}')
    logger.info(f'Solution R: {R1}')
    logger.info(f't: {T1}')
    logger.info('********************************************')
    logger.info(f'Ropt3: {Ropt3}')
    logger.info(f'top3: {Tgt3.reshape(-1)}')
    logger.info(f'Solution R: {R2}')
    logger.info(f't: {T2}')


if __name__ == "__main__":
    bearing_only_solver()
