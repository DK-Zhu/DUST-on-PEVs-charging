import numpy as np
import os
from algorithms import opt_solver, DUST, DOPP, dual_subgradient

# Set up the logger to print info messages for understandability.
import logging
import sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='')

log_dir = 'logs'
subdir_names = ['diffB', 'diffN', 'comp']
for subdir_name in subdir_names:
    subdir_path = f'{log_dir}/{subdir_name}'
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

def compare_B(C, R, net_B1, net_B8, net_B15):
    '''
    Compare B=1, 8, 15.
    Parameters:
        C: np.ndarray, shape=(N, Tmax, d)
        R: np.ndarray, shape=(N, Tmax)
        net_B1, net_B8, net_B15: str, network names with B=1, 8, 15
    '''

    the_opt_solver = opt_solver(C, R)
    opt_val_list = the_opt_solver.get_OptVal_list()

    W_subG = np.load(f'networks/{net_B1}/W_subG.npy')
    DUST_algo = DUST(W_subG, C, R, opt_val_list)
    reg_DUST, cons_vio_DUST = DUST_algo.compute_metrics()
    np.savetxt(f'logs/diffB/Reg_DUST_B1.csv', reg_DUST, delimiter=',')
    np.savetxt(f'logs/diffB/Vio_DUST_B1.csv', cons_vio_DUST, delimiter=',')

    W_subG = np.load(f'networks/{net_B8}/W_subG.npy')
    DUST_algo = DUST(W_subG, C, R, opt_val_list)
    reg_DUST, cons_vio_DUST = DUST_algo.compute_metrics()
    np.savetxt(f'logs/diffB/Reg_DUST_B8.csv', reg_DUST, delimiter=',')
    np.savetxt(f'logs/diffB/Vio_DUST_B8.csv', cons_vio_DUST, delimiter=',')

    W_subG = np.load(f'networks/{net_B15}/W_subG.npy')
    DUST_algo = DUST(W_subG, C, R, opt_val_list)
    reg_DUST, cons_vio_DUST = DUST_algo.compute_metrics()
    np.savetxt(f'logs/diffB/Reg_DUST_B15.csv', reg_DUST, delimiter=',')
    np.savetxt(f'logs/diffB/Vio_DUST_B15.csv', cons_vio_DUST, delimiter=',')

def compare_N(C_N10, R_N10, net_N10, C_N20, R_N20, net_N20, C_N30, R_N30, net_N30):
    '''
    Compare N=10, 20, 30.
    Parameters:
        C_N10, C_N20, C_N30: np.ndarray, shape=(N, Tmax, d)
        R_N10, R_N20, R_N30: np.ndarray, shape=(N, Tmax)
        net_N10, net_N20, net_N30: str, network names with N=10, 20, 30
    '''

    C, R = C_N10, R_N10
    the_opt_solver = opt_solver(C, R)
    opt_val_list = the_opt_solver.get_OptVal_list()

    W_subG = np.load(f'networks/{net_N10}/W_subG.npy')
    DUST_algo = DUST(W_subG, C, R, opt_val_list)
    reg_DUST, cons_vio_DUST = DUST_algo.compute_metrics()
    np.savetxt(f'logs/diffN/Reg_DUST_N10.csv', reg_DUST, delimiter=',')
    np.savetxt(f'logs/diffN/Vio_DUST_N10.csv', cons_vio_DUST, delimiter=',')

    C, R = C_N20, R_N20

    the_opt_solver = opt_solver(C, R)
    opt_val_list = the_opt_solver.get_OptVal_list()

    W_subG = np.load(f'networks/{net_N20}/W_subG.npy')
    DUST_algo = DUST(W_subG, C, R, opt_val_list)
    reg_DUST, cons_vio_DUST = DUST_algo.compute_metrics()
    np.savetxt(f'logs/diffN/Reg_DUST_N20.csv', reg_DUST, delimiter=',')
    np.savetxt(f'logs/diffN/Vio_DUST_N20.csv', cons_vio_DUST, delimiter=',')

    C, R = C_N30, R_N30

    the_opt_solver = opt_solver(C, R)
    opt_val_list = the_opt_solver.get_OptVal_list()

    W_subG = np.load(f'networks/{net_N30}/W_subG.npy')
    DUST_algo = DUST(W_subG, C, R, opt_val_list)
    reg_DUST, cons_vio_DUST = DUST_algo.compute_metrics()
    np.savetxt(f'logs/diffN/Reg_DUST_N30.csv', reg_DUST, delimiter=',')
    np.savetxt(f'logs/diffN/Vio_DUST_N30.csv', cons_vio_DUST, delimiter=',')

def compare_algos(C, R, net):
    '''
    Compare DUST, DOPP, and dual subgradient.
    Parameters:
        C: np.ndarray, shape=(N, Tmax, d)
        R: np.ndarray, shape=(N, Tmax)
        net: str, network name
    '''
    W_subG = np.load(f'networks/{net}/W_subG.npy')

    the_opt_solver = opt_solver(C, R)
    opt_val_list = the_opt_solver.get_OptVal_list()

    DUST_algo = DUST(W_subG, C, R, opt_val_list)
    reg_DUST, cons_vio_DUST = DUST_algo.compute_metrics()
    np.savetxt(f'logs/comp/Reg_DUST.csv', reg_DUST, delimiter=',')
    np.savetxt(f'logs/comp/Vio_DUST.csv', cons_vio_DUST, delimiter=',')

    DOPP_algo = DOPP(W_subG, C, R, opt_val_list)
    reg_DOPP, cons_vio_DOPP = DOPP_algo.compute_metrics()
    np.savetxt(f'logs/comp/Reg_DOPP.csv', reg_DOPP, delimiter=',')
    np.savetxt(f'logs/comp/Vio_DOPP.csv', cons_vio_DOPP, delimiter=',')

    the_dual_subgradient = dual_subgradient(W_subG, C, R, opt_val_list)
    reg_ds, cons_vio_ds = the_dual_subgradient.compute_metrics()
    np.savetxt(f'logs/comp/Reg_DS.csv', reg_ds, delimiter=',')
    np.savetxt(f'logs/comp/Vio_DS.csv', cons_vio_ds, delimiter=',')