import numpy as np
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')

save_dir = 'figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

font = {'size': 40}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = [12, 9]
plt.rcParams['lines.linewidth'] = 5

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.rcParams['text.usetex'] = True

# fig 1: different B
Reg_DUST_B1 = np.loadtxt('logs/diffB/Reg_DUST_B1.csv', delimiter=',')
Reg_DUST_B8 = np.loadtxt('logs/diffB/Reg_DUST_B8.csv', delimiter=',')
Reg_DUST_B15 = np.loadtxt('logs/diffB/Reg_DUST_B15.csv', delimiter=',')

Vio_DUST_B1 = np.loadtxt('logs/diffB/Vio_DUST_B1.csv', delimiter=',')
Vio_DUST_B8 = np.loadtxt('logs/diffB/Vio_DUST_B8.csv', delimiter=',')
Vio_DUST_B15 = np.loadtxt('logs/diffB/Vio_DUST_B15.csv', delimiter=',')

plt.figure()
plt.plot(Reg_DUST_B1, label=r'$B=1$', linestyle='-')
plt.plot(Reg_DUST_B8, label=r'$B=8$', linestyle='-')
plt.plot(Reg_DUST_B15, label=r'$B=15$', linestyle='-')
plt.xlim(0, 300)
plt.ylim(bottom=0.85e1)
plt.yscale('log')
plt.legend(fontsize=40)
plt.xlabel(r'$T$', fontsize=45)
plt.ylabel(r'$\mathrm{Reg}(T)/T$', fontsize=45)
plt.savefig(f'{save_dir}/diffB_reg.pdf', bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(Vio_DUST_B1, label=r'$B=1$', linestyle='-')
plt.plot(Vio_DUST_B8, label=r'$B=8$', linestyle='-')
plt.plot(Vio_DUST_B15, label=r'$B=15$', linestyle='-')
plt.xlim(0, 300)
plt.yscale('log')
plt.legend(fontsize=40)
plt.xlabel(r'$T$', fontsize=45)
plt.ylabel(r'$\mathrm{Reg}^c (T)/T$', fontsize=45)
plt.savefig(f'{save_dir}/diffB_vio.pdf', bbox_inches='tight')
plt.close()

# fig 2: different N
Reg_DUST_N10 = np.loadtxt('logs/diffN/Reg_DUST_N10.csv', delimiter=',')
Reg_DUST_N20 = np.loadtxt('logs/diffN/Reg_DUST_N20.csv', delimiter=',')
Reg_DUST_N30 = np.loadtxt('logs/diffN/Reg_DUST_N30.csv', delimiter=',')

Vio_DUST_N10 = np.loadtxt('logs/diffN/Vio_DUST_N10.csv', delimiter=',')
Vio_DUST_N20 = np.loadtxt('logs/diffN/Vio_DUST_N20.csv', delimiter=',')
Vio_DUST_N30 = np.loadtxt('logs/diffN/Vio_DUST_N30.csv', delimiter=',')

plt.figure()
plt.plot(Reg_DUST_N10, label=r'$N=10$', linestyle='-')
plt.plot(Reg_DUST_N20, label=r'$N=20$', linestyle='-')
plt.plot(Reg_DUST_N30, label=r'$N=30$', linestyle='-')
plt.xlim(0, 300)
plt.ylim(bottom=0.85e1)
plt.yscale('log')
plt.legend(fontsize=40)
plt.xlabel(r'$T$', fontsize=45)
plt.ylabel(r'$\mathrm{Reg}(T)/T$', fontsize=45)
plt.savefig(f'{save_dir}/diffN_reg.pdf', bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(Vio_DUST_N10, label=r'$N=10$', linestyle='-')
plt.plot(Vio_DUST_N20, label=r'$N=20$', linestyle='-')
plt.plot(Vio_DUST_N30, label=r'$N=30$', linestyle='-')
plt.xlim(0, 300)
plt.yscale('log')
plt.legend(fontsize=40)
plt.xlabel(r'$T$', fontsize=45)
plt.ylabel(r'$\mathrm{Reg}^c (T)/T$', fontsize=45)
plt.savefig(f'{save_dir}/diffN_vio.pdf', bbox_inches='tight')
plt.close()

# fig 3: compare algorithms
Reg_DUST = np.loadtxt('logs/comp/Reg_DUST.csv', delimiter=',')
Reg_DOPP = np.loadtxt('logs/comp/Reg_DOPP.csv', delimiter=',')
Reg_DS = np.loadtxt('logs/comp/Reg_DS.csv', delimiter=',')

Vio_DUST = np.loadtxt('logs/comp/Vio_DUST.csv', delimiter=',')
Vio_DOPP = np.loadtxt('logs/comp/Vio_DOPP.csv', delimiter=',')
Vio_DS = np.loadtxt('logs/comp/Vio_DS.csv', delimiter=',')

plt.figure()
plt.plot(Reg_DUST, label=r'$\mathrm{DUST}$', linestyle='-')
plt.plot(Reg_DOPP, label=r'$\mathrm{DOPP}$', linestyle='-.')
plt.plot(Reg_DS, label=r'$\mathrm{Dual\ Subgradient}$', linestyle='--')
plt.xlim(0, 300)
plt.yscale('log')
plt.legend(fontsize=40)
plt.xlabel(r'$T$', fontsize=45)
plt.ylabel(r'$\mathrm{Reg}(T)/T$', fontsize=45)
plt.savefig(f'{save_dir}/comp_reg.pdf', bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(Vio_DUST, label=r'$\mathrm{DUST}$', linestyle='-')
plt.plot(Vio_DOPP, label=r'$\mathrm{DOPP}$', linestyle='-.')
plt.plot(Vio_DS, label=r'$\mathrm{Dual\ Subgradient}$', linestyle='--')
plt.xlim(0, 300)
plt.yscale('log')
plt.legend(fontsize=40)
plt.xlabel(r'$T$', fontsize=45)
plt.ylabel(r'$\mathrm{Reg}^c (T)/T$', fontsize=45)
plt.savefig(f'{save_dir}/comp_vio.pdf', bbox_inches='tight')
plt.close()