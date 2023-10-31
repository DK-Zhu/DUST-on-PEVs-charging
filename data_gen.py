import numpy as np
import os

num_of_nodes = 10
d = 24

# Generate cit for max T.
Tmax = 200
C = np.random.uniform(low=0., high=1., size=(num_of_nodes, Tmax, d))
np.save('C.npy', C)

# Generate rit. rit is the parameter of the quadratic term (rit/2).
R = np.random.uniform(low=0.5, high=1., size=(num_of_nodes, Tmax))
np.save('R.npy', R)

# Generate Di. Di remains the same for each node i. Di is 48 * 24
Di = np.concatenate((np.identity(d), -np.identity(d)), axis=0)
np.save('Di.npy', Di)

# Generate b. Note: gi(x) = Di@x - bi, i.e., bi = b/N.
bi = np.concatenate((0.35*np.ones(d), 0.35*np.ones(d)), axis=None)
np.save('bi.npy', bi)

# Generate local constraint Xi: lb <= Ai@xit <= ub.
lb = np.concatenate((-2.0*np.ones(d), np.zeros(d)), axis=None)
lb[-1] = 8.
Ai = np.concatenate((np.identity(d), np.tril(np.ones((d,d)), k=0)), axis=0)
ub = np.concatenate((2.0*np.ones(d), 10.*np.ones(d)), axis=None)

save_dir = 'data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

np.save(f'{save_dir}/C.npy', C)
np.save(f'{save_dir}/R.npy', R)
np.save(f'{save_dir}/Di.npy', Di)
np.save(f'{save_dir}/bi.npy', bi)
np.save(f'{save_dir}/lb.npy', lb)
np.save(f'{save_dir}/Ai.npy', Ai)
np.save(f'{save_dir}/ub.npy', ub)