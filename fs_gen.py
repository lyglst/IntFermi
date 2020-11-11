import numpy as np
from mcu.wannier90 import pywannier90_vasp as pyw90
import interp_wav

def get_b(w90, k_vec, eig_vec, nkx=4, nky=4, nkz=4, supercell=[3,3,3], grid=None):
    nb = w90.num_wann
    supercell = np.array(supercell)
    grid = np.array(grid)
    psi_w_up, psi_w_down = interp_wav.get_psi_from_wannier(w90, k_vec, supercell, grid)
    psi_H_up, psi_H_down = interp_wav.get_psi_ham(w90, k_vec, [psi_w_up, psi_w_down], supercell, grid)
    spin = interp_wav.get_spin(psi_H_up, psi_H_down)
    return np.reshape(spin, (nkx, nky, nkz, nb))

def grid_gen(w90, nkx=4, nky=4, nkz=4, filename='fs.in'):
    with open(filename, 'w') as f:
        nb = w90.num_wann
        f.write('{0:4d} {1:4d} {2:4d} \n'.format(nkx, nky, nkz))
        f.write('1\n')
        f.write('{0:4d} \n'.format(nb))
        rec_lat = w90.recip_lattice_loc
        for i in range(3):
            f.write('{0:.12f} {1:.12f} {2:.12f}\n'.format(rec_lat[i][0], rec_lat[i][1], rec_lat[i][2]))
        k_vec = [[[[(i-1)/nkx, (j-1)/nky, (k-1)/nkz] for k in np.arange(1, nkz+1)] for j in np.arange(1, nky+1)] for i in np.arange(1, nkx+1)]
        k_vec = np.reshape(k_vec, (-1, 3))
        eig, eig_vec = w90.interpolate_band(k_vec)
        eig = np.array(eig)
        eig = eig.real.astype(np.float)
        eig = np.reshape(eig, (nkx, nky, nkz, nb))
        for ib in range(nb):
            for ik1 in range(nkx):
                for ik2 in range(nky):
                    for ik3 in range(nkz):
                        f.write('{0:.12f}\n'.format(eig[ik1][ik2][ik3][ib]))
        if w90.spinors:
            bs = get_b(w90, k_vec, eig_vec, nkx, nky, nkz)
            for ib in range(nb):
                for ik1 in range(nkx):
                    for ik2 in range(nky):
                        for ik3 in range(nkz):
                            f.write('{0:.12f}\n'.format(bs[ik1][ik2][ik3][ib]))
