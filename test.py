import sys
from mcu.wannier90 import pywannier90_vasp as pyw90
from fs_gen import grid_gen
import subprocess 

nk = [8, 8, 8] 
num_wann = 10
keywords = \
'''
Begin Projections
  Al: s
  Al: s
  Al: p
End Projections

# restart = plot
 bands_plot = true

begin kpoint_path
GM 0 0 0 X 0.5 0 0.5
X 0.5 0 0.5 W 0.5 0.25 0.75
W 0.5 0.25 0.75 K 0.375 0.375 0.75
end kpoint_path

bands_num_points 50
bands_plot_format gnuplot xmgrace

dis_win_max = 30
dis_froz_max = 13.5
dis_froz_min = -3.5
dis_win_min = -3.5

dis_num_iter = 5000
num_iter = 5000

'''

#w90 = pyw90.W90(nk, num_wann, spinors=True, other_keywords = None)
vasprun = pyw90.mcu.vasp.vasprun.main('.')
#vasprun.soc = True
w90 = pyw90.W90(vasprun=vasprun, mp_grid=nk, num_wann=num_wann, other_keywords = keywords)

w90.setup()
w90.M_matrix_loc = w90.read_M_mat()
w90.A_matrix_loc = w90.read_A_mat()
w90.eigenvalues_loc = w90.read_epsilon_mat()
w90.make_win()
print('initialization finished')
#w90.setup()
#w90.export_AME(grid=[24, 24, 24])
w90.run()
print('Wannierization finished')
#w90.plot_wf(outfile='MLWF_total', supercell = [3,3,3], grid=[32,32,32])
#w90.plot_wf(outfile='MLWF_up', spinor_mode = 'up', supercell = [3,3,3], grid=[32,32,32])
#w90.plot_wf(outfile='MLWF_down', spinor_mode = 'down', supercell = [3,3,3], grid=[32,32,32])


grid_gen(w90, 10, 10, 10)
#A0 = w90.read_A_mat()
#A1 = w90.get_A_mat()
#A2 = w90.get_M_mat()
#for i in range(27):
#    print((A0[i].real/A0[i].imag).sum(), (A1[i].real/A1[i].imag).sum())
#subprocess.call('rm wannier90.wout', shell=True)
#w90.kernel()
#subprocess.call('grep "================       Omega D      =" wannier90.wout', shell=True)
#subprocess.call('sed -i -e "s/num_bands       = 8/num_bands       = 4/g" wannier90.win', shell=True)

# '''Test the unk export here and from VASP'''
# import mcu
# import numpy as np
# from mcu.vasp import pywannier90_vasp as pyw90
# wave = mcu.WAVECAR()
# ngrid = [16,16,16]
# kpt = 5
# coords, weights = pyw90.periodic_grid(wave.cell[0], ngrid, supercell = [1,1,1], order = 'F')
# exp_ikr = np.exp(1j*coords.dot(wave.kpts[kpt].dot(wave.cell[1]))).reshape(ngrid, order = 'F') 
# u0 = wave.get_unk(kpt=kpt+1, band=2, ngrid=ngrid)
# psi0 = np.einsum('xyz,xyz->xyz', exp_ikr, u0, optimize = True)
# wave.write_vesta(u0,filename='unk')
# wave.write_vesta(psi0,filename='psi')



# for kpt in range(27):
    # for band in range(4):
        # u0 = wave.get_unk(kpt=kpt+1, band=band+1, ngrid=ngrid)
        # u1 = mcu.read_unk(path='./unk', kpt=kpt+1, band=band+1)
        # print("Error:", abs(u0-u1).max())
        
