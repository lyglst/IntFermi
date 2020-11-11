import numpy as np

def get_psi_from_wannier(w90, k_vecs, supercell, grid):
    '''
    Inverse FFT from wannier function to wavefunction $psi^w$
    to be noticed, $psi^w$ is not eigenfunction of hamiltonian
    Args: 
            w90         :       pywannier90 object
            k_vecs      :       the k-vector list that is being interpolated
            supercell   :       the supercell used for wannier function
            grid        :       the grid density for wannier function
    '''
    grid = np.asarray(grid)
    supercell = np.asarray(supercell)
    k_vecs = np.asarray(k_vecs)
    origin = np.asarray([-(grid[i]*(supercell[i]//2) + 1)/grid[i] for i in range(3)]).dot(w90.real_lattice_loc)       
    real_lattice_loc = (grid*supercell-1)/grid * w90.real_lattice_loc
    nx, ny, nz = grid*supercell
    nband = w90.num_wann

    if w90.spinors:
        wann_up = w90.get_wannier_mod(spinor_mode='up', supercell=supercell, grid=grid).reshape(np.append(grid*supercell, -1))
        wann_down = w90.get_wannier_mod(spinor_mode='down', supercell=supercell, grid=grid).reshape(np.append(grid*supercell, -1))
        #wann_up = w90.get_wannier(spinor_mode='up', supercell=supercell, grid=grid).reshape(np.append(grid*supercell, -1))
        #wann_down = w90.get_wannier(spinor_mode='down', supercell=supercell, grid=grid).reshape(np.append(grid*supercell, -1))
        lpsi_w_up = []
        lpsi_w_down = []
        for kpt in k_vecs:
            x = np.arange(0, supercell[0])
            y = np.arange(0, supercell[1])
            z = np.arange(0, supercell[2])
            xv, yv, zv = np.meshgrid(x, y, z, sparse=False, indexing='ij')
            mesh = np.array(list(zip(xv.flat, yv.flat, zv.flat)))
            mesh_grid = mesh*grid
            #mesh_R = (mesh - supercell//2).dot(w90.real_lattice_loc)
            #phase = np.exp(1.j*kpt.dot(w90.recip_lattice_loc).dot(mesh_R.T))
            phase = np.zeros(grid*supercell, dtype=np.complex128)
            for ix in range((grid*supercell)[0]):
                for iy in range((grid*supercell)[1]):
                    for iz in range((grid*supercell)[2]):
                        phase[ix, iy, iz] = np.exp(
                            1.j*kpt.dot(w90.recip_lattice_loc).dot(
                                (np.array([ix-1, iy-1, iz-1])//grid - supercell//2).dot(w90.real_lattice_loc)))

            psi_w_up = np.zeros(np.append(grid, nband),  dtype=np.dtype('complex128'))
            psi_w_down = np.zeros(np.append(grid, nband),  dtype=np.dtype('complex128'))
            for ix in range(grid[0]):
                for iy in range(grid[1]):
                    for iz in range(grid[2]):
                        for imesh in range(len(mesh_grid)):
                            #psi_w_up[ix, iy, iz] += phase[imesh]*wann_up[ix+mesh_grid[imesh][0]][iy+mesh_grid[imesh][1]][iz+mesh_grid[imesh][2]]
                            #psi_w_down[ix, iy, iz] += phase[imesh]*wann_down[ix+mesh_grid[imesh][0]][iy+mesh_grid[imesh][1]][iz+mesh_grid[imesh][2]]
                            iix = ix + 1 + mesh_grid[imesh][0]
                            iiy = iy + 1 + mesh_grid[imesh][1]
                            iiz = iz + 1 + mesh_grid[imesh][2]
                            if all(np.array([iix, iiy, iiz])<grid*supercell):
                                psi_w_up[ix, iy, iz] += phase[iix, iiy, iiz] * wann_up[iix, iiy, iiz]
                                psi_w_down[ix, iy, iz] += phase[iix, iiy, iiz] * wann_down[iix, iiy, iiz]
            lpsi_w_up.append(psi_w_up)
            lpsi_w_down.append(psi_w_down)
        return lpsi_w_up, lpsi_w_down

def get_psi_ham(w90, k_vecs, psi_w, supercell, grid):
    '''
    transform the wavefunction $psi_w$ to hamiltonian basis
    Args: 
            w90         :       pywannier90 object
            k_vecs      :       the k-vector list that is being interpolated
            psi_w       :       wavefunction $psi_w$
            supercell   :       the supercell used for wannier function
            grid        :       the grid density for wannier function
    '''
    
    if w90.spinors:
        grid = np.asarray(grid)
        supercell = np.asarray(supercell)
        k_vecs = np.asarray(k_vecs)
        psi_w_up, psi_w_down = psi_w
        eig, eigv = w90.interpolate_band(k_vecs, ws_search_size=[4, 4, 4])
        psi_H_up = np.einsum('pijkm, pmn->pijkn',psi_w_up,eigv)
        psi_H_down = np.einsum('pijkm, pmn->pijkn',psi_w_down,eigv)
        return psi_H_up, psi_H_down

def get_spin(psi_H_up, psi_H_down):
    spin = []
    Sx = np.array([[0, 1], [1, 0]])
    Sy = np.array([[0, -1.j], [1.j, 0]])
    Sz = np.array([[1, 0], [0, 1]])
    for ik in range(len(psi_H_up)):
        spin.append([])
        for iband in range(psi_H_up[ik].shape[3]):
            #alpha = np.sum(np.abs(psi_H_up[ik][:,:,:,iband])**2)
            #beta = np.sum(np.abs(psi_H_down[ik][:,:,:,iband])**2)
            #spin[ik].append((alpha-beta)/(alpha+beta))
            ak = np.array(psi_H_up[ik][:,:,:,iband].flat)
            bk = np.array(psi_H_down[ik][:,:,:,iband].flat)
            norm = np.dot(np.conj(ak), ak) + np.dot(np.conj(bk), bk)
            Sz = np.sum([np.matmul(np.matmul(np.conj(np.array([ak[i], bk[i]])), Sz), np.array([[ak[i]],[bk[i]]])) for i in range(len(ak))])/norm
            Sx = np.sum([np.matmul(np.matmul(np.conj(np.array([ak[i], bk[i]])), Sx), np.array([[ak[i]],[bk[i]]])) for i in range(len(ak))])/norm
            Sy = np.sum([np.matmul(np.matmul(np.conj(np.array([ak[i], bk[i]])), Sy), np.array([[ak[i]],[bk[i]]])) for i in range(len(ak))])/norm
            spin[ik].append(float(1-Sx**2-Sy**2-Sz**2))
    return spin

def is_orth(a, b):
    prod = np.conj(a.flat).dot(np.array(b.flat))
    print('The product is {0}'.format(prod))
