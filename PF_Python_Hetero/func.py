import cupy as cp
import numpy as np

def m_ang_ab(d_y, y, v, n_step, step_w, basrad):
    c_step = n_step-1
    
    if c_step < 2:
        k1_y = step_w*d_y[c_step]
        k2_y = step_w*(basrad*(v+k1_y/2.0-1.0))
        k3_y = step_w*(basrad*(v+k2_y/2.0-1.0))
        k4_y = step_w*(basrad*(v+k3_y-1.0))
        y[n_step] = y[c_step]+(k1_y+2*k2_y+2*k3_y+k4_y)/6.0
    else:
        y[n_step] = y[c_step]+step_w*(23*d_y[c_step]-16*d_y[c_step-1]+5*d_y[c_step-2])/12.0
        
    return y


def m_spd_ab(d_y, y, pmech, mva, pelect, g_do, g_H, n_step, step_w):
    c_step = n_step-1
    
    if c_step < 2:
        k1_y = step_w*d_y[c_step]
        k2_y = step_w*(pmech-mva*pelect-g_do*(y[c_step]+k1_y/2.0-1.0))/(2.0*g_H)
        k3_y = step_w*(pmech-mva*pelect-g_do*(y[c_step]+k2_y/2.0-1.0))/(2.0*g_H)
        k4_y = step_w*(pmech-mva*pelect-g_do*(y[c_step]+k3_y-1.0))/(2.0*g_H)
        y[n_step] = y[c_step]+(k1_y+2*k2_y+2*k3_y+k4_y)/6.0
    else:
        y[n_step] = y[c_step]+step_w*(23*d_y[c_step]-16*d_y[c_step-1]+5*d_y[c_step-2])/12.0
    
    return y


def sp_mat(command_1):
    if command_1 == 'gpu':
        from cupyx.scipy.sparse import csr_matrix, csc_matrix
        return csr_matrix, csc_matrix
    else:
        from scipy.sparse import csc_matrix, csr_matrix
        return csr_matrix, csc_matrix
    
    
def solver(A, B, command_1):
    if command_1 == 'gpu':
        csr_matrix, csc_matrix = sp_mat(command_1)
        from cupyx.scipy.sparse.linalg import splu
        from cupyx.scipy.sparse import isspmatrix
        if not isspmatrix(A):
            A=csc_matrix(A)

        if isspmatrix(B):
            B=B.toarray()
        else:
            B=csc_matrix(B).toarray()

        return csc_matrix(splu(A).solve(B))
    else:
        from scipy.sparse.linalg import spsolve
        return spsolve(A, B)

    
def stack(command_1):
    if command_1 == 'gpu':
        from cupyx.scipy.sparse import vstack, hstack
        return vstack, hstack
    else:
        from scipy.sparse import vstack, hstack
        return vstack, hstack

    
def methods(command_1):
    import numpy as np
    if command_1 == 'gpu':
        np = cp
        #cuda_setup=np.empty(1)

    return np.complex128, np.float64, np.count_nonzero, np.where, np.ones, np.zeros, np.arange, np.exp, np.append, np.eye, np.logical_and, np.array, np.amax


def parser(command_2, array, dtype):
    n=0
    m=0
    x=[]
    
    with open("input/"+command_2+'.txt') as f:
        for line in f:
            line = line.split()
            if len(line)==1 and m==0:
                nb_1=n
                a=x
                m=m+1
                x=[]
                continue
            elif len(line)==1 and m==1:
                nb_2=n-nb_1
                b=x
                m=m+1
                x=[]
                continue
            elif len(line)==1 and m==2:
                nb_3=n-nb_2-nb_1
                c=x
                m=m+1
                x=[]
                continue
            elif len(line)==1 and m==3:
                nb_4=n-nb_3-nb_2-nb_1
                d=x
                continue
            x.append(line)
            n=n+1
            
    b_1=array(a,dtype=dtype)
    b_2=array(b,dtype=dtype)
    b_3=array(c,dtype=dtype)
    b_4=array(d,dtype=dtype)
    
    return b_1, b_2, b_3, b_4, nb_1, nb_2, nb_3, nb_4


def array_partition(array, npart):
    #build a function to partition the related data
    split_array = np.array_split(array,npart,axis=0)
    
    #create a list to store each splitted array's size
    array_sizes = []
    for i in range(0,len(split_array),1):
        array_sizes = np.append(array_sizes, len(split_array[i])).astype(int)

    #for indexing or slicing ngen arrays
    absolute_ps = np.insert(np.cumsum(array_sizes), 0, 0)
    
    return array_sizes, absolute_ps