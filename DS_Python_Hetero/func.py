def m_ang_3step_ab(d_y, y, v, n_step, step_w, basrad):
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


def m_spd_3step_ab(d_y, y, pmech, mva, pelect, g_do, g_H, n_step, step_w):
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


def solver(A, B, command_1):
    if command_1 == 'gpu':
        from cupyx.scipy.sparse.linalg import splu
        return splu(A).solve(B.T.toarray())
    else:
        from scipy.sparse.linalg import spsolve
        return spsolve(A, B.T)

    
def sparse_matrix(command_1):
    if command_1 == 'gpu':
        from cupyx.scipy.sparse import csr_matrix, csc_matrix
        return csr_matrix, csc_matrix
    else:
        from scipy.sparse import csc_matrix, csr_matrix
        return csr_matrix, csc_matrix

    
def methods(command_1):
    if command_1 == 'gpu':
        import cupy as cp
        from cupy import array
        cuda_setup=cp.empty(1)
        complex128=cp.complex128
        float64=cp.float64
        count_nonzero=cp.count_nonzero
        where=cp.where
        ones=cp.ones
        zeros=cp.zeros
        arange=cp.arange
        exp=cp.exp
        a_append=cp.append
        eye=cp.eye
        logical_and=cp.logical_and
    else:
        import numpy as np
        from numpy import array
        complex128=np.complex128
        float64=np.float64
        count_nonzero=np.count_nonzero
        where=np.where
        ones=np.ones
        zeros=np.zeros
        arange=np.arange
        exp=np.exp
        a_append=np.append
        eye=np.eye
        logical_and=np.logical_and
        
    return complex128, float64, count_nonzero, where, ones, zeros, arange, exp, a_append, eye, logical_and, array


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