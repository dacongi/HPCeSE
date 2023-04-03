import multiprocessing
import sys
import numpy as np
import time 
import dsmulti_CPU as ds
from functools import partial

# method to run processes and different cases on CPU
def dca(instance, case_id):

    # run processing on CPU
    ident = multiprocessing.current_process().ident
    #print('process {}: starting on CPU. '.format(ident))
    
    # deal with fault cases on CPU
    instance.reduced_Y(case_id) # case_id
    
    # method on CPU to perform numerical integration
    mac_ang,mac_spd = instance.reduced_Y_solver() # case_id
    #print('process {}: finished. '.format(ident))
    return mac_ang, mac_spd


if __name__ == "__main__":
    
    t1 = time.time()
    # define the number of processes launched on CPU
    PROC_ON_CPU = int(sys.argv[-1])

    pool = multiprocessing.Pool(processes = PROC_ON_CPU)
    fault_cases = [x for x in range(int(sys.argv[-2]))]

    # launch a global instance for dynamic simulation and initialize the matrices and variables on CPU
    DCA = ds.DynSim_CPU(sys.argv[-3], 1, 2, 3)
    DCA.init()

    t3 = time.time()
    # pass the function and iterables to the processes
    func = partial(dca, DCA)
    for _ in pool.imap_unordered(func, fault_cases):
        pass

    pool.close()
    pool.join()

    t2 = time.time()
    print('------------------------------------------------------')
    print('Global initialization in {:.2f} s'.format(t3 - t1))
    print('Local solving DAEs in {:.2f} s'.format(t2-t3))
    print('Program finished in {:.2f} s' .format(t2 - t1))
    print('------------------------------------------------------')
