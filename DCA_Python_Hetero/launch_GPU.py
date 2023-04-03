from multiprocessing import Pool, current_process, Queue
import sys
#import cupy as cp
import time 
import dsmulti_GPU as ds
from functools import partial

# method to run processes and different cases on GPU
def dca(instance, case_id):

    # run processing on GPU <gpu_id>
    pid = current_process().ident
    #print('process {}: starting on GPU. '.format(pid),'to solve case:', case_id)
    
    # deal with fault cases on GPU
    # method on GPU to compute reduced_Y matrices
    instance.reduced_Y(case_id) # case_id

    # method on GPU to perform numerical integration
    mac_ang,mac_spd = instance.reduced_Y_solver() # case_id
    #print('process {}: finished on GPU. '.format(pid))

    return mac_ang, mac_spd


if __name__ == "__main__":

    t1 = time.time()
    # define the number of GPUs used and processes launched on each GPU
    PROC_PER_GPU = int(sys.argv[-1])
    CASE_PER_GPU = int(sys.argv[-2])
    GID = int(sys.argv[-4])
    #queue = Queue()

    # initialize the queue with the GPU ids
    #for gpu_ids in range(NUM_GPUS):
        #for _ in range(PROC_PER_GPU):
            #queue.put(gpu_ids)

    # create process pool, distribute fault cases
    pool = Pool(processes = PROC_PER_GPU)
    fault_cases = [x for x in range(GID*CASE_PER_GPU, (GID+1)*CASE_PER_GPU)]

    # launch a global instance for DCA, initialize the global variables on GPU
    DCA = ds.DynSim_GPU(sys.argv[-3], 1, 2, 3)
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
    print('Global initialization in {:.2f} s'.format(t3 - t1), 'on GPU', GID)
    print('Local solving DAEs in {:.2f} s'.format(t2-t3), 'on GPU', GID)
    print('Program finished in {:.2f} s' .format(t2 - t1), 'on GPU', GID)
    print('------------------------------------------------------')