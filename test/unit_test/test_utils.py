import multiprocessing as mp
import logging
import traceback
# Configure logging
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s - %(levelname)s - %(message)s')



def wrap_try_func(func):
    def try_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BaseException as e:
            traceback.print_exc()
    return try_func

def spawn(nprocs, func, *args):
    # mp.set_start_method('spawn')
    # size = 4
    processes = []
    q = mp.Queue()
    for i in range(nprocs):
        p = mp.Process(target=func,args=(i, nprocs, q) + args)
        p.start()
        processes.append(p)

    for p in processes:
        p.join(100)
    
    return q
