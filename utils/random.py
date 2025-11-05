import contextlib
import numpy as np
import random

@contextlib.contextmanager
def temp_seed(seed):
    np_state = np.random.get_state()
    np.random.seed(seed)

    py_state = random.getstate()
    random.seed(int(seed))
    
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)



def gen_item_seeds(n, seed):
    with temp_seed(seed):
        item_seeds = np.random.randint(0, 2**31 - 1, size=n)
    return item_seeds