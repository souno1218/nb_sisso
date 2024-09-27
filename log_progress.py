import time
import threading
import logging
import numpy as np
from numba import int64
from numba.experimental import jitclass
from numba_progress.numba_atomic import atomic_add
        
class loop_log:
    def __init__(self,logger=None,interval=5,tot_loop=None,header="",footer=""):    
        if logger is None:
            self.logger=logging.getLogger("loop_log")
            self.logger.setLevel(logging.DEBUG)
            for h in self.logger.handlers[:]:
                logger.removeHandler(h)
                h.close()
            st_handler = logging.StreamHandler()
            st_handler.setLevel(logging.INFO)
            _format = "%(asctime)s %(name)s [%(levelname)s] : %(message)s"
            st_handler.setFormatter(logging.Formatter(_format))
            self.logger.addHandler(st_handler)
        else:
            self.logger=logger

        self.is_in_progress=False
        self.interval=interval
        self.counter=jit_counter()
        self.str_tot_loop="" if tot_loop is None else f"/{tot_loop}"
        self.header=header
        self.footer=footer
    
    def out_log(self):
        time.sleep(self.interval)
        while self.is_in_progress:
            self.logger.info(self.header+f"{str(self.counter.count[0]).rjust(len(self.str_tot_loop))}{self.str_tot_loop}"+self.footer)
            time.sleep(self.interval)
        
    def __enter__(self):
        self.is_in_progress=True
        thread = threading.Thread(target=self.out_log)
        thread.start()
        return self.counter
        
    def __exit__(self,*args):
        self.is_in_progress=False

spec = [('count', int64[:])]
@jitclass(spec)
class jit_counter:
    def __init__(self):
        self.count=np.zeros(1,dtype="int64")
    def update(self,n):
        atomic_add(self.count,0,n)
        