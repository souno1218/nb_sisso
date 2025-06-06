#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numba import int64
import time, datetime, logging, threading, os, psutil
from numba.experimental import jitclass
from numba_progress.numba_atomic import atomic_add


class loop_log:
    def __init__(self, logger=None, interval=5, tot_loop=None, header="", footer="", print_mem=True):
        if logger is None:
            self.logger = logging.getLogger("loop_log")
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
            self.logger = logger

        self.is_in_progress = False
        self.interval = interval
        self.counter = jit_counter()
        self.print_mem = print_mem
        if tot_loop is None:
            self.is_tot_loop = False
        else:
            if not isinstance(tot_loop, int):
                raise TypeError(f"Expected variable 'tot_loop' to be of type int, but got {type(tot_loop)}.")
            self.is_tot_loop = True
            self.tot_loop = tot_loop
            self.str_tot_loop = str(tot_loop)
        self.header = header
        self.footer = footer

    def out_log(self):
        time.sleep(self.interval)
        while self.is_in_progress:
            dtime = datetime.datetime.now() - self.start_time
            count = self.counter.count[0]
            txt = ""
            if self.is_tot_loop:
                if count != 0:
                    txt += f"{str(count).rjust(len(self.str_tot_loop))}/{self.str_tot_loop}  "
                else:
                    txt += f"{str(count).rjust(len(self.str_tot_loop))}/{self.str_tot_loop}  "
            else:
                txt += f"{str(count)}  "
            if self.print_mem:
                txt += f"({str_using_mem()})  "
            txt += f"{dtime} "
            if self.is_tot_loop:
                if count != 0:
                    left_time = ((self.tot_loop - count) / count) * dtime
                    txt += f": {left_time}"
                else:
                    txt += f": inf"
            self.logger.info(self.header + txt + self.footer)
            time.sleep(self.interval)

    def __enter__(self):
        self.is_in_progress = True
        self.start_time = datetime.datetime.now()
        thread = threading.Thread(target=self.out_log)
        thread.start()
        return self.counter

    def __exit__(self, *args):
        self.is_in_progress = False


spec = [("count", int64[:])]


@jitclass(spec)
class jit_counter:
    def __init__(self):
        self.count = np.zeros(1, dtype="int64")

    def update(self, n):
        atomic_add(self.count, 0, n)


def str_using_mem():
    using_mem = psutil.Process(os.getpid()).memory_info().rss
    return f"{using_mem /(1024**2):.1f} MB"
