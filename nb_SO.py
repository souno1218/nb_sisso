#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import comb
from log_progress import loop_log
from numba_progress import ProgressBar
import datetime,logging,multiprocessing
from numba import njit,prange,set_num_threads,objmode
from utils import thread_check,calc_RPN,argmin_and_min,eq_list_to_num,raise_and_log,dtype_shape_check,type_check

def SO(
    list_x,
    y,
    score_func,
    which_arr_to_choose_from,
    combination_dim=2,
    num_threads=None,
    how_many_to_save=50,
    verbose=True,
    is_progress=False,
    log_interval=10,
    logger=None):
    """
    select all conmbinations from the given ndarray(list_x), throw them into score_func, and store how_many_to_save in order of size.
    
    Parameters
    ----------        
    list_x : list of ndarray of shape (n_saved_in_SIS,n_samples)
        Training data.
        Even if you choose more than one (e.g., two) from a single array, please list.
        which_arr_to_choose_from sets which elements of the list are combined.
        
    y : ndarray of shape (n_samples)
        Target values.
        
    score_func : callable
        func that returns a score,jit compilation by numba is required.
        Parameters
            - x : ndarray of shape (combination_dim,n_samples)
                Combination created from List_x.
            - y : ndarray of shape (n_samples)
                Target values.
        Returns
            - score : float
                Values for which greater is better, such as negative mean-square error or accuracies.
                
    which_arr_to_choose_from : dict
        Decide here how many to choose.
        keys set all combination_dim from 1({1:0,~~combination_dim:0).
        Key starts at 1.
        Values specify the index of list_x.
        ex) combination_dim=3 , {1:1,2:2,3:3}

    combination_dim : int
        Number of equations to combine.
        
    num_threads : int
        Number of CPU cores used. If not set, all cpu cores are used.
        
    how_many_to_save : int
        Specify how many combinations to save up to the top.
        
    verbose : bool
        An optional parameter that will be ignored if logger is set.
        Print log or.
        
    is_progress : bool
        Use progress bar or.
        
    log_interval : int or float
        An optional parameter.This will be ignored if is_progress is True.
        Time interval for writing out progress.
    
    logger : logging.Logger
        A logger instance to handle logging. It is expected to be a standard.
        Python `logging.Logger` instance.

    Returns
    ----------
    score_list : ndarray of shape (how_many_to_save)
        sorted score
        
    index_list : ndarray of shape (how_many_to_save,combination_dim)
        index in list_x corresponding to score_list
    """
    Nan_number=-100
    
    #logger
    if logger is None:
        logger = logging.getLogger("SO")
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            h.close()
        if verbose:
            st_handler = logging.StreamHandler()
            st_handler.setLevel(logging.INFO)
            #_format = "%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"
            _format = "%(asctime)s %(name)s [%(levelname)s] : %(message)s"
            st_handler.setFormatter(logging.Formatter(_format))
            logger.addHandler(st_handler)

    #num_threads
    if num_threads is None:
        num_threads=multiprocessing.cpu_count()
        
    set_num_threads(num_threads)
    if not thread_check(num_threads):
        logger.info(f"can't set thread : {num_threads}")
        
    #x
    type_check(logger,list_x,"list_x",list)
    n_features=list_x[0].shape[1]
    for i in range(len(list_x)):
        dtype_shape_check(logger,list_x[i],f"list_x[{i}]",dtype_=np.float64,ndim=2,dict_index_len={1:n_features})

    max_x_len=np.max([i.shape[0] for i in list_x])
    arr_x=np.full((len(list_x),max_x_len,list_x[0].shape[1]),Nan_number,dtype="float64")
    for i in range(len(list_x)):
        arr_x[i,:list_x[i].shape[0]]=list_x[i]

    #which_arr_to_choose_from
    arr_which_arr_to_choose_from=np.empty(combination_dim,dtype="int64")
    for i in range(combination_dim):
        arr_which_arr_to_choose_from[i]=which_arr_to_choose_from[i+1]

    logger.info("SO")
    logger.info(f"num_threads={num_threads}, how_many_to_save={how_many_to_save}, ")
    logger.info(f"combination_dim={combination_dim}, score_func={score_func.__name__}, ")
    logger.info(f"which_arr_to_choose_from={which_arr_to_choose_from}")
    repeat=loop_counter(arr_x,arr_which_arr_to_choose_from)
    logger.info(f"loop={repeat}")
    
    time0=datetime.datetime.now()
    if is_progress:
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        with ProgressBar(total=repeat,dynamic_ncols=False,bar_format=bar_format,leave=False) as progress:
            score_list,index_list=SO_loop(num_threads,arr_x,y,how_many_to_save,score_func,arr_which_arr_to_choose_from,progress)
    else:
        with loop_log(logger,interval=log_interval,tot_loop=repeat,header="  ") as progress:
            score_list,index_list=SO_loop(num_threads,arr_x,y,how_many_to_save,score_func,arr_which_arr_to_choose_from,progress) 
    dtime=datetime.datetime.now()-time0
    logger.info(f"  END : time={dtime}")
    logger.info(f"best : score={score_list[0]},index={index_list[0]}")
    return score_list,index_list


def type_check_which_arr_to_choose_from(logger,combination_dim,which_arr_to_choose_from,list_x):
    type_check(logger,which_arr_to_choose_from,"which_arr_to_choose_from",dict)
    keys=list(which_arr_to_choose_from.keys())
    for x in keys:
        type_check(logger,x,"which_arr_to_choose_from.key",int)
    if 0 in keys:
        raise#raise_and_log(logger,
    if sorted(keys)!=list(range(1,max(keys))):
        raise#raise_and_log(logger,
    values=list(which_arr_to_choose_from.values())
    if not all(isinstance(x, int) for x in values):
        raise#raise_and_log(logger,
    if max(values)>len(list_x)-1:
        raise#raise_and_log(logger,

def loop_counter(arr_x,arr_which_arr_to_choose_from):
    unique_arr=np.unique(arr_which_arr_to_choose_from)
    num=1
    for i in unique_arr:
        same=np.sum(arr_which_arr_to_choose_from==i)
        num*=comb(arr_x[i].shape[0],same, exact=True)
    return num

@njit(error_model="numpy")
def make_check_list(arr_which_arr_to_choose_from):
    return_arr=np.full((arr_which_arr_to_choose_from.shape[0],2),10000,dtype="int64")
    unique_arr=np.unique(arr_which_arr_to_choose_from)
    arange=np.arange(arr_which_arr_to_choose_from.shape[0])
    n=0
    for i in unique_arr:
        same_index=arange[arr_which_arr_to_choose_from==i]
        if same_index.shape[0]>1:
            sorted_index=np.sort(same_index)
            for i in range(sorted_index.shape[0]-1):
                return_arr[n]=sorted_index[i:i+2]
                n+=1
    index=np.argsort(return_arr[:,1])
    return return_arr[index]

@njit(error_model="numpy")
def make_index_arr(number,check_list,len_x_arr,arr_which_arr_to_choose_from,index_arr):
    n=0
    for i in range(arr_which_arr_to_choose_from.shape[0]):
        index_arr[i]=number%len_x_arr[i]
        if check_list[n,1]==i:
            if index_arr[check_list[n,0]]<=index_arr[i]:
                return False
            else:
                n+=1
        number//=len_x_arr[i]
    return True

@njit(parallel=True,error_model="numpy")#,fastmath=True
def SO_loop(num_threads,arr_x,y,how_many_to_save,score_func,arr_which_arr_to_choose_from,progress_proxy):
    Nan_number=-100
    how_many_to_choose=arr_which_arr_to_choose_from.shape[0]
    len_x_arr=np.array([np.sum(arr_x[i,:,0]!=Nan_number) for i in arr_which_arr_to_choose_from])
    repeat=np.prod(len_x_arr)
    
    score_list_thread=np.full((num_threads,int(how_many_to_save)),np.finfo(np.float64).min,dtype="float64")
    index_list_thread=np.full((num_threads,int(how_many_to_save),how_many_to_choose),Nan_number,dtype="int64")
    for thread_id in prange(num_threads):
        score_list=np.full((int(how_many_to_save)),np.finfo(np.float64).min,dtype="float64")
        index_list=np.full((int(how_many_to_save),how_many_to_choose),Nan_number,dtype="int64")
        border=np.finfo(np.float64).min
        min_index=0
        index_arr=np.zeros((how_many_to_choose),dtype="int64")
        selected_X=np.empty((how_many_to_choose,arr_x.shape[2]),dtype="float64")
        check_list=make_check_list(arr_which_arr_to_choose_from)
        for i in range(thread_id,repeat,num_threads):
            is_calc=make_index_arr(i,check_list,len_x_arr,arr_which_arr_to_choose_from,index_arr)
            if not is_calc:
                continue
            for j,k in enumerate(index_arr):
                selected_X[j]=arr_x[arr_which_arr_to_choose_from[j],k]
            score=score_func(selected_X,y)
            if score>=border:
                score_list[min_index]=score
                index_list[min_index]=index_arr
                border,min_index=argmin_and_min(score_list)
            progress_proxy.update(1)
        score_list_thread[thread_id]=score_list
        index_list_thread[thread_id]=index_list
    sort_index=np.argsort(score_list_thread.ravel())[::-1][:how_many_to_save]
    return score_list_thread.ravel()[sort_index],index_list_thread.reshape(-1,how_many_to_choose)[sort_index]
