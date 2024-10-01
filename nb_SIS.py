#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from log_progress import loop_log
from numba_progress import ProgressBar
import datetime,logging,multiprocessing
from numba import njit,prange,set_num_threads,objmode
from utils import thread_check,calc_RPN,argmin_and_min,raise_and_log,dtype_shape_check,type_check

def SIS(
    x,
    y,
    model_score,
    units=None,
    how_many_to_save=50000,
    is_use_1=False,
    max_n_op=5,
    operators_to_use=["+","-","*","/"],
    num_threads=None,
    verbose=True,
    is_progress=False,
    log_interval=10,
    logger=None):

    
    
    Nan_number=-100

    #logger
    if logger is None:
        logger = logging.getLogger("SIS")
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
    dtype_shape_check(logger,x,"x",dtype_=np.float64,ndim=2)

    #y
    dtype_shape_check(logger,y,"y",ndim=1,dict_index_len={0:x.shape[1]})
        
    #units
    if units is None:
        units=np.zeros((x.shape[0],1),dtype="int64")
    else:
        dtype_shape_check(logger,units,"units",dtype_=np.int64,ndim=2,dict_index_len={0:x.shape[0]})
        
    if "sqrt" in operators_to_use:
        units*=2**max_n_op
    if "cbrt" in operators_to_use:
        units*=3**max_n_op
        
    dict_op_str_to_num={"+":-1,"-":-2,"*":-3,"/":-4,"*-1":-5,"**-1":-6,"**2":-7,
                        "sqrt":-8,"| |":-9,"**3":-10,"cbrt":-11,"**6":-12,"exp":-13,
                        "exp-":-14,"log":-15,"sin":-16,"cos":-17,"scd":-18}

    type_check(logger,operators_to_use,"operators_to_use",list)

    #use_binary_op
    use_binary_op=[]
    for i in list(dict_op_str_to_num.keys())[:4]:
        if i in operators_to_use:
            use_binary_op+=[dict_op_str_to_num[i]]

    #use_unary_op
    use_unary_op=[]
    for i in list(dict_op_str_to_num.keys())[4:]:
        if i in operators_to_use:
            use_unary_op+=[dict_op_str_to_num[i]]
    
    logger.info("SIS")
    logger.info(f"num_threads={num_threads}, how_many_to_save={how_many_to_save}, ")
    logger.info(f"max_n_op={max_n_op}, model_score={model_score.__name__}, ")
    logger.info(f"x.shape={x.shape}, is_use_1={is_use_1}")
    logger.info(f"use_binary_op={use_binary_op}, ")
    logger.info(f"use_unary_op={use_unary_op}")
    str_units=" , ".join([str(units[i]) for i in range(units.shape[0])])
    logger.info(f"units={str_units}")

    save_score_list=np.full((num_threads,how_many_to_save,2),np.finfo(np.float64).min,dtype="float64")
    save_eq_list=np.full((num_threads,how_many_to_save,2*max_n_op+1),Nan_number,dtype="int8")
    min_index_list=np.zeros(num_threads,dtype="int64")
    border_list=np.full((num_threads,2),np.finfo(np.float64).min,dtype="float64")
    
    used_eq_dict,used_unit_dict,used_shape_id_dict,used_info_dict=sub_loop_non_op(x,y,units,is_use_1,
                                                                                  model_score,save_score_list,save_eq_list,min_index_list,border_list)

    time0=datetime.datetime.now()
    for n_op in range(1,max_n_op+1):
        time1=datetime.datetime.now()
        logger.info(f"  n_op={n_op}")
        for n_op1 in range(n_op):
            n_op2=n_op-1-n_op1
            loop=loop_counter_binary(n_op1,n_op2,used_eq_dict)
            logger.info(f"    binary_op n_op1:n_op2 = {n_op1}:{n_op2},  loop:{loop}")
            time2=datetime.datetime.now()
            if is_progress:
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                with ProgressBar(total=loop,dynamic_ncols=False,bar_format=bar_format,leave=False) as progress:
                    sub_loop_binary_op(x,y,model_score,how_many_to_save,n_op,n_op1,use_binary_op,save_score_list,
                                       save_eq_list,min_index_list,border_list,used_eq_dict,used_unit_dict,used_shape_id_dict,used_info_dict,progress)
            else:
                header="      "
                with loop_log(logger,interval=log_interval,tot_loop=loop,header=header) as progress:
                    sub_loop_binary_op(x,y,model_score,how_many_to_save,n_op,n_op1,use_binary_op,save_score_list,
                                       save_eq_list,min_index_list,border_list,used_eq_dict,used_unit_dict,used_shape_id_dict,used_info_dict,progress)
            logger.info(f"      time : {datetime.datetime.now()-time2}")
        if len(use_unary_op)!=0:
            loop=loop_counter_unary(n_op,used_eq_dict)
            logger.info(f"    unary_op,  loop:{loop}")
            time2=datetime.datetime.now()
            if is_progress:
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                with ProgressBar(total=loop,dynamic_ncols=False,bar_format=bar_format,leave=False) as progress:
                    sub_loop_unary_op(x,y,model_score,how_many_to_save,n_op,use_unary_op,
                                      save_score_list,save_eq_list,min_index_list,border_list,used_eq_dict,used_unit_dict,used_shape_id_dict,used_info_dict,progress)
            else:
                header="      "
                with loop_log(logger,interval=log_interval,tot_loop=loop,header=header) as progress:
                    sub_loop_unary_op(x,y,model_score,how_many_to_save,n_op,use_unary_op,
                                      save_score_list,save_eq_list,min_index_list,border_list,used_eq_dict,used_unit_dict,used_shape_id_dict,used_info_dict,progress)
            logger.info(f"      time : {datetime.datetime.now()-time2}")
        logger.info(f"    END, time={datetime.datetime.now()-time1}")
    
    index=np.lexsort((save_score_list[:,:,1].ravel(),save_score_list[:,:,0].ravel()))[::-1][:how_many_to_save]
    return_score_list=save_score_list.reshape((-1,2))[index]
    return_eq_list=save_eq_list.reshape(-1,2*max_n_op+1)[index]
    logger.info(f"total time={datetime.datetime.now()-time0}")
    return return_score_list,return_eq_list

@njit(error_model="numpy")
def loop_counter_binary(n_op1,n_op2,used_eq_dict):
    loop=0
    for n_binary_op1 in range(n_op1+1):
        len_use_eq_arr1=used_eq_dict[n_op1][n_binary_op1].shape[0]
        for n_binary_op2 in range(n_op2+1):
            loop+=(len_use_eq_arr1*used_eq_dict[n_op2][n_binary_op2].shape[0])
    return loop

@njit(error_model="numpy")
def loop_counter_unary(n_op,used_eq_dict):
    loop=0
    for before_n_binary_op in range(n_op):
        loop+=used_eq_dict[n_op-1][before_n_binary_op].shape[0]
    return loop

@njit(error_model="numpy")
def make_change_x_id(mask,x_max):
    if np.sum(mask)==0:
        return 0
    TF=np.ones(x_max,dtype="bool")
    return_num=0
    for i in range(mask.shape[0]):
        return_num*=(x_max-i)
        return_num+=np.sum(TF[:mask[i]-1])
        TF[mask[i]-1]=False
    return return_num

@njit(error_model="numpy")
def eq_to_num(eq,x_max):
    Nan_number=-100
    len_eq=np.sum(eq!=Nan_number)
    num=eq[len_eq-1]+18 # -5~-18
    for i in range(len_eq-2,0,-1):
        num=num*(x_max+19)+(18+eq[i])
    num=num*(x_max+1)+eq[0]
    return num
    
@njit(error_model="numpy")
def load_preprocessed_results(n_binary_op,n_binary_op1):
    Nan_number=-100
    #back_eq -> make_change_x_id -> changed_back_eq_x_num
    #num_to_index         :  [changed_back_eq_x_num]->changed_back_eq_x_index
    #preprocessed_results :  [id1,id2,changed_back_eq_x_index,merge_op]->eq_id or Nan_number
    #eq -> make_change_x_id -> shuffled_eq_x_num
    #need_calc_change_x_dict  :  [n_binary_op,eq_id,shuffled_eq_x_num]->bool
    with objmode(preprocessed_results='int64[:,:,:,:]'):
        preprocessed_results=np.load(f'cache_folder/cache_{n_binary_op}.npz')[str(n_binary_op1)]
    with objmode(data='int64[:]'):
        data=np.load(f'cache_folder/num_to_index_{n_binary_op}.npz')[str(n_binary_op1)]
    num_to_index=np.full((np.max(data)+1),Nan_number,dtype="int64")
    for i,j in enumerate(np.sort(data)):
        num_to_index[j]=i
    #num_to_index={j:i for i,j in enumerate(data)}
    with objmode(need_calc_change_x='boolean[:,:]'):
        need_calc_change_x=np.load(f'cache_folder/need_calc_change_x_{n_binary_op}.npz')["arr_0"]
    return preprocessed_results,num_to_index,need_calc_change_x

@njit(error_model="numpy")
def load_max_id_need_calc():
    #max_id_need_calc  :  [n_binary_op]->max_eq_id
    with objmode(max_id_need_calc='int64[:]'):
        max_id_need_calc=np.load("cache_folder/max_id_need_calc.npy")
    return max_id_need_calc
        
@njit(error_model="numpy")
def make_eq_id(n_binary_op1,info):
    n_binary_op=info.shape[0]-1
    arg_sort=np.argsort(info)
    mask_1=np.ones(n_binary_op+1,dtype="bool")
    mask_2=info[n_binary_op1+1:]!=0
    if info[arg_sort[0]]==0:
        mask_1[0]=False
    for i in range(1,n_binary_op+1):
        if info[arg_sort[i-1]]==info[arg_sort[i]]:
            mask_1[i]=False
            if n_binary_op1<arg_sort[i-1]:
                mask_2[arg_sort[i]-n_binary_op1-1]=False
    retuen_arr=np.empty(n_binary_op+1,dtype="int8")
    retuen_arr[np.sort(arg_sort[mask_1])]=np.arange(1,np.sum(mask_1)+1)
    for i in np.arange(n_binary_op+1)[~mask_1]:
        retuen_arr[arg_sort[i]]=retuen_arr[arg_sort[i-1]]
    changed_back_eq_x_num=make_change_x_id(retuen_arr[n_binary_op1+1:][mask_2],n_binary_op+1)
    shuffled_eq_x_num=make_change_x_id(np.argsort(arg_sort[mask_1])+1,np.sum(mask_1))#ok
    return changed_back_eq_x_num,shuffled_eq_x_num
    
@njit(error_model="numpy")#,fastmath=True)
def sub_loop_non_op(x,y,units,is_use_1,model_score,save_score_list,save_eq_list,min_index_list,border_list):
    Nan_number=-100
    used_eq_dict=dict()
    used_unit_dict=dict()
    used_shape_id_dict=dict()
    used_info_dict=dict()
    
    used_eq_arr=np.empty((x.shape[0]+1,1),dtype="int8")
    used_unit_arr=np.empty((x.shape[0]+1,units.shape[1]),dtype="int64")
    used_shape_id_arr=np.empty((x.shape[0]+1),dtype="int64")
    used_info_arr=np.empty((x.shape[0]+1,1),dtype="int64")
    last_index=0

    border1,border2=border_list[0]
    
    if is_use_1:
        used_eq_arr[last_index]=0
        used_unit_arr[last_index]=0
        used_shape_id_arr[last_index]=1
        used_info_arr[last_index,0]=0
        last_index+=1
    for i in range(1,x.shape[0]+1):
        score1,score2=model_score(x[i-1],y)
        used_eq_arr[last_index]=i
        used_unit_arr[last_index]=units[i-1]
        used_shape_id_arr[last_index]=0
        used_info_arr[last_index,0]=i
        last_index+=1
        if np.logical_not(np.isnan(score1)):
            if score1>border1:
                save_score_list[0,min_index_list[0],0]=score1
                save_score_list[0,min_index_list[0],1]=score2
                save_eq_list[0,min_index_list[0],0]=i
                min_num1,min_num2,min_index=argmin_and_min(save_score_list[0])
                min_index_list[0]=min_index
                if border1>min_num1:
                    border1=min_num1
                    border2=min_num2
                elif border1==min_num1:
                    if border2>min_num2:
                        border2=min_num2
            elif score1==border1:
                if score2>border2:
                    save_score_list[0,min_index_list[0],0]=score1
                    save_score_list[0,min_index_list[0],1]=score2
                    save_eq_list[0,min_index_list[0],0]=i
                    min_num1,min_num2,min_index=argmin_and_min(save_score_list[0])
                    min_index_list[0]=min_index
                    if border1>min_num1:
                        border1=min_num1
                        border2=min_num2
                    elif border1==min_num1:
                        if border2>min_num2:
                            border2=min_num2
    border_list[0,0]=border1
    border_list[0,1]=border2
    used_eq_dict[0]={0:used_eq_arr[:last_index]}
    used_unit_dict[0]={0:used_unit_arr[:last_index]}
    used_shape_id_dict[0]={0:used_shape_id_arr[:last_index]}
    used_info_dict[0]={0:used_info_arr[:last_index]}
    
    return used_eq_dict,used_unit_dict,used_shape_id_dict,used_info_dict


@njit(parallel=True,error_model="numpy")#,fastmath=True)
def sub_loop_binary_op(x,y,model_score,how_many_to_save,n_op,n_op1,
                       use_binary_op,save_score_list,save_eq_list,min_index_list,border_list,used_eq_dict,used_unit_dict,used_shape_id_dict,used_info_dict,progress):
    Nan_number=-100

    max_id_need_calc=load_max_id_need_calc()
    num_threads=save_score_list.shape[0]
    max_n_op=(save_eq_list.shape[2]-1)//2
    n_op2=n_op-1-n_op1

    for n_binary_op1 in range(n_op1+1):
        use_eq_arr1=used_eq_dict[n_op1][n_binary_op1]
        use_unit_arr1=used_unit_dict[n_op1][n_binary_op1]
        use_shape_id_arr1=used_shape_id_dict[n_op1][n_binary_op1]
        use_info_arr1=used_info_dict[n_op1][n_binary_op1]
        if use_eq_arr1.shape[0]==0:
            continue
        for n_binary_op2 in range(n_op2+1):
            use_eq_arr2=used_eq_dict[n_op2][n_binary_op2]
            use_unit_arr2=used_unit_dict[n_op2][n_binary_op2]
            use_shape_id_arr2=used_shape_id_dict[n_op2][n_binary_op2]
            use_info_arr2=used_info_dict[n_op2][n_binary_op2]
            if use_eq_arr2.shape[0]==0:
                continue
            n_binary_op=n_binary_op1+n_binary_op2+1
            now_use_binary_op=use_binary_op.copy()
            if n_binary_op1<n_binary_op2:
                for i in [-1,-2,-3]:
                    if i in now_use_binary_op:
                        now_use_binary_op.remove(i)
            preprocessed_results,num_to_index,need_calc_change_x=load_preprocessed_results(n_binary_op,n_binary_op1)
            len_units=use_unit_arr1[0].shape[0]
            len_eq_arr1=use_eq_arr1.shape[0]
            len_eq_arr2=use_eq_arr2.shape[0]
            if max_n_op!=n_op:
                len_used_arr=len(now_use_binary_op)*((len_eq_arr1*len_eq_arr2)//num_threads+1)
            else:#max_n_op==n_op:
                len_used_arr=0
            used_eq_arr_thread=np.full((num_threads,len_used_arr,2*n_op+1),Nan_number,dtype="int8")
            used_unit_arr_thread=np.full((num_threads,len_used_arr,len_units),Nan_number,dtype="int64")
            used_shape_id_arr_thread=np.full((num_threads,len_used_arr),Nan_number,dtype="int64")
            used_info_arr_thread=np.full((num_threads,len_used_arr,n_binary_op+1),Nan_number,dtype="int64")
            last_index_thread=np.zeros(num_threads,dtype="int64")
            loop=len_eq_arr1*len_eq_arr2
            for thread_id in prange(num_threads):
                score_list=save_score_list[thread_id].copy()
                eq_list=save_eq_list[thread_id].copy()
                min_index=min_index_list[thread_id]
                border1,border2=border_list[thread_id]
                last_index=0
                equation=np.empty((2*n_op+1),dtype="int8")
                info=np.empty((n_binary_op+1),dtype="int64")
                for i in range(thread_id,loop,num_threads):
                    id1=i%len_eq_arr1
                    i//=len_eq_arr1
                    id2=i%len_eq_arr2
                    shape_id1=use_shape_id_arr1[id1]
                    shape_id2=use_shape_id_arr2[id2]
                    info[:n_binary_op1+1]=use_info_arr1[id1]
                    info[n_binary_op1+1:]=use_info_arr2[id2]
                    changed_back_eq_x_num,shuffled_eq_x_num=make_eq_id(n_binary_op1,info)
                    eq_id_binary_op_arr=preprocessed_results[shape_id1,shape_id2,num_to_index[changed_back_eq_x_num]]
                    for merge_op in now_use_binary_op:
                        eq_id=eq_id_binary_op_arr[merge_op+4]
                        if eq_id!=Nan_number:
                            if need_calc_change_x[eq_id,shuffled_eq_x_num]:
                                unit1=use_unit_arr1[id1]
                                unit2=use_unit_arr2[id2]
                                if merge_op in [-1,-2]:
                                    if np.any(unit1!=unit2):
                                        continue
                                eq1=use_eq_arr1[id1]
                                eq2=use_eq_arr2[id2]
                                len_eq1=np.sum(eq1!=Nan_number)
                                len_eq2=np.sum(eq2!=Nan_number)
                                equation[:len_eq1]=eq1[:len_eq1]
                                equation[len_eq1:len_eq1+len_eq2]=eq2[:len_eq2]
                                equation[len_eq1+len_eq2]=merge_op
                                equation[len_eq1+len_eq2+1:]=Nan_number
                                ans_num=calc_RPN(x,equation)
                                if np.logical_not(np.isnan(ans_num[0])):
                                    if max_n_op!=n_op:
                                        match merge_op:
                                            case -1:#+
                                                used_unit_arr_thread[thread_id,last_index]=unit1
                                            case -2:#-
                                                used_unit_arr_thread[thread_id,last_index]=unit1
                                            case -3:#*
                                                used_unit_arr_thread[thread_id,last_index]=unit1+unit2
                                            case -4:#/
                                                used_unit_arr_thread[thread_id,last_index]=unit1-unit2
                                        used_eq_arr_thread[thread_id,last_index]=equation
                                        used_shape_id_arr_thread[thread_id,last_index]=eq_id
                                        used_info_arr_thread[thread_id,last_index]=info
                                        last_index+=1
                                    if max_id_need_calc[n_binary_op]>eq_id:
                                        score1,score2=model_score(ans_num,y)
                                        if np.logical_not(np.isnan(score1)):
                                            if score1>border1:
                                                score_list[min_index,0]=score1
                                                score_list[min_index,1]=score2
                                                eq_list[min_index,:2*n_op+1]=equation
                                                min_num1,min_num2,min_index=argmin_and_min(score_list)
                                                if border1>min_num1:
                                                    border1=min_num1
                                                    border2=min_num2
                                                elif border1==min_num1:
                                                    if border2>min_num2:
                                                        border2=min_num2
                                            elif score1==border1:
                                                if score2>border2:
                                                    score_list[min_index,0]=score1
                                                    score_list[min_index,1]=score2
                                                    eq_list[min_index,:2*n_op+1]=equation
                                                    min_num1,min_num2,min_index=argmin_and_min(score_list)
                                                    if border1>min_num1:
                                                        border1=min_num1
                                                        border2=min_num2
                                                    elif border1==min_num1:
                                                        if border2>min_num2:
                                                            border2=min_num2
                    progress.update(1)
                save_score_list[thread_id]=score_list
                save_eq_list[thread_id]=eq_list
                min_index_list[thread_id]=min_index
                border_list[thread_id,0]=border1
                border_list[thread_id,1]=border2
                if max_n_op!=n_op:
                    last_index_thread[thread_id]=last_index
                    
            if num_threads!=1:
                border1=np.partition(save_score_list[:,:,0].ravel(),-how_many_to_save)[-how_many_to_save]
                n_save_same_score1=how_many_to_save-np.sum(save_score_list[:,:,0]>border1)
                border2=np.sort(save_score_list[:,:,1].ravel()[save_score_list[:,:,0].ravel()==border1])[::-1][n_save_same_score1-1]
                border_list[:,0]=border1
                border_list[:,1]=border2
                
            if max_n_op!=n_op:
                sum_last_index=np.sum(last_index_thread)
                used_eq_arr=np.full((sum_last_index,2*n_op+1),Nan_number,dtype="int8")
                used_unit_arr=np.full((sum_last_index,len_units),Nan_number,dtype="int64")
                used_shape_id_arr=np.full((sum_last_index),Nan_number,dtype="int64")
                used_info_arr=np.full((sum_last_index,n_binary_op+1),Nan_number,dtype="int64")
                for thread_id in prange(num_threads):
                    before_index=np.sum(last_index_thread[:thread_id])
                    used_eq_arr[before_index:before_index+last_index_thread[thread_id]]=used_eq_arr_thread[thread_id,:last_index_thread[thread_id]]
                    used_unit_arr[before_index:before_index+last_index_thread[thread_id]]=used_unit_arr_thread[thread_id,:last_index_thread[thread_id]]
                    used_shape_id_arr[before_index:before_index+last_index_thread[thread_id]]=used_shape_id_arr_thread[thread_id,:last_index_thread[thread_id]]
                    used_info_arr[before_index:before_index+last_index_thread[thread_id]]=used_info_arr_thread[thread_id,:last_index_thread[thread_id]]
                if n_op in used_eq_dict:
                    if n_binary_op in used_eq_dict[n_op]:
                        used_eq_dict[n_op][n_binary_op]=np.concatenate((used_eq_dict[n_op][n_binary_op],used_eq_arr))
                        used_unit_dict[n_op][n_binary_op]=np.concatenate((used_unit_dict[n_op][n_binary_op],used_unit_arr))
                        used_shape_id_dict[n_op][n_binary_op]=np.concatenate((used_shape_id_dict[n_op][n_binary_op],used_shape_id_arr))
                        used_info_dict[n_op][n_binary_op]=np.concatenate((used_info_dict[n_op][n_binary_op],used_info_arr))
                    else:
                        used_eq_dict[n_op][n_binary_op]=used_eq_arr
                        used_unit_dict[n_op][n_binary_op]=used_unit_arr
                        used_shape_id_dict[n_op][n_binary_op]=used_shape_id_arr
                        used_info_dict[n_op][n_binary_op]=used_info_arr
                else:
                    used_eq_dict[n_op]={n_binary_op:used_eq_arr}
                    used_unit_dict[n_op]={n_binary_op:used_unit_arr}
                    used_shape_id_dict[n_op]={n_binary_op:used_shape_id_arr}
                    used_info_dict[n_op]={n_binary_op:used_info_arr}

@njit(error_model="numpy")
def check_eq(eq,ban_ops):#use exp,exp-,log,sin,cos
    for ban_op in ban_ops:
        if ban_op in eq:
            return False
    return True
    
@njit(parallel=True,error_model="numpy")#,fastmath=True)
def sub_loop_unary_op(x,y,model_score,how_many_to_save,n_op,use_unary_op,
                      save_score_list,save_eq_list,min_index_list,border_list,used_eq_dict,used_unit_dict,used_shape_id_dict,used_info_dict,progress):
    Nan_number=-100
    num_threads=save_score_list.shape[0]
    max_n_op=(save_eq_list.shape[2]-1)//2
    x_max=x.shape[0]
    for before_n_binary_op in range(n_op):

        use_eq_arr=used_eq_dict[n_op-1][before_n_binary_op]
        use_unit_arr=used_unit_dict[n_op-1][before_n_binary_op]
        
        len_units=use_unit_arr[0].shape[0]
        if max_n_op!=n_op:
            len_used_arr=len(use_unary_op)*(use_eq_arr.shape[0]//num_threads+1)
        else:#max_n_op==n_op:
            len_used_arr=0
        used_eq_arr_thread=np.full((num_threads,len_used_arr,2*n_op+1),Nan_number,dtype="int8")
        used_unit_arr_thread=np.full((num_threads,len_used_arr,len_units),Nan_number,dtype="int64")
        used_shape_id_arr_thread=np.full((num_threads,len_used_arr),Nan_number,dtype="int64")
        used_info_arr_thread=np.full((num_threads,len_used_arr,1),Nan_number,dtype="int64")
        last_index_thread=np.zeros(num_threads,dtype="int64")
        loop=use_eq_arr.shape[0]
        for thread_id in prange(num_threads):
            score_list=save_score_list[thread_id].copy()
            eq_list=save_eq_list[thread_id].copy()
            min_index=min_index_list[thread_id]
            border1,border2=border_list[thread_id,0],border_list[thread_id,1]
            last_index=0
            equation=np.empty((2*n_op+1),dtype="int8")
            for i in range(thread_id,loop,num_threads):
                base_eq=use_eq_arr[i]
                len_base_eq=np.sum(base_eq!=Nan_number)
                base_unit=use_unit_arr[i]
                equation[:len_base_eq]=base_eq[:len_base_eq]
                equation[len_base_eq+1:]=Nan_number
                for op in use_unary_op:
                    checked=False
                    unit=base_unit
                    match op:
                        case -5:#*-1
                            #if equation[len_base_eq-1]!=-5:#*-1
                                #checked=True
                                #unit=base_unit#units更新なし
                            None  #存在意義が分からない
                        case -6:#^-1
                            #if equation[len_base_eq-1]!=-6:#^-1
                                #checked=True
                                #unit=-1*base_unit#units更新
                            None  #存在意義が分からない
                        case -7:#^2
                            if equation[len_base_eq-1]!=-8:#sqrt
                                checked=True
                                unit=2*base_unit#units更新
                        case -8:#sqrt
                            if not equation[len_base_eq-1] in [-7,-12]:#^2,^6
                                checked=True
                                unit=base_unit//2#units更新
                        case -9:#| |
                            checked=True
                            unit=base_unit#units更新なし
                        case -10:#^3
                            if equation[len_base_eq-1]!=-11:#cbrt
                                checked=True
                                unit=3*base_unit#units更新
                        case -11:#cbrt
                            if not equation[len_base_eq-1] in [-10,-12]:#^3,^6
                                checked=True
                                unit=base_unit//3#units更新
                        case -12:#^6
                            if not equation[len_base_eq-1] in [-8,-11]:#sqrt,cbrt
                                checked=True
                                unit=6*base_unit#units更新
                        case -13:#exp
                            if check_eq(base_eq,[-13,-14,-15,-16,-17]):#exp,exp-,log,sin,cos
                                if np.all(base_unit==0):# 無次元に制限
                                    checked=True
                                    #unit=base_unit
                        case -14:#exp-
                            if check_eq(base_eq,[-13,-14,-15,-16,-17]):#exp,exp-,log,sin,cos
                                if np.all(base_unit==0):# 無次元に制限
                                    checked=True
                                    #unit=base_unit
                        case -15:#log
                            if check_eq(base_eq,[-13,-14,-15,-16,-17]):#exp,exp-,log,sin,cos
                                if np.all(base_unit==0):# 無次元に制限
                                    checked=True
                                    #unit=base_unit
                        case -16:#sin
                            if check_eq(base_eq,[-13,-14,-15,-16,-17]):#exp,exp-,log,sin,cos
                                if np.all(base_unit==0):# 無次元に制限
                                    checked=True
                                    #unit=base_unit
                        case -17:#cos
                            if check_eq(base_eq,[-13,-14,-15,-16,-17]):#exp,exp-,log,sin,cos
                                if np.all(base_unit==0):# 無次元に制限
                                    checked=True
                                    #unit=base_unit
                        case -18:#scd  #わっかんね
                            if np.all(base_unit==0):# 無次元に制限????
                                checked=True
                                #unit=base_unit#units更新????
                    if checked:
                        equation[len_base_eq]=op
                        ans_num=calc_RPN(x,equation)
                        if np.logical_not(np.isnan(ans_num[0])):
                            score1,score2=model_score(ans_num,y)
                            if np.logical_not(np.isnan(score1)):
                                if max_n_op!=n_op:
                                    used_unit_arr_thread[thread_id,last_index]=unit
                                    used_eq_arr_thread[thread_id,last_index]=equation
                                    used_shape_id_arr_thread[thread_id,last_index]=0
                                    used_info_arr_thread[thread_id,last_index]=eq_to_num(equation,x_max)
                                    last_index+=1
                                if score1>border1:
                                    score_list[min_index,0]=score1
                                    score_list[min_index,1]=score2
                                    eq_list[min_index,:2*n_op+1]=equation
                                    min_num1,min_num2,min_index=argmin_and_min(score_list)
                                    if border1>min_num1:
                                        border1=min_num1
                                        border2=min_num2
                                    elif border1==min_num1:
                                        if border2>min_num2:
                                            border2=min_num2
                                elif score1==border1:
                                    if score2>border2:
                                        score_list[min_index,0]=score1
                                        score_list[min_index,1]=score2
                                        eq_list[min_index,:2*n_op+1]=equation
                                        min_num1,min_num2,min_index=argmin_and_min(score_list)
                                        if border1>min_num1:
                                            border1=min_num1
                                            border2=min_num2
                                        elif border1==min_num1:
                                            if border2>min_num2:
                                                border2=min_num2
                progress.update(1)
            save_score_list[thread_id]=score_list
            save_eq_list[thread_id]=eq_list
            min_index_list[thread_id]=min_index
            border_list[thread_id,0]=border1
            border_list[thread_id,1]=border2
            if max_n_op!=n_op:
                last_index_thread[thread_id]=last_index
        if num_threads!=1:
            border1=np.partition(save_score_list[:,:,0].ravel(),-how_many_to_save)[-how_many_to_save]
            n_save_same_score1=how_many_to_save-np.sum(save_score_list[:,:,0]>border1)
            border2=np.sort(save_score_list[:,:,1].ravel()[save_score_list[:,:,0].ravel()==border1])[::-1][n_save_same_score1-1]
            border_list[:,0]=border1
            border_list[:,1]=border2
        
        if max_n_op!=n_op:
            sum_last_index=np.sum(last_index_thread)
            used_eq_arr=np.full((sum_last_index,2*n_op+1),Nan_number,dtype="int8")
            used_unit_arr=np.full((sum_last_index,len_units),Nan_number,dtype="int64")
            used_shape_id_arr=np.full((sum_last_index),Nan_number,dtype="int64")
            used_info_arr=np.full((sum_last_index,1),Nan_number,dtype="int64")
            for thread_id in prange(num_threads):
                before_index=np.sum(last_index_thread[:thread_id])
                used_eq_arr[before_index:before_index+last_index_thread[thread_id]]=used_eq_arr_thread[thread_id,:last_index_thread[thread_id]]
                used_unit_arr[before_index:before_index+last_index_thread[thread_id]]=used_unit_arr_thread[thread_id,:last_index_thread[thread_id]]
                used_shape_id_arr[before_index:before_index+last_index_thread[thread_id]]=used_shape_id_arr_thread[thread_id,:last_index_thread[thread_id]]
                used_info_arr[before_index:before_index+last_index_thread[thread_id]]=used_info_arr_thread[thread_id,:last_index_thread[thread_id]]
            if 0 in used_eq_dict:
                if 0 in used_eq_dict[n_op]:
                        used_eq_dict[n_op][0]=np.concatenate((used_eq_dict[n_op][0],used_eq_arr))
                        used_unit_dict[n_op][0]=np.concatenate((used_unit_dict[n_op][0],used_unit_arr))
                        used_shape_id_dict[n_op][0]=np.concatenate((used_shape_id_dict[n_op][0],used_shape_id_arr))
                        used_info_dict[n_op][0]=np.concatenate((used_info_dict[n_op][0],used_info_arr))
                else:
                    used_eq_dict[n_op][0]=used_eq_arr
                    used_unit_dict[n_op][0]=used_unit_arr
                    used_shape_id_dict[n_op][0]=used_shape_id_arr
                    used_info_dict[n_op][0]=used_info_arr
            else:
                used_eq_dict[n_op]={0:used_eq_arr}
                used_unit_dict[n_op]={0:used_unit_arr}
                used_shape_id_dict[n_op]={0:used_shape_id_arr}
                used_info_dict[n_op]={0:used_info_arr}