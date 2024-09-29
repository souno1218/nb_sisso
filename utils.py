import numpy as np
from numba import njit,prange,set_num_threads,objmode

def raise_and_log(logger,error):
    logger.error(error)#exception()
    raise error

def type_check(logger,var,var_name,type_):
    if not isinstance(var,type_):
        raise_and_log(logger,TypeError(f"Expected variable '{var_name}' to be of type {type_}, but got {type(var)}."))

def dtype_shape_check(logger,var,var_name,dtype_=None,ndim=None,dict_index_len=None):
    type_check(logger,var,var_name,np.ndarray)
    if not dtype_ is None:
        if var.dtype!=dtype_:
            raise_and_log(logger,ValueError(f"Expected dtype for variable '{var_name}' to be {dtype_}, but got {var.dtype}."))
    if not ndim is None:
        if var.ndim!=ndim:
            raise_and_log(logger,ValueError(f"Expected ndim for variable '{var_name}' to be {ndim}, but got {var.ndim}."))
    if not dict_index_len is None:
        for k, v in dict_index_len.items():
            if var.shape[k]!=v:
                raise_and_log(logger,ValueError(f"Expected size for dimension {k} of variable '{var_name}' to be {v}, but got {var.shape[k]}."))

@njit(parallel=True,cache=True)
def thread_check(num_threads):
    checker=np.zeros(num_threads,dtype="bool")
    for thread_id in prange(num_threads):
        checker[thread_id]=True
    return np.all(checker)

@njit(error_model="numpy",cache=True)
def argmin_and_min(arr):
    min_num,min_index=arr[0],0
    for i in range(1,arr.shape[0]):
        if min_num>arr[i]:
            min_num,min_index=arr[i],i
    return min_num,min_index

def decryption(equation,columns=None):
    if columns is None:
        columns=[chr(i+97) for i in range(np.max(equation))]
    stack=[]
    for i in equation:
        match i:
            case 0:#1
                stack.append("1")
            case t if t>0:#number
                stack.append(columns[i-1])
            case -1:#+
                b,a=stack.pop(),stack.pop()
                stack.append(f"({a}+{b})")
            case -2:#-
                b,a=stack.pop(),stack.pop()
                stack.append(f"({a}-{b})")
            case -3:#*
                b,a=stack.pop(),stack.pop()
                stack.append(f"({a}*{b})")
            case -4:#/
                b,a=stack.pop(),stack.pop()
                stack.append(f"({a}/{b})")
            case -5:#*-1
                stack[len(stack)-1]=f"({stack[len(stack)-1]}*-1)"
            case -6:#^-1
                stack[len(stack)-1]=f"({stack[len(stack)-1]}**-1)"
            case -7:#^2
                stack[len(stack)-1]=f"({stack[len(stack)-1]}**2)"
            case -8:#sqrt
                stack[len(stack)-1]=f"np.sqrt({stack[len(stack)-1]})"
            case -9:#| |
                stack[len(stack)-1]=f"np.abs({stack[len(stack)-1]})"
            case -10:#^3
                stack[len(stack)-1]=f"({stack[len(stack)-1]}**3)"
            case -11:#cbrt
                stack[len(stack)-1]=f"np.cbrt({stack[len(stack)-1]})"
            case -12:#^6
                stack[len(stack)-1]=f"({stack[len(stack)-1]}**6)"
            case -13:#exp
                stack[len(stack)-1]=f"np.exp({stack[len(stack)-1]})"
            case -14:#exp-
                stack[len(stack)-1]=f"np.exp(-1*{stack[len(stack)-1]})"
            case -15:#log
                stack[len(stack)-1]=f"np.log({stack[len(stack)-1]})"
            case -16:#sin
                stack[len(stack)-1]=f"np.sin({stack[len(stack)-1]})"
            case -17:#cos
                stack[len(stack)-1]=f"np.cos({stack[len(stack)-1]})"
            case -18:#scd
                stack[len(stack)-1]=f"np.scd({stack[len(stack)-1]})"
    return stack[0]

@njit(error_model="numpy",cache=True)#parallel=True,
def eq_list_to_num(x,eq_list):
    return_arr=np.empty((eq_list.shape[0],x.shape[1]))
    for i in range(eq_list.shape[0]):
        return_arr[i]=calc_RPN(x,eq_list[i])
    return return_arr

@njit(error_model="numpy",cache=True)#,fastmath=True)
def calc_RPN(x,equation):
    #Nan_number=-100
    stack=np.full((np.sum(equation>=0),x.shape[1]),np.nan,dtype="float64")
    last_stack_index=-1
    for i in range(equation.shape[0]):
        last_op=equation[i]
        if last_op==0:
            last_stack_index+=1
            stack[last_stack_index]=1
        if last_op>0:
            last_stack_index+=1
            stack[last_stack_index]=x[last_op-1]
        else:
            match last_op:
                case -1:#+
                    last_stack_index-=1
                    stack[last_stack_index]+=stack[last_stack_index+1]
                case -2:#-
                    last_stack_index-=1
                    stack[last_stack_index]-=stack[last_stack_index+1]
                case -3:#*
                    last_stack_index-=1
                    stack[last_stack_index]*=stack[last_stack_index+1]
                case -4:#/
                    last_stack_index-=1
                    stack[last_stack_index]/=stack[last_stack_index+1]
                case -5:#*-1
                    stack[last_stack_index]*=-1
                case -6:#^-1
                    stack[last_stack_index]**=-1
                case -7:#^2
                    stack[last_stack_index]**=2
                case -8:#sqrt
                    # only x>=0
                    if np.any(stack[last_stack_index]<0):
                        stack[0,0]=np.nan
                        return stack[0]
                    stack[last_stack_index]**=0.5
                case -9:#| |
                    # not all(x>=0),all(x<0)
                    if np.all((stack[last_stack_index,0]*stack[last_stack_index,1:])>=0):
                        stack[0,0]=np.nan
                        return stack[0]
                    stack[last_stack_index]=np.abs(stack[last_stack_index])
                case -10:#^3
                    stack[last_stack_index]**=3
                case -11:#cbrt
                    # only x>=0
                    if np.any(stack[last_stack_index]<0):
                        stack[0,0]=np.nan
                        return stack[0]
                    stack[last_stack_index]=np.cbrt(stack[last_stack_index])
                case -12:#^6
                    stack[last_stack_index]**=6
                case -13:#exp
                    stack[last_stack_index]=np.exp(stack[last_stack_index])
                case -14:#exp-
                    stack[last_stack_index]=np.exp(-stack[last_stack_index])
                case -15:#log
                    # only x>=0
                    if np.any(stack[last_stack_index]<0):
                        stack[0,0]=np.nan
                        return stack[0]
                    stack[last_stack_index]=np.log(stack[last_stack_index])
                case -16:#sin
                    stack[last_stack_index]=np.sin(stack[last_stack_index])
                case -17:#cos
                    stack[last_stack_index]=np.cos(stack[last_stack_index])
                case -18:#scd
                    stack[last_stack_index]=np.sin(stack[last_stack_index])# よくわからないため未実装
                case -100:#END
                    break#None
    if np.any(np.isinf(stack[last_stack_index])):#演算子の意味があるか
        stack[0,0]=np.nan
    elif np.any(np.isnan(stack[last_stack_index])):
        stack[0,0]=np.nan
    elif is_zero(stack[last_stack_index]):
        stack[0,0]=np.nan
    #if is_const(stack[0]):
        #stack[0,0]=np.nan
    return stack[0]

@njit(error_model="numpy",cache=True)#,fastmath=True)
def is_zero(num):
    rtol=1e-010
    return np.all(np.abs(num)<=rtol)

@njit(error_model="numpy",cache=True)#,fastmath=True)
def is_const(num):
    rtol=1e-010
    return (np.max(num)-np.min(num))/np.abs(np.mean(num))<=rtol