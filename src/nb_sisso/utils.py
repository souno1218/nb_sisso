#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, datetime, threading, signal
import numpy as np
from scipy import integrate
from numba import njit, prange, objmode
from numba.types import int8, int64


def decryption(equation, columns=None):
    if columns is None:
        columns = [chr(i + 97) for i in range(np.max(equation))]
    stack = []
    if equation[0] == -100:
        return "0"
    for i in equation:
        match i:
            case 0:  # 1
                stack.append("1")
            case t if t > 0:  # number
                stack.append(columns[i - 1])
            case -1:  # +
                b, a = stack.pop(), stack.pop()
                stack.append(f"({a}+{b})")
            case -2:  # -
                b, a = stack.pop(), stack.pop()
                stack.append(f"({a}-{b})")
            case -3:  # *
                b, a = stack.pop(), stack.pop()
                stack.append(f"({a}*{b})")
            case -4:  # a / b  (a > b)
                b, a = stack.pop(), stack.pop()
                stack.append(f"({a}/{b})")
            case -5:  # *-1
                stack[len(stack) - 1] = f"({stack[len(stack)-1]}*-1)"
            case -6:  # ^-1
                stack[len(stack) - 1] = f"({stack[len(stack)-1]}**-1)"
            case -7:  # ^2
                stack[len(stack) - 1] = f"({stack[len(stack)-1]}**2)"
            case -8:  # sqrt
                stack[len(stack) - 1] = f"np.sqrt({stack[len(stack)-1]})"
            case -9:  # | |
                stack[len(stack) - 1] = f"np.abs({stack[len(stack)-1]})"
            case -10:  # ^3
                stack[len(stack) - 1] = f"({stack[len(stack)-1]}**3)"
            case -11:  # cbrt
                stack[len(stack) - 1] = f"np.cbrt({stack[len(stack)-1]})"
            case -12:  # ^6
                stack[len(stack) - 1] = f"({stack[len(stack)-1]}**6)"
            case -13:  # exp
                stack[len(stack) - 1] = f"np.exp({stack[len(stack)-1]})"
            case -14:  # exp-
                stack[len(stack) - 1] = f"np.exp(-1*{stack[len(stack)-1]})"
            case -15:  # log
                stack[len(stack) - 1] = f"np.log({stack[len(stack)-1]})"
            case -16:  # sin
                stack[len(stack) - 1] = f"np.sin({stack[len(stack)-1]})"
            case -17:  # cos
                stack[len(stack) - 1] = f"np.cos({stack[len(stack)-1]})"
            case -18:  # scd
                stack[len(stack) - 1] = f"np.scd({stack[len(stack)-1]})"
    return stack[0]


@njit(error_model="numpy", cache=True)  # parallel=True,
def eq_list_to_num(x, eq_list):
    return_arr = np.empty((eq_list.shape[0], x.shape[1]))
    for i in range(eq_list.shape[0]):
        return_arr[i] = calc_RPN(x, eq_list[i])
    return return_arr


## ------------------ Not use ------------------


def raise_and_log(logger, error):
    logger.error(error)  # exception()
    raise error


def type_check(logger, var, var_name, type_):
    if not isinstance(var, type_):
        raise_and_log(
            logger,
            TypeError(f"Expected variable '{var_name}' to be of type {type_}, but got {type(var)}."),
        )


def dtype_shape_check(logger, var, var_name, dtype_=None, ndim=None, dict_index_len=None):
    type_check(logger, var, var_name, np.ndarray)
    if not dtype_ is None:
        if var.dtype != dtype_:
            raise_and_log(
                logger,
                ValueError(f"Expected dtype for variable '{var_name}' to be {dtype_}, but got {var.dtype}."),
            )
    if not ndim is None:
        if var.ndim != ndim:
            raise_and_log(
                logger,
                ValueError(f"Expected ndim for variable '{var_name}' to be {ndim}, but got {var.ndim}."),
            )
    if not dict_index_len is None:
        for k, v in dict_index_len.items():
            if var.shape[k] != v:
                raise_and_log(
                    logger,
                    ValueError(
                        f"Expected size for dimension {k} of variable '{var_name}' to be {v}, but got {var.shape[k]}."
                    ),
                )


@njit(parallel=True)
def thread_check(num_threads):
    checker = np.zeros(num_threads, dtype="bool")
    for thread_id in prange(num_threads):
        checker[thread_id] = True
    return np.all(checker)


@njit(error_model="numpy")
def argmin_and_min(arr):
    min_num1, min_num2, min_index = arr[0, 0], arr[0, 1], 0
    for i in range(1, arr.shape[0]):
        if min_num1 > arr[i, 0]:
            min_num1, min_num2, min_index = arr[i, 0], arr[i, 1], i
        elif min_num1 == arr[i, 0]:
            if min_num2 > arr[i, 1]:
                min_num1, min_num2, min_index = arr[i, 0], arr[i, 1], i
    return min_num1, min_num2, min_index


@njit(error_model="numpy")
def calc_RPN(x, equation, fmax_max=1e15, fmax_min=1e-15):
    n_samples = x.shape[1]
    stack = np.empty((count_True(equation, 6, 0), n_samples), dtype="float64")  # 6 -> lambda x: x >= border
    next_index = 0
    for i in range(equation.shape[0]):
        last_op = equation[i]
        if last_op == 0:
            stack[next_index] = 1
            next_index += 1
        if last_op > 0:
            set_x(stack[next_index], x[last_op - 1])
            next_index += 1
        else:
            match last_op:
                case -1:  # +
                    calc_arr_binary_op(last_op, stack[next_index - 2], stack[next_index - 1])
                    next_index -= 1
                case -2:  # -
                    calc_arr_binary_op(last_op, stack[next_index - 2], stack[next_index - 1])
                    next_index -= 1
                case -3:  # *
                    calc_arr_binary_op(last_op, stack[next_index - 2], stack[next_index - 1])
                    next_index -= 1
                case -4:  # a / b  (a > b)
                    calc_arr_binary_op(last_op, stack[next_index - 2], stack[next_index - 1])
                    next_index -= 1
                case -5:  # *-1
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -6:  # ^-1
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -7:  # ^2
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -8:  # sqrt
                    # only x>=0
                    for j in range(n_samples):
                        if stack[next_index - 1, j] < 0:
                            stack[0, 0] = np.nan
                            return stack[0]
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -9:  # | |
                    # not all(x>=0),all(x<0)
                    if (0 >= np.max(stack[next_index - 1])) or (np.min(stack[next_index - 1]) >= 0):
                        stack[0, 0] = np.nan
                        return stack[0]
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -10:  # ^3
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -11:  # cbrt
                    # only x>=0
                    for j in range(n_samples):
                        if stack[next_index - 1, j] < 0:
                            stack[0, 0] = np.nan
                            return stack[0]
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -12:  # ^6
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -13:  # exp
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -14:  # exp-
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -15:  # log
                    # only x>=0
                    for j in range(n_samples):
                        if stack[next_index - 1, j] < 0:
                            stack[0, 0] = np.nan
                            return stack[0]
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -16:  # sin
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -17:  # cos
                    calc_arr_unary_op(last_op, stack[next_index - 1])
                case -18:  # scd
                    # よくわからないため未実装
                    None
                case -100:  # END
                    break  # None
    max_, min_ = np.max(stack[0]), np.min(stack[0])
    if np.abs(max_) > fmax_max:
        stack[0, 0] = np.nan
    elif np.abs(min_) > fmax_max:
        stack[0, 0] = np.nan
    elif is_nan(stack[0]):
        stack[0, 0] = np.nan
    elif is_all_zero(stack[0], atol=fmax_min):
        stack[0, 0] = np.nan
    elif is_all_one(stack[0], atol=fmax_min):
        stack[0, 0] = np.nan
    # if is_const(stack[0]):
    # stack[0,0]=np.nan
    return stack[0]


@njit(error_model="numpy")  # ,fastmath=True)
def set_x(arr_in, arr_out):
    for i in range(arr_in.shape[0]):
        arr_in[i] = arr_out[i]


@njit(error_model="numpy")  # ,fastmath=True)
def calc_arr_binary_op(op, arr1, arr2):
    match op:
        case -1:  # +
            for i in range(arr1.shape[0]):
                arr1[i] += arr2[i]
        case -2:  # -
            for i in range(arr1.shape[0]):
                arr1[i] -= arr2[i]
        case -3:  # *
            for i in range(arr1.shape[0]):
                arr1[i] *= arr2[i]
        case -4:  # a / b  (a > b)
            for i in range(arr1.shape[0]):
                arr1[i] /= arr2[i]


@njit(error_model="numpy")  # ,fastmath=True)
def calc_arr_unary_op(op, arr):
    match op:
        case -5:  # *-1
            for i in range(arr.shape[0]):
                arr[i] = -arr[i]
        case -6:  # ^-1
            for i in range(arr.shape[0]):
                arr[i] = arr[i] ** -1
        case -7:  # ^2
            for i in range(arr.shape[0]):
                arr[i] = arr[i] ** 2
        case -8:  # sqrt
            for i in range(arr.shape[0]):
                arr[i] = arr[i] ** 0.5
        case -9:  # | |
            for i in range(arr.shape[0]):
                arr[i] = np.abs(arr[i])
        case -10:  # ^3
            for i in range(arr.shape[0]):
                arr[i] = arr[i] ** 3
        case -11:  # cbrt
            for i in range(arr.shape[0]):
                arr[i] = np.cbrt(arr[i])
        case -12:  # ^6
            for i in range(arr.shape[0]):
                arr[i] = arr[i] ** 6
        case -13:  # exp
            for i in range(arr.shape[0]):
                arr[i] = np.exp(arr[i])
        case -14:  # exp-
            for i in range(arr.shape[0]):
                arr[i] = np.exp(-arr[i])
        case -15:  # log
            for i in range(arr.shape[0]):
                arr[i] = np.log(arr[i])
        case -16:  # sin
            for i in range(arr.shape[0]):
                arr[i] = np.sin(arr[i])
        case -17:  # cos
            for i in range(arr.shape[0]):
                arr[i] = np.cos(arr[i])
        # case -18:  # scd


@njit(error_model="numpy")
def isclose(a, b, rtol=1e-06, atol=0):
    return np.abs(a - b) <= atol + rtol * np.abs(b)


@njit(error_model="numpy")
def is_nan(arr):
    for i in range(arr.shape[0]):
        if np.isnan(arr[i]):
            return True
    return False


@njit(error_model="numpy")
def is_all_zero(arr, atol=1e-06):
    for i in range(arr.shape[0]):
        if not isclose(0, arr[i], rtol=0, atol=atol):
            return False
    return True


@njit(error_model="numpy")
def is_all_one(arr, atol=1e-06):
    for i in range(arr.shape[0]):
        if not isclose(1, arr[i], rtol=0, atol=atol):
            return False
    return True


@njit(error_model="numpy")  # ,fastmath=True)
def is_const(num, rtol=1e-010):
    return (np.max(num) - np.min(num)) <= rtol * np.abs(np.mean(num))


@njit(error_model="numpy")
def isclose_arr(arr1, arr2, rtol=1e-06, atol=0):
    # dim is 1 only, need arr1.shape==arr2.shape
    # if np.all(isclose(arr1, arr2, rtol=rtol, atol=atol)):
    #    return True
    for i in range(arr1.shape[0]):
        if not isclose(arr1[i], arr2[i], rtol=rtol, atol=atol):
            return False
    return True


@njit(error_model="numpy")  # ,fastmath=True)
def count_True(arr, mode, border):
    count = 0
    match mode:
        case 0:  # lambda x: x
            for i in arr.flat:
                if i:
                    count += 1
        case 1:  # lambda x: x == border
            for i in arr.flat:
                if i == border:
                    count += 1
        case 2:  # lambda x: x != border
            for i in arr.flat:
                if i != border:
                    count += 1
        case 3:  # lambda x: x < border
            for i in arr.flat:
                if i < border:
                    count += 1
        case 4:  # lambda x: x <= border
            for i in arr.flat:
                if i <= border:
                    count += 1
        case 5:  # lambda x: x > border
            for i in arr.flat:
                if i > border:
                    count += 1
        case 6:  # lambda x: x >= border
            for i in arr.flat:
                if i >= border:
                    count += 1
    return count


@njit(error_model="numpy")  # ,fastmath=True)
def jit_cov(X, ddof=1):
    n = X.shape[0]
    sum_0, sum_1, sum_cov = 0.0, 0.0, 0.0
    mean_0 = np.sum(X[:, 0]) / n
    mean_1 = np.sum(X[:, 1]) / n
    for i in range(n):
        sum_0 += (X[i, 0] - mean_0) ** 2
        sum_1 += (X[i, 1] - mean_1) ** 2
        sum_cov += (X[i, 0] - mean_0) * (X[i, 1] - mean_1)
    var_0 = sum_0 / (n - ddof)  # 不偏分散、sklearnに準拠
    var_1 = sum_1 / (n - ddof)
    cross_cov = sum_cov / (n - ddof)
    return var_0, var_1, cross_cov


@njit(error_model="numpy")  # ,fastmath=True)
def quartile_deviation(x):
    n_samples = x.shape[0]
    n_4 = n_samples // 4
    sorted_x = np.sort(x)
    match n_samples % 4:
        case 0:
            return (sorted_x[3 * n_4 - 1] + sorted_x[3 * n_4] - sorted_x[n_4 - 1] - sorted_x[n_4]) / 4
        case 1:
            return (sorted_x[3 * n_4] + sorted_x[3 * n_4 + 1] - sorted_x[n_4 - 1] - sorted_x[n_4]) / 4
        case 2:
            return (sorted_x[3 * n_4 + 1] - sorted_x[n_4]) / 2
        case 3:
            return (sorted_x[3 * n_4 + 2] - sorted_x[n_4]) / 2


def p_upper_x(n, x, pattern):
    def bin(x):
        # 二項分布の正規近似
        omega = n / pattern * (1 - 1 / pattern)
        return np.exp(-((x - n / 1 / pattern) ** 2) / (2 * omega)) / ((2 * np.pi * omega) ** 0.5)

    p, err = integrate.quad(bin, 0, x)
    return 1 - p**pattern


def sig_handler(signum, frame):
    sys.exit(1)


def emergency_save(logger, emergency_save_folder_path, **kwargs):
    now = datetime.datetime.now()
    file_path = f"{emergency_save_folder_path}/emergency_save_{now.strftime('%Y%m%d_%H%M%S')}"
    logger.error(f"emergency save path: {file_path}.npz")
    logger.error(f"emergency save keys: {list(kwargs.keys())}")
    np.savez(file_path, **kwargs)
