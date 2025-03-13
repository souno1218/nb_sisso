#!pip install -e /Users/sounosuke/Documents/python/nb_sisso

import numpy as np
import logging
from numba import njit
from nb_sisso import SIS
from nb_sisso.utils import decryption, eq_list_to_num, calc_RPN
from nb_sisso.model_score_1d import debug_1d


@njit(error_model="numpy")
def isclose(a, b, rtol=1e-06, atol=0):
    return np.abs(a - b) <= atol + rtol * min(np.abs(a), np.abs(b))


"""
@njit(error_model="numpy")
def isclose_arr(arr1, arr2, rtol=1e-06, atol=0):
    for i in range(arr1.shape[0]):
        if not isclose(arr1[i], arr2[i], rtol=rtol, atol=atol):
            return False
    return True
"""


@njit(error_model="numpy")
def isclose_arr(arr1, arr2, rtol=1e-7, atol=0):
    sum = 0.0
    len_arr = arr1.shape[0]
    for i in range(len_arr):
        sum += (np.abs(arr1[i] - arr2[i]) - atol) / min(np.abs(arr1[i]), np.abs(arr2[i]))
    return sum / len_arr <= rtol


@njit(error_model="numpy")
def is_inf(arr):
    for i in range(arr.shape[0]):
        if np.isinf(arr[i]):
            return True
    return False


@njit(error_model="numpy")
def is_all_zero(arr, atol=1e-06):
    for i in range(arr.shape[0]):
        if not isclose(0, arr[i], rtol=0, atol=atol):
            return False
    return True


@njit(error_model="numpy")
def is_all_const(arr, rtol=1e-06, atol=0):
    min, max, sum = arr[0], arr[0], arr[0]
    for i in range(1, arr.shape[0]):
        if min > arr[i]:
            min = arr[i]
        if max < arr[i]:
            max = arr[i]
        sum += arr[i]
    return (max - min) <= atol + rtol * np.abs(sum / arr.shape[0])


@njit(error_model="numpy")
def isclose_RMESE(arr1, arr2, border=1e-6):
    len_arr = arr1.shape[0]
    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    std1, std2, d = 0.0, 0.0, 0.0
    for i in range(len_arr):
        std1 += (arr1[i] - mean1) ** 2
        std2 += (arr2[i] - mean2) ** 2
        d += (arr1[i] - arr2[i]) ** 2
    if (std1 == 0) or (std2 == 0):
        return d == 0
    return d < border * np.sqrt(std1 * std2)


@njit(error_model="numpy")
def normalize(x):
    return_x = np.empty_like(x)
    n_sample = x.shape[0]
    mean = np.mean(x)
    sum = 0.0
    for i in range(n_sample):
        sum += (x[i] - mean) ** 2
    std = np.sqrt(sum / n_sample)
    plus_minus = np.sign(x[0] - mean)
    for i in range(n_sample):
        return_x[i] = plus_minus * (x[i] - mean) / std
    return return_x


@njit(error_model="numpy")
def mat_normalize(mat_x):
    return_mat_x = np.empty_like(mat_x)
    n_sample = mat_x.shape[1]
    for i in range(mat_x.shape[0]):
        mean = np.mean(mat_x[i])
        sum = 0.0
        for j in range(n_sample):
            sum += (mat_x[i, j] - mean) ** 2
        std = np.sqrt(sum / n_sample)
        plus_minus = np.sign(mat_x[i, 0] - mean)
        for j in range(n_sample):
            return_mat_x[i, j] = plus_minus * (mat_x[i, j] - mean) / std
    return return_mat_x


def make_unique_eqs(x, max_n_op, how_many_to_save=1000000, is_print=False):
    int_nan = -100
    operators_to_use = ["+", "-", "*", "/"]
    model_score = debug_1d
    logger = logging.getLogger(__name__)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    if is_print:
        _format = "%(asctime)s : %(message)s"
        logger.setLevel(logging.DEBUG)
        st_handler = logging.StreamHandler()
        st_handler.setLevel(logging.INFO)
        st_handler.setFormatter(logging.Formatter(_format))
        logger.addHandler(st_handler)
    y = np.random.choice((True, False), x.shape[1])
    units = np.zeros((x.shape[0], 1), dtype="int")
    _, eq = SIS(
        x,
        y,
        model_score=model_score,
        units=units,
        how_many_to_save=how_many_to_save,
        is_use_1=True,
        max_n_op=max_n_op,
        fast=True,
        operators_to_use=operators_to_use,
        log_interval=1,
        logger=logger,
    )
    eq = eq[: np.sum(eq[:, 0] != int_nan)]
    mat_x = eq_list_to_num(x, eq)
    normalized_mat_x = mat_normalize(mat_x)
    index = np.argsort(normalized_mat_x[:, 0])
    sorted_normalized_mat_x = normalized_mat_x[index].copy()
    eq = eq[index].copy()
    arr_columns = np.array([decryption(eq[i]) for i in range(eq.shape[0])])
    return sorted_normalized_mat_x, eq, arr_columns


@njit(error_model="numpy")
def check(n_random, max_n_op, x, calced_x, calced_eq, find):
    int_nan = -100
    head_calced_x = calced_x[:, 0].copy()
    random_eqs = np.full((2 * max_n_op + 1), int_nan, dtype="int8")
    n_ops = np.arange(max_n_op + 1)
    nums = np.arange(max_n_op + 2)
    ops = np.arange(-1, -5, -1)
    TFs = np.arange(2)
    calced_TF = np.zeros(calced_x.shape[0], dtype="bool")
    arange = np.arange(calced_x.shape[0])
    for _ in range(n_random):
        n_op = np.random.choice(n_ops)
        random_eqs[2 * n_op + 1 :] = int_nan
        count_num, count_op = 0, 0
        for i in range(2 * n_op + 1):
            if count_num <= count_op + 1:
                num_or_op = 0  # num
            elif n_op + 1 == count_num:
                num_or_op = 1  # op
            elif n_op == count_op:
                num_or_op = 0  # num
            else:
                num_or_op = np.random.choice(TFs)
            if num_or_op == 0:  # num
                random_eqs[i] = np.random.choice(nums)
                count_num += 1
            else:  # op
                random_eqs[i] = np.random.choice(ops)
                count_op += 1
        random_x = calc_RPN(x, random_eqs)
        if np.any(np.isnan(random_x)):
            continue
        elif is_inf(random_x):
            continue
        elif is_all_zero(random_x):
            continue
        elif is_all_const(random_x):
            continue
        else:
            normalized_random_x = normalize(random_x)
            count = 0
            calced_TF[:] = False
            for i in range(calced_x.shape[0]):
                if isclose(normalized_random_x[0], head_calced_x[i], rtol=1e-4):
                    if isclose_arr(normalized_random_x, calced_x[i]):
                        calced_TF[i] = True
                        count += 1
                        # print("")
                elif normalized_random_x[0] < head_calced_x[i]:
                    break
            if count == 0:
                if 0 in find:
                    print("not calced : ", random_eqs, normalized_random_x[0])
            elif count != 1:
                if 1 in find:
                    print("many calc : ", count, random_eqs, normalized_random_x[0])
                    for i in arange[calced_TF]:
                        print(calced_eq[i], calced_x[i, 0])
    return True


@njit(error_model="numpy")
def sub_finder(target_eq, x, calced_x, calced_eq):
    tagert_x = calc_RPN(x, target_eq)
    if np.any(np.isnan(tagert_x)):
        print("target x is contain nan")
    elif is_inf(tagert_x):
        print("target x is contain inf")
    elif is_all_zero(tagert_x):
        print("target x is all zero")
    elif is_all_const(tagert_x):
        print("target x is all const")
    else:
        head_calced_x = calced_x[:, 0].copy()
        normalized_tagert_x = normalize(tagert_x)
        calced_TF = np.zeros(calced_x.shape[0], dtype="bool")
        count = 0
        for i in range(calced_x.shape[0]):
            if isclose(normalized_tagert_x[0], head_calced_x[i], rtol=1e-4):
                if isclose_arr(normalized_tagert_x, calced_x[i]):
                    calced_TF[i] = True
                    count += 1
                elif normalized_tagert_x[0] < head_calced_x[i]:
                    break
        if count == 0:
            print("target eq is not calced")
        else:
            for i in range(calced_x.shape[0]):
                if calced_TF[i]:
                    print("same : ", calced_eq[i])


def random_test(
    max_n_op,
    n_random,
    len_x=30,
    how_many_to_save=1000000,
    is_print=False,
    x=None,
    calced_x=None,
    calced_eq=None,
    find=["not_calced", "many_calc"],
    upper=0.3,
    lower=10,
):
    num_find = []
    if "not_calced" in find:
        num_find.append(0)
    if "many_calc" in find:
        num_find.append(1)
    if x is None:
        if not calced_x is None:
            print("??????????")
            raise
        x = np.random.uniform(upper, lower, (max_n_op + 1, len_x))
    if calced_x is None:
        print("doing : make calced_x")
        calced_x, calced_eq, calced_columns = make_unique_eqs(
            x, max_n_op, how_many_to_save=how_many_to_save, is_print=is_print
        )
        print("done : make calced_x")
        if calced_x.shape[0] >= how_many_to_save:
            print(f"warning calced_x.shape[0]({calced_x.shape[0]}) >= how_many_to_save({how_many_to_save})")
    else:
        if calced_eq is None:
            print("??????????")
            raise
    print("doing : make random_x")
    all_calced = check(n_random, max_n_op, x, calced_x, calced_eq, num_find)
    print("done : make random_x")
    if all_calced:
        print("all calced.")


def finder(
    target_eq,
    max_n_op=None,
    len_x=30,
    how_many_to_save=1000000,
    is_print=False,
    x=None,
    calced_x=None,
    calced_eq=None,
    upper=0.3,
    lower=10,
):
    if max_n_op is None:
        max_n_op = (target_eq.shape[0] - 1) // 2
    if x is None:
        if not calced_x is None:
            print("??????????")
            raise
        x = np.random.uniform(upper, lower, (max_n_op + 1, len_x))
    if calced_x is None:
        print("doing : make calced_x")
        calced_x, calced_eq, calced_columns = make_unique_eqs(
            x, max_n_op, how_many_to_save=how_many_to_save, is_print=is_print
        )
        print("done : make calced_x")
        if calced_x.shape[0] >= how_many_to_save:
            print(f"warning calced_x.shape[0]({calced_x.shape[0]}) >= how_many_to_save({how_many_to_save})")
    else:
        if calced_eq is None:
            print("??????????")
            raise
    print("finding : ")
    sub_finder(target_eq, x, calced_x, calced_eq)
    print("done : make random_x")
