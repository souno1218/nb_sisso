import os
import numpy as np
import pandas as pd
import os
import sys
import datetime
import psutil
from numba_progress import ProgressBar
from numba import njit, prange, set_num_threads, objmode, get_num_threads
from numba_progress.numba_atomic import atomic_add, atomic_xchg, atomic_min
from utils import thread_check


@njit(error_model="numpy")
def cache_load(max_ops):
    unique_eq_dict = dict()
    for i in range(max_ops + 1):
        with objmode(data="int8[:,:]"):
            data = np.load(f"cache_folder/operator_{i}.npy")
        unique_eq_dict[i] = data
    return unique_eq_dict


@njit(error_model="numpy")
def make_dict_change_x_pattern(max_ops):
    dict_change_x_pattern = dict()
    dict_change_x_pattern[0] = np.zeros((1, 1), dtype="int8")
    dict_max_loop = np.ones((max_ops + 1, max_ops + 1), dtype="int64")
    for i in range(1, max_ops + 1):
        change_x_pattern = nb_permutations(max_ops + 1, i) + 1
        max_drop_num = np.zeros(change_x_pattern.shape[0], dtype="int8")
        for j in range(change_x_pattern.shape[0]):
            max_ = change_x_pattern[j, 0]
            save = max_ - 1
            for k in range(1, change_x_pattern.shape[1]):
                if max_ + 1 == change_x_pattern[j, k]:
                    max_ = change_x_pattern[j, k]
                elif max_ + 1 < change_x_pattern[j, k]:
                    max_ = change_x_pattern[j, k]
                    save = max_ - 1
            max_drop_num[j] = save
        dict_change_x_pattern[i] = change_x_pattern[np.argsort(max_drop_num)].astype(
            "int8"
        )
        dict_max_loop[i] = np.array(
            [np.sum(max_drop_num <= j) for j in range(np.max(max_drop_num) + 1)]
        )
    return dict_change_x_pattern, dict_max_loop


@njit(error_model="numpy")
def make_eq(base_back_equation, change_x_pattern):
    return_equation = base_back_equation.copy()
    for i in range(1, np.max(base_back_equation) + 1):
        return_equation[base_back_equation == i] = change_x_pattern[i - 1]
    return return_equation


@njit(error_model="numpy")
def nb_permutations(pattern, len):
    loop = 1
    for i in range(len):
        loop *= pattern - i
    return_arr = np.zeros((loop, len), dtype="int8")
    use_pattern = np.arange(pattern)
    for i in range(loop):
        mask = np.ones(pattern, dtype="bool")
        num = i
        for j in range(len):
            select_num = (use_pattern[mask])[num % (pattern - j)]
            return_arr[i, j] = select_num
            num //= pattern - j
            mask[select_num] = False
    return return_arr  # [nb_sort_sum_axis_1(return_arr)]


@njit(error_model="numpy")
def nb_calc_RPN(x, equation):
    Nan_number = -100
    stack = np.full((np.sum(equation >= 0), x.shape[1]), Nan_number, dtype="float64")
    stack[0] = x[equation[0]]
    last_stack_index = 0
    for i in range(1, equation.shape[0]):
        last_op = equation[i]
        if last_op >= 0:
            last_stack_index += 1
            stack[last_stack_index] = x[last_op]
        else:
            match last_op:
                case -1:  # +
                    last_stack_index -= 1
                    stack[last_stack_index] += stack[last_stack_index + 1]
                case -2:  # -
                    last_stack_index -= 1
                    stack[last_stack_index] -= stack[last_stack_index + 1]
                case -3:  # *
                    last_stack_index -= 1
                    stack[last_stack_index] *= stack[last_stack_index + 1]
                case -4:  # /
                    last_stack_index -= 1
                    stack[last_stack_index] /= stack[last_stack_index + 1]
                case Nan_number:
                    break
    return_stack = np.empty((2, x.shape[1]), dtype="float64")
    return_stack[:] = stack[0]
    if np.any(np.isinf(stack[0])):
        return_stack[:] = Nan_number
        return return_stack
    if is_all_zero(stack[0]):
        return_stack[0] = Nan_number
        return_stack[1] = 0
        return return_stack
    if is_all_const(stack[0]):
        return_stack[0] = Nan_number
        return return_stack
    # return_stack[0]=(stack[0]-np.mean(stack[0]))/np.sqrt(np.mean((stack[0]-np.mean(stack[0]))**2))
    return_stack[0] -= np.mean(stack[0])
    return_stack[0] /= np.sqrt(np.mean((stack[0] - np.mean(stack[0])) ** 2))
    return_stack[0] *= np.sign(return_stack[0, 0])
    return_stack[1] *= np.sign(return_stack[1, 0])
    return return_stack


@njit(error_model="numpy")
def is_all_zero(num):
    rtol = 1e-010
    return np.all(np.abs(num) <= rtol)


@njit(error_model="numpy")
def is_all_const(num):
    rtol = 1e-010
    return (np.max(num) - np.min(num)) / np.abs(np.mean(num)) <= rtol


@njit(error_model="numpy")
def is_close_arr(arr1, arr2):
    rtol = 1e-06
    if np.abs(arr1[0] - arr2[0]) <= rtol * np.abs(arr2[0]):
        if np.abs(arr1[1] - arr2[1]) <= rtol * np.abs(arr2[1]):
            if np.abs(arr1[2] - arr2[2]) <= rtol * np.abs(arr2[2]):
                if np.all(np.abs(arr1[3:] - arr2[3:]) <= rtol * np.abs(arr2[3:])):
                    return True
    return False


@njit(error_model="numpy")
def is_close_arr_index(num, arr):
    rtol = 1e-06
    return np.arange(arr.shape[0])[np.abs(arr - num) <= rtol * np.abs(num)]


@njit(error_model="numpy")
def find_similar_num(arr, mat):
    rtol = 1e-06
    for i in range(mat.shape[0]):
        if is_close_arr(arr, mat[i]):
            return True
    return False


@njit(error_model="numpy")
def find_similar_num_index(arr, mat):
    rtol = 1e-06
    for i in range(mat.shape[0]):
        if is_close_arr(arr, mat[i]):
            return i
    return -100


@njit(error_model="numpy")
def find_min_x_max(equation, random_x, random_for_find_min_x_max):
    rtol = 1e-06
    if np.sum(equation > 0) == 0:
        return False
    similar_num = nb_calc_RPN(random_x, equation)[0]
    max_x = np.max(equation)
    counter = np.zeros(max_x + 1, dtype="int8")
    for i in range(1, max_x + 1):
        counter[i] = np.sum(equation == i)
    change_num = np.arange(max_x + 1)[counter > 1][
        np.argsort(counter[counter > 1])[::-1]
    ]
    changed_random = random_x.copy()
    for i in change_num:
        changed_random[i] = random_for_find_min_x_max
        changed_similar_num = nb_calc_RPN(changed_random, equation)[0]
        if np.abs(similar_num[0] - changed_similar_num[0]) <= rtol * np.abs(
            changed_similar_num[0]
        ):
            if np.all(
                np.abs(similar_num - changed_similar_num)
                <= rtol * np.abs(changed_similar_num)
            ):
                return True
        changed_random[i] = random_x[i]
    return False


@njit(error_model="numpy")
def make_dict_mask_x(eq_x_max):
    dict_mask_x = dict()
    for i in range(eq_x_max + 1):
        __mask_x = nb_permutations(i, i) + 1
        mask_x = np.zeros((__mask_x.shape[0], __mask_x.shape[1] + 1), dtype="int8")
        mask_x[:, 1:] = __mask_x
        change_x_num = np.array(
            [make_change_x_id(mask_x[k, 1:], i) for k in range(mask_x.shape[0])]
        )
        index = np.argsort(change_x_num)
        dict_mask_x[i] = mask_x[index].copy()
    return dict_mask_x


@njit(parallel=True, error_model="numpy")
def make_before_similar_num_list(
    num_threads, max_op, random_x, before_equations, progress_proxy
):
    Nan_number = -100
    dict_mask_x = make_dict_mask_x(max_op)
    loop_per_threads = before_equations.shape[0] // num_threads + 1
    save_similar_num_list = np.full(
        (num_threads, 2, loop_per_threads, random_x.shape[1]),
        Nan_number,
        dtype="float64",
    )
    last_index = np.zeros((num_threads), dtype="int64")
    for thread_id in prange(num_threads):
        thread_similar_num_list = np.full(
            (2, loop_per_threads, random_x.shape[1]), Nan_number, dtype="float64"
        )
        for i in range(thread_id, before_equations.shape[0], num_threads):
            eq_x_max = np.max(before_equations[i])
            mask_x = dict_mask_x[eq_x_max]
            save_similar_num = nb_calc_RPN(random_x, before_equations[i])
            for k in range(1, mask_x.shape[0]):
                similar_num = nb_calc_RPN(random_x[mask_x[k]], before_equations[i])
                if save_similar_num[0, 0] > similar_num[0, 0]:
                    save_similar_num[0] = similar_num[0]
                if save_similar_num[1, 0] > similar_num[1, 0]:
                    save_similar_num[1] = similar_num[1]
            thread_similar_num_list[:, last_index[thread_id]] = save_similar_num
            last_index[thread_id] += 1
            progress_proxy.update(1)
        save_similar_num_list[thread_id] = thread_similar_num_list
    return_similar_num_list = np.full(
        (2, np.sum(last_index), random_x.shape[1]), Nan_number, dtype="float64"
    )
    tot_last_index = 0
    for thread_id in range(num_threads):
        return_similar_num_list[
            0, tot_last_index : tot_last_index + last_index[thread_id]
        ] = save_similar_num_list[thread_id, 0, : last_index[thread_id]]
        return_similar_num_list[
            1, tot_last_index : tot_last_index + last_index[thread_id]
        ] = save_similar_num_list[thread_id, 1, : last_index[thread_id]]
        tot_last_index += last_index[thread_id]
    return return_similar_num_list


@njit(error_model="numpy")
def loop_count(max_op, n_op1, num_threads, is_print=False):
    base_dict = cache_load(max_op - 1)
    _, dict_max_loop = make_dict_change_x_pattern(max_op)
    last_index = np.zeros(num_threads, dtype="int64")
    count = np.zeros(max_op, dtype="int64")
    n_op2 = max_op - 1 - n_op1
    base_eq_arr1 = base_dict[n_op1]
    base_eq_arr2 = base_dict[n_op2]
    loop = base_eq_arr1.shape[0] * base_eq_arr2.shape[0]
    for thread_id in prange(num_threads):
        for i in range(thread_id, loop, num_threads):
            id1 = i % base_eq_arr1.shape[0]
            i //= base_eq_arr1.shape[0]
            id2 = i % base_eq_arr2.shape[0]
            if n_op1 >= n_op2:
                last_index[thread_id] += (
                    4
                    * dict_max_loop[
                        np.max(base_eq_arr2[id2]), np.max(base_eq_arr1[id1])
                    ]
                )
            else:
                last_index[thread_id] += dict_max_loop[
                    np.max(base_eq_arr2[id2]), np.max(base_eq_arr1[id1])
                ]
    tot_loop, loop_per_threads = np.sum(last_index), np.max(last_index)
    # 要調整
    # if max_op>=5:
    # if n_op2==0:
    # loop_per_threads=int(loop_per_threads*0.71)
    # if (n_op2>=1)&(n_op1>=n_op2):
    # loop_per_threads=int(loop_per_threads*0.151)
    return tot_loop, loop_per_threads


@njit(error_model="numpy")
def make_change_x_id(mask, x_max):
    if np.sum(mask) == 0:
        return 0
    TF = np.ones(x_max, dtype="bool")
    return_num = 0
    for i in range(mask.shape[0]):
        return_num *= x_max - i
        return_num += np.sum(TF[: mask[i] - 1])
        TF[mask[i] - 1] = False
    return return_num


@njit(parallel=True, error_model="numpy")
def make_unique_equations_thread(
    max_op,
    n_op1,
    num_threads,
    random_x,
    before_similar_num_list,
    saved_similar_num_list,
    progress_proxy,
):
    Nan_number = -100
    n_op2 = max_op - 1 - n_op1
    random_for_find_min_x_max = np.random.random(random_x.shape[1])
    head_before_similar_num_list = before_similar_num_list[0, :, 0].copy()
    dict_change_x_pattern, dict_max_loop = make_dict_change_x_pattern(max_op)
    dict_mask_x = make_dict_mask_x(max_op + 1)
    base_dict = cache_load(max_op - 1)
    base_eq_arr1 = base_dict[n_op1]
    base_eq_arr2 = base_dict[n_op2]
    len_base_eq1 = base_eq_arr1.shape[1]
    len_base_eq2 = base_eq_arr2.shape[1]
    tot_mask_x = nb_permutations(max_op + 1, max_op + 1)
    _, loop_per_threads = loop_count(max_op, n_op1, num_threads)
    head_saved_similar_num_list = saved_similar_num_list[0, :, 0].copy()
    save_similar_num_list = np.full(
        (num_threads, 2, loop_per_threads, random_x.shape[1]),
        Nan_number,
        dtype="float64",
    )
    head_save_similar_num_list = np.full(
        (num_threads, loop_per_threads), Nan_number, dtype="float64"
    )
    TF_list = np.zeros((num_threads, loop_per_threads), dtype="bool")
    save_need_calc_list = np.zeros((num_threads, loop_per_threads), dtype="bool")
    if n_op1 >= n_op2:
        ops = [-1, -2, -3, -4]
    else:
        ops = [-4]
    with objmode():
        print(
            "         using memory : ",
            psutil.Process().memory_info().rss / 1024**2,
            "M bytes",
        )
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), Nan_number, dtype="int8")
        cache_for_mask_x = np.full(
            (tot_mask_x.shape[0], 2, random_x.shape[1]), Nan_number, dtype="float64"
        )
        counter = 0
        loop = base_eq_arr1.shape[0] * base_eq_arr2.shape[0]
        for i in range(thread_id, loop, num_threads):
            id1 = i % base_eq_arr1.shape[0]
            i //= base_eq_arr1.shape[0]
            id2 = i % base_eq_arr2.shape[0]
            front_equation = base_eq_arr1[id1]
            equation[:len_base_eq1] = front_equation
            flont_x_max = np.max(front_equation)
            base_back_equation = base_eq_arr2[id2]
            base_back_x_max = np.max(base_back_equation)
            for merge_op in ops:
                equation[len_base_eq1 + len_base_eq2] = merge_op
                for number in range(dict_max_loop[base_back_x_max, flont_x_max]):
                    is_save, is_calc, n = True, True, 0
                    equation[len_base_eq1 : len_base_eq1 + len_base_eq2] = make_eq(
                        base_back_equation,
                        dict_change_x_pattern[base_back_x_max][number],
                    )
                    eq_x_max = np.max(equation)
                    mask_x = dict_mask_x[eq_x_max]
                    for k in range(mask_x.shape[0]):
                        if k == 0:
                            if find_min_x_max(
                                equation, random_x, random_for_find_min_x_max
                            ):
                                is_save, is_calc = False, False
                        if is_save:
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
                            if k == 0:
                                if similar_num[0, 0] == Nan_number:
                                    if np.sum(equation > 0) != 0:
                                        is_save, is_calc = False, False
                                    elif np.all(similar_num[1] == 0):
                                        is_save, is_calc = False, False
                                    else:
                                        is_calc = False
                            elif find_similar_num(
                                similar_num[1], cache_for_mask_x[:n, 1]
                            ):
                                continue
                            if is_save:
                                cache_for_mask_x[n] = similar_num
                                n += 1
                            else:  # is_save=False
                                break
                    if is_save:
                        normalized_min_index = np.argmin(cache_for_mask_x[:n, 0, 0])
                        min_index = np.argmin(cache_for_mask_x[:n, 1, 0])
                        if max_op + 1 != eq_x_max:
                            same_head_index = is_close_arr_index(
                                cache_for_mask_x[normalized_min_index, 0, 0],
                                head_before_similar_num_list,
                            )
                            for t in same_head_index:
                                if is_close_arr(
                                    cache_for_mask_x[normalized_min_index, 0],
                                    before_similar_num_list[0, t],
                                ):
                                    is_calc = False
                                    if is_close_arr(
                                        cache_for_mask_x[min_index, 1],
                                        before_similar_num_list[1, t],
                                    ):
                                        is_save = False
                                        break
                    if is_save:
                        same_head_index = is_close_arr_index(
                            cache_for_mask_x[normalized_min_index, 0, 0],
                            head_saved_similar_num_list,
                        )
                        for t in same_head_index:
                            if is_close_arr(
                                cache_for_mask_x[normalized_min_index, 0],
                                saved_similar_num_list[0, t],
                            ):
                                is_calc = False
                                if is_close_arr(
                                    cache_for_mask_x[min_index, 1],
                                    saved_similar_num_list[1, t],
                                ):
                                    is_save = False
                                    break
                    if is_save:
                        same_head_index = is_close_arr_index(
                            cache_for_mask_x[normalized_min_index, 0, 0],
                            head_save_similar_num_list[thread_id, :counter],
                        )
                        for t in same_head_index:
                            if is_close_arr(
                                cache_for_mask_x[normalized_min_index, 0],
                                save_similar_num_list[thread_id, 0, t],
                            ):
                                is_calc = False
                                if is_close_arr(
                                    cache_for_mask_x[min_index, 1],
                                    save_similar_num_list[thread_id, 1, t],
                                ):
                                    is_save = False
                                    break
                    if is_save:
                        TF_list[thread_id, counter] = True
                        save_similar_num_list[thread_id, 0, counter] = cache_for_mask_x[
                            normalized_min_index, 0
                        ]
                        save_similar_num_list[thread_id, 1, counter] = cache_for_mask_x[
                            min_index, 1
                        ]
                        head_save_similar_num_list[thread_id, counter] = (
                            cache_for_mask_x[normalized_min_index, 0, 0]
                        )
                        save_need_calc_list[thread_id, counter] = is_calc
                    counter += 1
                    progress_proxy.update(1)
    return TF_list, save_similar_num_list, save_need_calc_list


@njit(parallel=True, error_model="numpy")
def dim_reduction(TF_list, similar_num_list, need_calc_list, progress_proxy):
    Nan_number = -100
    num_threads = similar_num_list.shape[0]
    threads_last_index = np.array([np.sum(TF_list[i]) for i in range(num_threads)])
    sort_index = np.argsort(threads_last_index)[::-1]
    return_similar_num_list = np.full(
        (2, np.sum(TF_list), similar_num_list.shape[3]), Nan_number, dtype="float64"
    )
    return_similar_num_list[0, : np.sum(TF_list[sort_index[0]])] = similar_num_list[
        sort_index[0], 0, TF_list[sort_index[0]]
    ]
    return_similar_num_list[1, : np.sum(TF_list[sort_index[0]])] = similar_num_list[
        sort_index[0], 1, TF_list[sort_index[0]]
    ]
    head_return_similar_num_list = np.full(
        (np.sum(TF_list)), Nan_number, dtype="float64"
    )
    head_similar_num_list = similar_num_list[:, 0, :, 0]
    head_return_similar_num_list[: np.sum(TF_list[sort_index[0]])] = (
        head_similar_num_list[sort_index[0], TF_list[sort_index[0]]]
    )
    last_index = np.sum(TF_list[sort_index[0]])
    with objmode():
        print(
            "         using memory : ",
            psutil.Process().memory_info().rss / 1024**2,
            "M bytes",
        )
    for target_index in sort_index[1:]:
        true_index = np.arange(TF_list.shape[1])[TF_list[target_index]]
        for thread_id in prange(num_threads):
            true_index_thread = true_index[thread_id::num_threads].copy()
            for i in true_index_thread:
                for t in is_close_arr_index(
                    similar_num_list[target_index, 0, i, 0],
                    head_return_similar_num_list[:last_index],
                ):
                    if is_close_arr(
                        similar_num_list[target_index, 0, i],
                        return_similar_num_list[0, t],
                    ):
                        need_calc_list[target_index, i] = False
                        if is_close_arr(
                            similar_num_list[target_index, 1, i],
                            return_similar_num_list[1, t],
                        ):
                            TF_list[target_index, i] = False
                            break
                progress_proxy.update(1)
        return_similar_num_list[
            0, last_index : last_index + np.sum(TF_list[target_index])
        ] = similar_num_list[target_index, 0, TF_list[target_index]]
        return_similar_num_list[
            1, last_index : last_index + np.sum(TF_list[target_index])
        ] = similar_num_list[target_index, 1, TF_list[target_index]]
        head_return_similar_num_list[
            last_index : last_index + np.sum(TF_list[target_index])
        ] = head_similar_num_list[target_index, TF_list[target_index]]
        last_index += np.sum(TF_list[target_index])
    return TF_list, need_calc_list


@njit(parallel=True, error_model="numpy")
def make_unique_equations_info(
    max_op, n_op1, TF_list, need_calc_list, random_x, progress_proxy
):
    Nan_number = -100
    n_op2 = max_op - 1 - n_op1
    base_dict = cache_load(max_op - 1)
    tot_mask_x = nb_permutations(max_op + 1, max_op + 1)
    if n_op1 >= n_op2:
        ops = [-1, -2, -3, -4]
    else:
        ops = [-4]
    sum_TF = np.sum(TF_list)
    num_threads = TF_list.shape[0]
    return_similar_num_list = np.full(
        (2, sum_TF, random_x.shape[1]), Nan_number, dtype="float64"
    )
    return_equation_list = np.full((sum_TF, 2 * max_op + 1), Nan_number, dtype="int8")
    return_base_eq_id_list = np.full((sum_TF, 5), Nan_number, dtype="int64")
    return_need_calc_list = np.zeros((sum_TF), dtype="bool")
    return_need_calc_change_x = np.zeros((sum_TF, tot_mask_x.shape[0]), dtype="bool")
    with objmode():
        print(
            "         using memory : ",
            psutil.Process().memory_info().rss / 1024**2,
            "M bytes",
        )
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), Nan_number, dtype="int8")
        cache_for_mask_x = np.full(
            (tot_mask_x.shape[0], 2, random_x.shape[1]), Nan_number, dtype="float64"
        )
        last_index = np.sum(TF_list[:thread_id])
        TF_list_thread = TF_list[thread_id]
        thread_need_calc_change_x = np.ones((tot_mask_x.shape[0]), dtype="bool")
        dict_change_x_pattern, dict_max_loop = make_dict_change_x_pattern(max_op)
        dict_mask_x = make_dict_mask_x(max_op + 1)
        base_eq_arr1 = base_dict[n_op1]
        base_eq_arr2 = base_dict[n_op2]
        len_base_eq1 = base_eq_arr1.shape[1]
        len_base_eq2 = base_eq_arr2.shape[1]
        counter = 0
        loop = base_eq_arr1.shape[0] * base_eq_arr2.shape[0]
        for i in range(thread_id, loop, num_threads):
            id1 = i % base_eq_arr1.shape[0]
            i //= base_eq_arr1.shape[0]
            id2 = i % base_eq_arr2.shape[0]
            front_equation = base_eq_arr1[id1]
            equation[:len_base_eq1] = front_equation
            flont_x_max = np.max(front_equation)
            base_back_equation = base_eq_arr2[id2]
            base_back_x_max = np.max(base_back_equation)
            for merge_op in ops:
                equation[len_base_eq1 + len_base_eq2] = merge_op
                for number in range(dict_max_loop[base_back_x_max, flont_x_max]):
                    equation[len_base_eq1 : len_base_eq1 + len_base_eq2] = make_eq(
                        base_back_equation,
                        dict_change_x_pattern[base_back_x_max][number],
                    )
                    if TF_list_thread[counter]:
                        n = 0
                        equation[len_base_eq1 : len_base_eq1 + len_base_eq2] = make_eq(
                            base_back_equation,
                            dict_change_x_pattern[base_back_x_max][number],
                        )
                        eq_x_max = np.max(equation)
                        mask_x = dict_mask_x[eq_x_max]
                        thread_need_calc_change_x[:] = True
                        for k in range(mask_x.shape[0]):
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
                            if find_similar_num(
                                similar_num[1], cache_for_mask_x[:n, 1]
                            ):
                                thread_need_calc_change_x[k] = False
                                # neeeeeeeed
                                continue
                            else:
                                cache_for_mask_x[n] = similar_num
                                n += 1
                        normalized_min_index = np.argmin(cache_for_mask_x[:n, 0, 0])
                        min_index = np.argmin(cache_for_mask_x[:n, 1, 0])
                        return_similar_num_list[0, last_index] = cache_for_mask_x[
                            normalized_min_index, 0
                        ]
                        return_similar_num_list[1, last_index] = cache_for_mask_x[
                            min_index, 1
                        ]
                        return_equation_list[last_index] = equation
                        return_need_calc_list[last_index] = need_calc_list[
                            thread_id, counter
                        ]
                        return_base_eq_id_list[last_index, 0] = merge_op
                        return_base_eq_id_list[last_index, 1] = n_op1
                        return_base_eq_id_list[last_index, 2] = id1
                        return_base_eq_id_list[last_index, 3] = id2
                        changed_back_eq_id = make_change_x_id(
                            dict_change_x_pattern[base_back_x_max][number], max_op + 1
                        )
                        return_base_eq_id_list[last_index, 4] = changed_back_eq_id
                        return_need_calc_change_x[last_index, : mask_x.shape[0]] = (
                            thread_need_calc_change_x[: mask_x.shape[0]]
                        )
                        last_index += 1
                        progress_proxy.update(1)
                    counter += 1
    return (
        return_equation_list,
        return_similar_num_list,
        return_need_calc_list,
        return_base_eq_id_list,
        return_need_calc_change_x,
    )


@njit(parallel=True, error_model="numpy")
def make_unique_equations_thread_7(
    n_op1,
    num_threads,
    random_x,
    before_similar_num_list,
    saved_similar_num_list,
    progress_proxy,
):
    Nan_number = -100
    max_op = 7
    n_op2 = max_op - 1 - n_op1
    head_before_similar_num_list = before_similar_num_list[:, 0].copy()
    random_for_find_min_x_max = np.random.random(random_x.shape[1])
    dict_change_x_pattern, dict_max_loop = make_dict_change_x_pattern(max_op + 1)
    dict_mask_x = make_dict_mask_x(max_op + 1)
    base_dict = cache_load(max_op - 1)
    base_eq_arr1 = base_dict[n_op1]
    base_eq_arr2 = base_dict[n_op2]
    len_base_eq1 = base_eq_arr1.shape[1]
    len_base_eq2 = base_eq_arr2.shape[1]
    tot_mask_x = nb_permutations(max_op + 1, max_op + 1)
    _, loop_per_threads = loop_count(max_op, n_op1, num_threads)
    head_saved_similar_num_list = saved_similar_num_list[:, 0].copy()
    save_similar_num_list = np.full(
        (num_threads, loop_per_threads, random_x.shape[1]), Nan_number, dtype="float64"
    )
    head_save_similar_num_list = np.full(
        (num_threads, loop_per_threads), Nan_number, dtype="float64"
    )
    TF_list = np.zeros((num_threads, loop_per_threads), dtype="bool")
    if n_op1 >= n_op2:
        ops = [-1, -2, -3, -4]
    else:
        ops = [-4]
    with objmode():
        print(
            "         using memory : ",
            psutil.Process().memory_info().rss / 1024**2,
            "M bytes",
        )
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), Nan_number, dtype="int8")
        cache_for_mask_x = np.full(
            (tot_mask_x.shape[0], 2, random_x.shape[1]), Nan_number, dtype="float64"
        )
        counter = 0
        loop = base_eq_arr1.shape[0] * base_eq_arr2.shape[0]
        for i in range(thread_id, loop, num_threads):
            id1 = i % base_eq_arr1.shape[0]
            i //= base_eq_arr1.shape[0]
            id2 = i % base_eq_arr2.shape[0]
            front_equation = base_eq_arr1[id1]
            equation[:len_base_eq1] = front_equation
            flont_x_max = np.max(front_equation)
            base_back_equation = base_eq_arr2[id2]
            base_back_x_max = np.max(base_back_equation)
            for merge_op in ops:
                equation[len_base_eq1 + len_base_eq2] = merge_op
                for number in range(dict_max_loop[base_back_x_max, flont_x_max]):
                    is_save, n = True, 0
                    equation[len_base_eq1 : len_base_eq1 + len_base_eq2] = make_eq(
                        base_back_equation,
                        dict_change_x_pattern[base_back_x_max][number],
                    )
                    eq_x_max = np.max(equation)
                    mask_x = dict_mask_x[eq_x_max]
                    for k in range(mask_x.shape[0]):
                        if k == 0:
                            if find_min_x_max(
                                equation, random_x, random_for_find_min_x_max
                            ):
                                is_save = False
                        if is_save:
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
                            if k == 0:
                                if similar_num[0, 0] == Nan_number:
                                    is_save = False
                            elif find_similar_num(
                                similar_num[0], cache_for_mask_x[:n, 0]
                            ):
                                continue
                            if is_save:
                                cache_for_mask_x[n] = similar_num
                                n += 1
                            else:  # is_save=False
                                break
                    if is_save:
                        normalized_min_index = np.argmin(cache_for_mask_x[:n, 0, 0])
                        if max_op + 1 != eq_x_max:
                            same_head_index = is_close_arr_index(
                                cache_for_mask_x[normalized_min_index, 0, 0],
                                head_before_similar_num_list,
                            )
                            for t in same_head_index:
                                if is_close_arr(
                                    cache_for_mask_x[normalized_min_index, 0],
                                    before_similar_num_list[t],
                                ):
                                    is_save = False
                                    break
                    if is_save:
                        same_head_index = is_close_arr_index(
                            cache_for_mask_x[normalized_min_index, 0, 0],
                            head_saved_similar_num_list,
                        )
                        for t in same_head_index:
                            if is_close_arr(
                                cache_for_mask_x[normalized_min_index, 0],
                                saved_similar_num_list[t],
                            ):
                                is_save = False
                                break
                    if is_save:
                        same_head_index = is_close_arr_index(
                            cache_for_mask_x[normalized_min_index, 0, 0],
                            head_save_similar_num_list[thread_id, :counter],
                        )
                        for t in same_head_index:
                            if is_close_arr(
                                cache_for_mask_x[normalized_min_index, 0],
                                save_similar_num_list[thread_id, t],
                            ):
                                is_save = False
                                break
                    if is_save:
                        TF_list[thread_id, counter] = True
                        save_similar_num_list[thread_id, counter] = cache_for_mask_x[
                            normalized_min_index, 0
                        ]
                        head_save_similar_num_list[thread_id, counter] = (
                            cache_for_mask_x[normalized_min_index, 0, 0]
                        )
                    counter += 1
                    progress_proxy.update(1)
    return TF_list, save_similar_num_list


@njit(parallel=True, error_model="numpy")
def dim_reduction_7(TF_list, similar_num_list, progress_proxy):
    Nan_number = -100
    num_threads = similar_num_list.shape[0]
    threads_last_index = np.array([np.sum(TF_list[i]) for i in range(num_threads)])
    sort_index = np.argsort(threads_last_index)[::-1]
    return_similar_num_list = np.full(
        (np.sum(TF_list), similar_num_list.shape[2]), Nan_number, dtype="float64"
    )
    return_similar_num_list[: np.sum(TF_list[sort_index[0]])] = similar_num_list[
        sort_index[0], TF_list[sort_index[0]]
    ]
    head_return_similar_num_list = np.full(
        (np.sum(TF_list)), Nan_number, dtype="float64"
    )
    head_similar_num_list = similar_num_list[:, :, 0]
    head_return_similar_num_list[: np.sum(TF_list[sort_index[0]])] = (
        head_similar_num_list[sort_index[0], TF_list[sort_index[0]]]
    )
    last_index = np.sum(TF_list[sort_index[0]])
    with objmode():
        print(
            "         using memory : ",
            psutil.Process().memory_info().rss / 1024**2,
            "M bytes",
        )
    for target_index in sort_index[1:]:
        true_index = np.arange(TF_list.shape[1])[TF_list[target_index]]
        for thread_id in prange(num_threads):
            true_index_thread = true_index[thread_id::num_threads].copy()
            for i in true_index_thread:
                for t in is_close_arr_index(
                    similar_num_list[target_index, i, 0],
                    head_return_similar_num_list[:last_index],
                ):
                    if is_close_arr(
                        similar_num_list[target_index, i], return_similar_num_list[t]
                    ):
                        TF_list[target_index, i] = False
                        break
                progress_proxy.update(1)
        return_similar_num_list[
            last_index : last_index + np.sum(TF_list[target_index])
        ] = similar_num_list[target_index, TF_list[target_index]]
        head_return_similar_num_list[
            last_index : last_index + np.sum(TF_list[target_index])
        ] = head_similar_num_list[target_index, TF_list[target_index]]
        last_index += np.sum(TF_list[target_index])
    return TF_list


@njit(parallel=True, error_model="numpy")
def make_unique_equations_info_7(n_op1, TF_list, random_x, progress_proxy):
    Nan_number = -100
    max_op = 7
    n_op2 = max_op - 1 - n_op1
    base_dict = cache_load(max_op - 1)
    tot_mask_x = nb_permutations(max_op + 1, max_op + 1)
    if n_op1 >= n_op2:
        ops = [-1, -2, -3, -4]
    else:
        ops = [-4]
    sum_TF = np.sum(TF_list)
    num_threads = TF_list.shape[0]
    return_similar_num_list = np.full(
        (sum_TF, random_x.shape[1]), Nan_number, dtype="float64"
    )
    return_equation_list = np.full((sum_TF, 2 * max_op + 1), Nan_number, dtype="int8")
    return_base_eq_id_list = np.full((sum_TF, 5), Nan_number, dtype="int64")
    return_need_calc_change_x = np.zeros((sum_TF, tot_mask_x.shape[0]), dtype="bool")
    with objmode():
        print(
            f"         using memory : {psutil.Process().memory_info().rss / 1024**2} M bytes"
        )
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), Nan_number, dtype="int8")
        cache_for_mask_x = np.full(
            (tot_mask_x.shape[0], 2, random_x.shape[1]), Nan_number, dtype="float64"
        )
        last_index = np.sum(TF_list[:thread_id])
        TF_list_thread = TF_list[thread_id].copy()
        thread_need_calc_change_x = np.ones((tot_mask_x.shape[0]), dtype="bool")
        dict_change_x_pattern, dict_max_loop = make_dict_change_x_pattern(max_op)
        dict_mask_x = make_dict_mask_x(max_op + 1)
        base_eq_arr1 = base_dict[n_op1]
        base_eq_arr2 = base_dict[n_op2]
        len_base_eq1 = base_eq_arr1.shape[1]
        len_base_eq2 = base_eq_arr2.shape[1]
        counter = 0
        loop = base_eq_arr1.shape[0] * base_eq_arr2.shape[0]
        for i in range(thread_id, loop, num_threads):
            id1 = i % base_eq_arr1.shape[0]
            i //= base_eq_arr1.shape[0]
            id2 = i % base_eq_arr2.shape[0]
            front_equation = base_eq_arr1[id1]
            equation[:len_base_eq1] = front_equation
            flont_x_max = np.max(front_equation)
            base_back_equation = base_eq_arr2[id2]
            base_back_x_max = np.max(base_back_equation)
            for merge_op in ops:
                equation[len_base_eq1 + len_base_eq2] = merge_op
                for number in range(dict_max_loop[base_back_x_max, flont_x_max]):
                    if TF_list_thread[counter]:
                        n = 0
                        equation[len_base_eq1 : len_base_eq1 + len_base_eq2] = make_eq(
                            base_back_equation,
                            dict_change_x_pattern[base_back_x_max][number],
                        )
                        eq_x_max = np.max(equation)
                        mask_x = dict_mask_x[eq_x_max]
                        thread_need_calc_change_x[:] = True
                        for k in range(mask_x.shape[0]):
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
                            if find_similar_num(
                                similar_num[1], cache_for_mask_x[:n, 1]
                            ):
                                thread_need_calc_change_x[k] = False
                                # neeeeeeeed
                                continue
                            else:
                                cache_for_mask_x[n] = similar_num
                                n += 1
                        normalized_min_index = np.argmin(cache_for_mask_x[:n, 0, 0])
                        return_similar_num_list[last_index] = cache_for_mask_x[
                            normalized_min_index, 0
                        ]
                        return_equation_list[last_index] = equation
                        return_base_eq_id_list[last_index, 0] = merge_op
                        return_base_eq_id_list[last_index, 1] = n_op1
                        return_base_eq_id_list[last_index, 2] = id1
                        return_base_eq_id_list[last_index, 3] = id2
                        changed_back_eq_id = make_change_x_id(
                            dict_change_x_pattern[base_back_x_max][number], max_op + 1
                        )
                        return_base_eq_id_list[last_index, 4] = changed_back_eq_id
                        return_need_calc_change_x[last_index, : mask_x.shape[0]] = (
                            thread_need_calc_change_x[: mask_x.shape[0]]
                        )
                        last_index += 1
                        progress_proxy.update(1)
                    counter += 1
    return (
        return_equation_list,
        return_similar_num_list,
        return_base_eq_id_list,
        return_need_calc_change_x,
    )


@njit(parallel=True, error_model="numpy")
def make_similar_num_by_saved_7(n_op1, TF_list, random_x, progress_proxy):
    Nan_number = -100
    max_op = 7
    n_op2 = max_op - 1 - n_op1
    base_dict = cache_load(max_op - 1)
    tot_mask_x = nb_permutations(max_op + 1, max_op + 1)
    if n_op1 >= n_op2:
        ops = [-1, -2, -3, -4]
    else:
        ops = [-4]
    num_threads = TF_list.shape[0]
    return_similar_num_list = np.full(
        (num_threads, TF_list.shape[1], random_x.shape[1]), Nan_number, dtype="float64"
    )
    with objmode():
        print(
            "         using memory : ",
            psutil.Process().memory_info().rss / 1024**2,
            "M bytes",
        )
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), Nan_number, dtype="int8")
        last_index = np.sum(TF_list[:thread_id])
        TF_list_thread = TF_list[thread_id].copy()
        dict_change_x_pattern, dict_max_loop = make_dict_change_x_pattern(max_op)
        dict_mask_x = make_dict_mask_x(max_op + 1)
        base_eq_arr1 = base_dict[n_op1]
        base_eq_arr2 = base_dict[n_op2]
        len_base_eq1 = base_eq_arr1.shape[1]
        len_base_eq2 = base_eq_arr2.shape[1]
        counter = 0
        loop = base_eq_arr1.shape[0] * base_eq_arr2.shape[0]
        for i in range(thread_id, loop, num_threads):
            id1 = i % base_eq_arr1.shape[0]
            i //= base_eq_arr1.shape[0]
            id2 = i % base_eq_arr2.shape[0]
            front_equation = base_eq_arr1[id1]
            equation[:len_base_eq1] = front_equation
            flont_x_max = np.max(front_equation)
            base_back_equation = base_eq_arr2[id2]
            base_back_x_max = np.max(base_back_equation)
            for merge_op in ops:
                equation[len_base_eq1 + len_base_eq2] = merge_op
                for number in range(dict_max_loop[base_back_x_max, flont_x_max]):
                    if TF_list_thread[counter]:
                        equation[len_base_eq1 : len_base_eq1 + len_base_eq2] = make_eq(
                            base_back_equation,
                            dict_change_x_pattern[base_back_x_max][number],
                        )
                        eq_x_max = np.max(equation)
                        mask_x = dict_mask_x[eq_x_max]
                        save_similar_num = nb_calc_RPN(random_x[mask_x[0]], equation)
                        for k in range(1, mask_x.shape[0]):
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
                            if save_similar_num[1, 0] > similar_num[1, 0]:
                                save_similar_num = similar_num
                        return_similar_num_list[thread_id, counter] = save_similar_num[
                            0
                        ]
                        progress_proxy.update(1)
                    counter += 1
    return return_similar_num_list


def make_final_cache(
    max_ops,
    num_threads,
    equation_list,
    need_calc_list,
    base_eq_id_list,
    need_calc_change_x,
):
    # base_eq_id_list = merge_op,n_op1,id1,id2,changed_back_eq_id
    Nan_number = -100
    base_dict = cache_load(max_ops - 1)
    return_cache = dict()
    return_dict_num_to_index = dict()
    return_np_index_to_num = dict()
    max_id_need_calc = np.load("cache_folder/max_id_need_calc.npy")
    base = np.zeros(max(max_id_need_calc.shape[0], max_ops + 1), dtype="int64")
    base[: max_id_need_calc.shape[0]] = max_id_need_calc
    base[max_ops] = np.sum(need_calc_list)
    logger.info(f"   max_id_need_calc, {list(max_id_need_calc)} -> {list(base)}")
    np.save("cache_folder/max_id_need_calc", base)
    for n_op1 in range(max_ops - 1, -1, -1):
        n_op2 = max_ops - 1 - n_op1
        unique_num = np.empty((0), dtype="int64")
        for i in (1, n_op2 + 2):
            all_pattern = nb_permutations(max_ops + 1, i) + 1
            all_num = np.array(
                [
                    make_change_x_id(all_pattern[i], max_ops + 1)
                    for i in range(all_pattern.shape[0])
                ]
            )
            unique_num = np.unique(np.concatenate((unique_num, all_num)))
        dict_num_to_index = {j: i for i, j in enumerate(unique_num)}
        if n_op1 >= n_op2:
            cache = np.full(
                (
                    base_dict[n_op1].shape[0],
                    base_dict[n_op2].shape[0],
                    unique_num.shape[0],
                    4,
                ),
                Nan_number,
                dtype="int64",
            )
        else:
            cache = np.full(
                (
                    base_dict[n_op1].shape[0],
                    base_dict[n_op2].shape[0],
                    unique_num.shape[0],
                    1,
                ),
                Nan_number,
                dtype="int64",
            )
        index = np.arange(base_eq_id_list.shape[0])[base_eq_id_list[:, 1] == n_op1]
        for i in index:
            cache[
                base_eq_id_list[i, 2],
                base_eq_id_list[i, 3],
                dict_num_to_index[base_eq_id_list[i, 4]],
                base_eq_id_list[i, 0] + 4,
            ] = i
        return_cache[n_op1] = cache
        return_dict_num_to_index[n_op1] = dict_num_to_index
        return_np_index_to_num[n_op1] = unique_num
    np.savez(
        f"cache_folder/cache_{max_ops}",
        **{str(n_op1): return_cache[n_op1] for n_op1 in range(max_ops)},
    )
    np.savez(
        f"cache_folder/num_to_index_{max_ops}",
        **{str(n_op1): return_np_index_to_num[n_op1] for n_op1 in range(max_ops)},
    )
    np.savez(f"cache_folder/need_calc_change_x_{max_ops}", need_calc_change_x)
    # savez_compressed
    np.save(f"cache_folder/operator_{max_ops}", equation_list)


def make_unique_equations(
    max_op,
    num_threads,
    random_x,
    before_similar_num_list,
    bar_format,
    is_print,
    ProgressBar_disable,
):
    num_threads = int(num_threads)
    Nan_number = -100
    saved_equation_list = np.empty((0, 2 * max_op + 1), dtype="int8")
    saved_base_eq_id_list = np.empty((0, 5), dtype="int8")
    tot_mask_x = nb_permutations(max_op + 1, max_op + 1)
    saved_need_calc_change_x = np.empty((0, tot_mask_x.shape[0]), dtype="bool")
    if max_op != 7:
        saved_need_calc_list = np.empty((0), dtype="bool")
        saved_similar_num_list = np.empty((2, 0, random_x.shape[1]), dtype="float64")
        time1, time2 = datetime.datetime.now(), datetime.datetime.now()
        for n_op1 in range(max_op - 1, -1, -1):
            n_op2 = max_op - 1 - n_op1
            logger.info(f"   n_op1={n_op1},n_op2={n_op2}")
            how_loop, loop_per_threads = loop_count(
                max_op, n_op1, num_threads, is_print=is_print
            )
            logger.info(f"      make_unique_equations_thread")
            mem_size_per_1data = 2 * random_x.shape[1] * 8 + 8 + 1 + 1
            mem_size = (
                (mem_size_per_1data * loop_per_threads * num_threads) // 100000
            ) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
            )
            time1, time2 = time2, datetime.datetime.now()
            with ProgressBar(
                total=how_loop,
                dynamic_ncols=False,
                disable=ProgressBar_disable,
                bar_format=bar_format,
                leave=False,
            ) as progress:
                TF_list, similar_num_list, need_calc_list = (
                    make_unique_equations_thread(
                        max_op,
                        n_op1,
                        num_threads,
                        random_x,
                        before_similar_num_list,
                        saved_similar_num_list,
                        progress,
                    )
                )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")
            logger.info(f"      dim_reduction")
            how_loop = np.sum(np.sort(np.sum(TF_list, axis=1))[::-1][1:])
            mem_size_per_1data = 2 * random_x.shape[1] * 8 + 8
            mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
            loop = np.sum(TF_list)
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={loop})"
            )
            time1, time2 = time2, datetime.datetime.now()
            with ProgressBar(
                total=how_loop,
                dynamic_ncols=False,
                disable=ProgressBar_disable,
                bar_format=bar_format,
                leave=False,
            ) as progress:
                TF_list, need_calc_list = dim_reduction(
                    TF_list, similar_num_list, need_calc_list, progress
                )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")
            logger.info(f"      make_unique_equations_info")
            how_loop = np.sum(TF_list)
            mem_size_per_1data = (
                2 * random_x.shape[1] * 8
                + (2 * max_op + 1)
                + 5 * 8
                + 1
                + tot_mask_x.shape[0]
            )
            mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
            )
            time1, time2 = time2, datetime.datetime.now()
            with ProgressBar(
                total=how_loop,
                dynamic_ncols=False,
                disable=ProgressBar_disable,
                bar_format=bar_format,
                leave=False,
            ) as progress:
                (
                    equation_list,
                    similar_num_list,
                    need_calc_list,
                    base_eq_id_list,
                    need_calc_change_x,
                ) = make_unique_equations_info(
                    max_op, n_op1, TF_list, need_calc_list, random_x, progress
                )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")
            saved_equation_list = np.concatenate((saved_equation_list, equation_list))
            saved_similar_num_list = np.concatenate(
                (saved_similar_num_list, similar_num_list), axis=1
            )
            saved_need_calc_list = np.concatenate(
                (saved_need_calc_list, need_calc_list)
            )
            saved_base_eq_id_list = np.concatenate(
                (saved_base_eq_id_list, base_eq_id_list)
            )
            saved_need_calc_change_x = np.concatenate(
                (saved_need_calc_change_x, need_calc_change_x)
            )
        sort_index = np.argsort(saved_need_calc_list.astype("int8"))[::-1]
        saved_equation_list = saved_equation_list[sort_index].copy()
        saved_similar_num_list = saved_similar_num_list[:, sort_index].copy()
        saved_need_calc_list = saved_need_calc_list[sort_index].copy()
        saved_base_eq_id_list = saved_base_eq_id_list[sort_index].copy()
        saved_need_calc_change_x = saved_need_calc_change_x[sort_index].copy()
        make_final_cache(
            max_op,
            num_threads,
            saved_equation_list,
            saved_need_calc_list,
            saved_base_eq_id_list,
            saved_need_calc_change_x,
        )
    else:  # max_op==7
        saved_similar_num_list = np.empty((0, random_x.shape[1]), dtype="float64")
        before_similar_num_list_7 = before_similar_num_list[0].copy()
        time1, time2 = datetime.datetime.now(), datetime.datetime.now()
        for n_op1 in range(max_op - 1, -1, -1):
            n_op2 = max_op - 1 - n_op1
            logger.info(f"   n_op1={n_op1},n_op2={n_op2}")
            how_loop, loop_per_threads = loop_count(
                max_op, n_op1, num_threads, is_print=is_print
            )
            logger.info(f"      make_unique_equations_thread")
            if os.path.isfile(
                f"cache_folder/saved_7_make_unique_equations_thread_{n_op1}_{num_threads}.npz"
            ):
                TF_list = np.load(
                    f"cache_folder/saved_7_make_unique_equations_thread_{n_op1}_{num_threads}.npz"
                )["TF_list"]
                how_loop = np.sum(TF_list)
                logger.info(f"         make_similar_num_by_saved_7")
                with ProgressBar(
                    total=how_loop,
                    dynamic_ncols=False,
                    disable=ProgressBar_disable,
                    bar_format=bar_format,
                    leave=False,
                ) as progress:
                    similar_num_list = make_similar_num_by_saved_7(
                        n_op1, TF_list, random_x, progress
                    )
            else:
                mem_size_per_1data = random_x.shape[1] * 8 + 8 + 1 + 1
                mem_size = (
                    (mem_size_per_1data * loop_per_threads * num_threads) // 100000
                ) / 10
                logger.info(
                    f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
                )
                time1, time2 = time2, datetime.datetime.now()
                with ProgressBar(
                    total=how_loop,
                    dynamic_ncols=False,
                    disable=ProgressBar_disable,
                    bar_format=bar_format,
                    leave=False,
                ) as progress:
                    TF_list, similar_num_list = make_unique_equations_thread_7(
                        n_op1,
                        num_threads,
                        random_x,
                        before_similar_num_list_7,
                        saved_similar_num_list,
                        progress,
                    )
                time1, time2 = time2, datetime.datetime.now()
                np.savez(
                    f"cache_folder/saved_7_make_unique_equations_thread_{n_op1}_{num_threads}",
                    TF_list=TF_list,
                )
                logger.info(f"         time : {time2-time1}")
            logger.info(f"      dim_reduction")
            if os.path.isfile(
                f"cache_folder/saved_7_dim_reduction_{n_op1}_{num_threads}.npz"
            ):
                TF_list = np.load(
                    f"cache_folder/saved_7_dim_reduction_{n_op1}_{num_threads}.npz"
                )["TF_list"]
                need_calc_list = TF_list
            else:
                how_loop = np.sum(np.sort(np.sum(TF_list, axis=1))[::-1][1:])
                loop = np.sum(TF_list)
                mem_size_per_1data = random_x.shape[1] * 8 + 8
                mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
                logger.info(
                    f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={loop})"
                )
                time1, time2 = time2, datetime.datetime.now()
                with ProgressBar(
                    total=how_loop,
                    dynamic_ncols=False,
                    disable=ProgressBar_disable,
                    bar_format=bar_format,
                    leave=False,
                ) as progress:
                    TF_list = dim_reduction_7(TF_list, similar_num_list, progress)
                    need_calc_list = TF_list
                time1, time2 = time2, datetime.datetime.now()
                np.savez(
                    f"cache_folder/saved_7_dim_reduction_{n_op1}_{num_threads}",
                    TF_list=TF_list,
                )
                logger.info(f"         time : {time2-time1}")
            logger.info(f"      make_unique_equations_info")
            how_loop = np.sum(TF_list)
            mem_size_per_1data = (
                random_x.shape[1] * 8
                + (2 * max_op + 1)
                + 5 * 8
                + 1
                + tot_mask_x.shape[0]
            )
            mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
            loop = np.sum(TF_list)
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={loop})"
            )
            time1, time2 = time2, datetime.datetime.now()
            with ProgressBar(
                total=how_loop,
                dynamic_ncols=False,
                disable=ProgressBar_disable,
                bar_format=bar_format,
                leave=False,
            ) as progress:
                equation_list, similar_num_list, base_eq_id_list, need_calc_change_x = (
                    make_unique_equations_info_7(n_op1, TF_list, random_x, progress)
                )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")
            saved_equation_list = np.concatenate((saved_equation_list, equation_list))
            saved_similar_num_list = np.concatenate(
                (saved_similar_num_list, similar_num_list)
            )
            saved_base_eq_id_list = np.concatenate(
                (saved_base_eq_id_list, base_eq_id_list)
            )
            saved_need_calc_change_x = np.concatenate(
                (saved_need_calc_change_x, need_calc_change_x)
            )
        saved_need_calc_list = np.ones(saved_equation_list.shape[0], dtype="bool")
        make_final_cache(
            max_op,
            num_threads,
            saved_equation_list,
            saved_need_calc_list,
            saved_base_eq_id_list,
            saved_need_calc_change_x,
        )
    return (
        saved_equation_list,
        saved_need_calc_list,
        saved_base_eq_id_list,
        saved_similar_num_list,
    )


def main(max_op, num_threads, len_x=5, ProgressBar_disable=False, logger=None):
    Nan_number = -100
    # logger
    if logger is None:
        logger = logging.getLogger("SIS")
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            h.close()
        if verbose:
            st_handler = logging.StreamHandler()
            st_handler.setLevel(logging.INFO)
            # _format = "%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s"
            _format = "%(asctime)s %(name)s [%(levelname)s] : %(message)s"
            st_handler.setFormatter(logging.Formatter(_format))
            logger.addHandler(st_handler)
    set_num_threads(num_threads)
    if not thread_check(num_threads):
        logger.info(f"can't set thread : {num_threads}")

    time0 = datetime.datetime.now()
    logger.info(f"max_op = {max_op}  ,  date : {time0}")
    logger.info(f"use cores : {get_num_threads()}")
    # np.random.seed(100)
    logger.info("making random_x")
    save_random_x = np.random.uniform(0.3, 10, (max_op + 2, len_x))
    save_corrcoef = np.max(np.abs(np.corrcoef(save_random_x[1:]) - np.eye(max_op + 1)))
    for i in range(100000):
        random_x = np.random.uniform(0.3, 10, (max_op + 2, len_x))
        corrcoef = np.max(np.abs(np.corrcoef(random_x[1:]) - np.eye(max_op + 1)))
        if save_corrcoef > corrcoef:
            save_corrcoef = corrcoef
            save_random_x = random_x
    random_x = save_random_x.copy()
    random_x[0] = 1
    logger.info(f"random_x corrcoef = {save_corrcoef}")
    base_dict = cache_load(max_op - 1)
    before_equations = np.array([[1], [0]], dtype="int8")
    for i in range(1, max_op):
        equations = base_dict[i]
        base = np.full(
            (before_equations.shape[0] + equations.shape[0], equations.shape[1]),
            Nan_number,
            dtype="int8",
        )
        base[: before_equations.shape[0], : before_equations.shape[1]] = (
            before_equations
        )
        base[before_equations.shape[0] :] = equations
        before_equations = base.copy()
    loop = before_equations.shape[0]
    logger.info("make_before_similar_num_list")
    dict_mask_x = make_dict_mask_x(max_op)
    loop_per_threads = before_equations.shape[0] // num_threads + 1
    mem_size_per_1data = (2 * random_x.shape[1] * 8) * 2
    mem_size = ((num_threads * loop_per_threads * mem_size_per_1data) // 100000) / 10
    loop = before_equations.shape[0]
    logger.info(
        f"   Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={loop})"
    )
    time1, time2 = datetime.datetime.now(), datetime.datetime.now()
    with ProgressBar(
        total=loop,
        dynamic_ncols=False,
        disable=ProgressBar_disable,
        bar_format=bar_format,
        leave=False,
    ) as progress:
        before_similar_num_list = make_before_similar_num_list(
            num_threads, max_op, random_x, before_equations, progress
        )
    time1, time2 = time2, datetime.datetime.now()
    dTime = time2 - time1
    logger.info(f"   time : {dTime}")
    logger.info(f"make_unique_equations")
    time1, time2 = time2, datetime.datetime.now()
    equations, need_calc, base_eq_id, similar_num = make_unique_equations(
        max_op,
        num_threads,
        random_x,
        before_similar_num_list,
        bar_format,
        True,
        ProgressBar_disable,
    )
    time1, time2 = time2, datetime.datetime.now()
    dTime = time2 - time1
    logger.info(f"   time : {dTime}")
    logger.info(f"END")
    Tcount = np.sum(need_calc)
    Fcount = np.sum(~need_calc)
    logger.info(f"need calc  {Tcount}:{Fcount}")
    dTime = time2 - time0
    logger.info(f"total time : {dTime}")
    return equations, need_calc, base_eq_id, similar_num


@njit(error_model="numpy")
def decryption(columns, equation):
    if len(columns) == 0:
        columns = ["1", "a", "b", "c", "d", "e", "f", "g", "h"]
    stack = [str(columns[equation[0]])]
    for i in equation[1:]:
        match i:
            case t if t >= 0:  # number
                stack.append(str(columns[i]))
            case -1:  # +
                b, a = stack.pop(), stack.pop()
                stack.append(("(" + a + "+" + b + ")"))
            case -2:  # -
                b, a = stack.pop(), stack.pop()
                stack.append(("(" + a + "-" + b + ")"))
            case -3:  # *
                b, a = stack.pop(), stack.pop()
                stack.append(("(" + a + "*" + b + ")"))
            case -4:  # /
                b, a = stack.pop(), stack.pop()
                txt = "(" + a + "/" + b + ")"
                stack.append(txt)
    return stack[0]
