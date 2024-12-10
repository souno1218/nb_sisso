import numpy as np
import pandas as pd
import sys
import os, psutil, logging, datetime
from numba_progress import ProgressBar
from numba import njit, prange, set_num_threads, objmode, get_num_threads
from numba_progress.numba_atomic import atomic_add, atomic_xchg, atomic_min

from .utils_for_make_cache import *


def main(max_op, num_threads, len_x=5, ProgressBar_disable=False, verbose=True, logger=None):
    int_nan = -100
    # logger
    if logger is None:
        logger = logging.getLogger("make_cache")
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
            int_nan,
            dtype="int8",
        )
        base[: before_equations.shape[0], : before_equations.shape[1]] = before_equations
        base[before_equations.shape[0] :] = equations
        before_equations = base.copy()
    loop = before_equations.shape[0]
    logger.info("make_before_similar_num_list")
    loop_per_threads = before_equations.shape[0] // num_threads + 1
    mem_size_per_1data = (2 * random_x.shape[1] * 8) * 2
    mem_size = ((num_threads * loop_per_threads * mem_size_per_1data) // 100000) / 10
    loop = before_equations.shape[0]
    logger.info(
        f"   Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={loop})"
    )
    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
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
        logger,
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


def make_final_cache(
    max_ops,
    num_threads,
    equation_list,
    need_calc_list,
    base_eq_id_list,
    need_calc_change_x,
    logger,
):
    # base_eq_id_list = merge_op,n_op1,id1,id2,changed_back_eq_id
    int_nan = -100
    base_dict = cache_load(max_ops - 1)
    return_cache = dict()
    return_dict_num_to_index = dict()
    return_np_index_to_num = dict()
    max_id_need_calc = np.load("max_id_need_calc.npy")
    base = np.zeros(max(max_id_need_calc.shape[0], max_ops + 1), dtype="int64")
    base[: max_id_need_calc.shape[0]] = max_id_need_calc
    base[max_ops] = np.sum(need_calc_list)
    logger.info(f"   max_id_need_calc, {list(max_id_need_calc)} -> {list(base)}")
    np.save("max_id_need_calc", base)
    for n_op1 in range(max_ops - 1, -1, -1):
        n_op2 = max_ops - 1 - n_op1
        unique_num = np.empty((0), dtype="int64")
        for i in (1, n_op2 + 2):
            all_pattern = nb_permutations(np.arange(1, max_ops + 2), i)
            all_num = np.array([make_change_x_id(all_pattern[i], max_ops + 1) for i in range(all_pattern.shape[0])])
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
                int_nan,
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
                int_nan,
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
        f"cache_{max_ops}",
        **{str(n_op1): return_cache[n_op1] for n_op1 in range(max_ops)},
    )
    np.savez(
        f"num_to_index_{max_ops}",
        **{str(n_op1): return_np_index_to_num[n_op1] for n_op1 in range(max_ops)},
    )
    np.savez(f"need_calc_change_x_{max_ops}", need_calc_change_x)
    # savez_compressed
    np.save(f"operator_{max_ops}", equation_list)


def make_unique_equations(
    max_op,
    num_threads,
    random_x,
    before_similar_num_list,
    logger,
    bar_format,
    is_print,
    ProgressBar_disable,
):
    num_threads = int(num_threads)
    int_nan = -100
    saved_equation_list = np.empty((0, 2 * max_op + 1), dtype="int8")
    saved_base_eq_id_list = np.empty((0, 5), dtype="int8")
    tot_mask_x = nb_permutations(np.arange(max_op + 1), max_op + 1)
    saved_need_calc_change_x = np.empty((0, tot_mask_x.shape[0]), dtype="bool")
    if max_op != 7:
        saved_need_calc_list = np.empty((0), dtype="bool")
        saved_similar_num_list = np.empty((2, 0, random_x.shape[1]), dtype="float64")
        time1, time2 = datetime.datetime.now(), datetime.datetime.now()
        for n_op1 in range(max_op - 1, -1, -1):
            n_op2 = max_op - 1 - n_op1
            logger.info(f"   n_op1={n_op1},n_op2={n_op2}")
            how_loop, loop_per_threads = loop_count(max_op, n_op1, num_threads, is_print=is_print)
            logger.info(f"      make_unique_equations_thread")
            mem_size_per_1data = 2 * random_x.shape[1] * 8 + 8 + 1 + 1
            mem_size = ((mem_size_per_1data * loop_per_threads * num_threads) // 100000) / 10
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
                TF_list, similar_num_list, need_calc_list = make_unique_equations_thread(
                    max_op,
                    n_op1,
                    num_threads,
                    random_x,
                    before_similar_num_list,
                    saved_similar_num_list,
                    progress,
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
                TF_list, need_calc_list = dim_reduction(TF_list, similar_num_list, need_calc_list, progress)
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")
            logger.info(f"      make_unique_equations_info")
            how_loop = np.sum(TF_list)
            mem_size_per_1data = 2 * random_x.shape[1] * 8 + (2 * max_op + 1) + 5 * 8 + 1 + tot_mask_x.shape[0]
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
                ) = make_unique_equations_info(max_op, n_op1, TF_list, need_calc_list, random_x, progress)
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")
            saved_equation_list = np.concatenate((saved_equation_list, equation_list))
            saved_similar_num_list = np.concatenate((saved_similar_num_list, similar_num_list), axis=1)
            saved_need_calc_list = np.concatenate((saved_need_calc_list, need_calc_list))
            saved_base_eq_id_list = np.concatenate((saved_base_eq_id_list, base_eq_id_list))
            saved_need_calc_change_x = np.concatenate((saved_need_calc_change_x, need_calc_change_x))
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
            logger,
        )
    else:  # max_op==7
        saved_similar_num_list = np.empty((0, random_x.shape[1]), dtype="float64")
        before_similar_num_list_7 = before_similar_num_list[0].copy()
        time1, time2 = datetime.datetime.now(), datetime.datetime.now()
        for n_op1 in range(max_op - 1, -1, -1):
            n_op2 = max_op - 1 - n_op1
            logger.info(f"   n_op1={n_op1},n_op2={n_op2}")
            how_loop, loop_per_threads = loop_count(max_op, n_op1, num_threads, is_print=is_print)
            logger.info(f"      make_unique_equations_thread")
            if os.path.isfile(f"saved_7_make_unique_equations_thread_{n_op1}_{num_threads}.npz"):
                TF_list = np.load(f"saved_7_make_unique_equations_thread_{n_op1}_{num_threads}.npz")["TF_list"]
                how_loop = np.sum(TF_list)
                logger.info(f"         make_similar_num_by_saved_7")
                with ProgressBar(
                    total=how_loop,
                    dynamic_ncols=False,
                    disable=ProgressBar_disable,
                    bar_format=bar_format,
                    leave=False,
                ) as progress:
                    similar_num_list = make_similar_num_by_saved_7(n_op1, TF_list, random_x, progress)
            else:
                mem_size_per_1data = random_x.shape[1] * 8 + 8 + 1 + 1
                mem_size = ((mem_size_per_1data * loop_per_threads * num_threads) // 100000) / 10
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
                    f"saved_7_make_unique_equations_thread_{n_op1}_{num_threads}",
                    TF_list=TF_list,
                )
                logger.info(f"         time : {time2-time1}")
            logger.info(f"      dim_reduction")
            if os.path.isfile(f"saved_7_dim_reduction_{n_op1}_{num_threads}.npz"):
                TF_list = np.load(f"saved_7_dim_reduction_{n_op1}_{num_threads}.npz")["TF_list"]
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
                    f"saved_7_dim_reduction_{n_op1}_{num_threads}",
                    TF_list=TF_list,
                )
                logger.info(f"         time : {time2-time1}")
            logger.info(f"      make_unique_equations_info")
            how_loop = np.sum(TF_list)
            mem_size_per_1data = random_x.shape[1] * 8 + (2 * max_op + 1) + 5 * 8 + 1 + tot_mask_x.shape[0]
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
                equation_list, similar_num_list, base_eq_id_list, need_calc_change_x = make_unique_equations_info_7(
                    n_op1, TF_list, random_x, progress
                )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")
            saved_equation_list = np.concatenate((saved_equation_list, equation_list))
            saved_similar_num_list = np.concatenate((saved_similar_num_list, similar_num_list))
            saved_base_eq_id_list = np.concatenate((saved_base_eq_id_list, base_eq_id_list))
            saved_need_calc_change_x = np.concatenate((saved_need_calc_change_x, need_calc_change_x))
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
    int_nan = -100
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
    tot_mask_x = nb_permutations(np.arange(max_op + 1), max_op + 1)
    _, loop_per_threads = loop_count(max_op, n_op1, num_threads)
    head_saved_similar_num_list = saved_similar_num_list[0, :, 0].copy()
    save_similar_num_list = np.full(
        (num_threads, 2, loop_per_threads, random_x.shape[1]),
        int_nan,
        dtype="float64",
    )
    head_save_similar_num_list = np.full((num_threads, loop_per_threads), int_nan, dtype="float64")
    TF_list = np.zeros((num_threads, loop_per_threads), dtype="bool")
    save_need_calc_list = np.zeros((num_threads, loop_per_threads), dtype="bool")
    if n_op1 >= n_op2:
        ops = [-1, -2, -3, -4]
    else:
        ops = [-4]
    with objmode():
        using_memory = psutil.Process().memory_info().rss / 1024**2
        print(f"         using memory : {using_memory} M bytes")
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), int_nan, dtype="int8")
        cache_for_mask_x = np.full((tot_mask_x.shape[0], 2, random_x.shape[1]), int_nan, dtype="float64")
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
                            if find_min_x_max(equation, random_x, random_for_find_min_x_max):
                                is_save, is_calc = False, False
                        if is_save:
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
                            if k == 0:
                                if similar_num[0, 0] == int_nan:
                                    if np.sum(equation > 0) != 0:
                                        is_save, is_calc = False, False
                                    elif np.all(similar_num[1] == 0):
                                        is_save, is_calc = False, False
                                    else:
                                        is_calc = False
                            elif arg_close_arr(similar_num[1], cache_for_mask_x[:n, 1]) != int_nan:
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
                            same_head_index = arg_close_arr(
                                cache_for_mask_x[normalized_min_index, 0, 0],
                                head_before_similar_num_list,
                            )
                            for t in same_head_index:
                                if isclose_arr(
                                    cache_for_mask_x[normalized_min_index, 0],
                                    before_similar_num_list[0, t],
                                ):
                                    is_calc = False
                                    if isclose_arr(
                                        cache_for_mask_x[min_index, 1],
                                        before_similar_num_list[1, t],
                                    ):
                                        is_save = False
                                        break
                    if is_save:
                        same_head_index = arg_close_arr(
                            cache_for_mask_x[normalized_min_index, 0, 0],
                            head_saved_similar_num_list,
                        )
                        for t in same_head_index:
                            if isclose_arr(
                                cache_for_mask_x[normalized_min_index, 0],
                                saved_similar_num_list[0, t],
                            ):
                                is_calc = False
                                if isclose_arr(
                                    cache_for_mask_x[min_index, 1],
                                    saved_similar_num_list[1, t],
                                ):
                                    is_save = False
                                    break
                    if is_save:
                        same_head_index = arg_close_arr(
                            cache_for_mask_x[normalized_min_index, 0, 0],
                            head_save_similar_num_list[thread_id, :counter],
                        )
                        for t in same_head_index:
                            if isclose_arr(
                                cache_for_mask_x[normalized_min_index, 0],
                                save_similar_num_list[thread_id, 0, t],
                            ):
                                is_calc = False
                                if isclose_arr(
                                    cache_for_mask_x[min_index, 1],
                                    save_similar_num_list[thread_id, 1, t],
                                ):
                                    is_save = False
                                    break
                    if is_save:
                        TF_list[thread_id, counter] = True
                        save_similar_num_list[thread_id, 0, counter] = cache_for_mask_x[normalized_min_index, 0]
                        save_similar_num_list[thread_id, 1, counter] = cache_for_mask_x[min_index, 1]
                        head_save_similar_num_list[thread_id, counter] = cache_for_mask_x[normalized_min_index, 0, 0]
                        save_need_calc_list[thread_id, counter] = is_calc
                    counter += 1
                    progress_proxy.update(1)
    return TF_list, save_similar_num_list, save_need_calc_list


@njit(parallel=True, error_model="numpy")
def dim_reduction(TF_list, similar_num_list, need_calc_list, progress_proxy):
    int_nan = -100
    num_threads = similar_num_list.shape[0]
    threads_last_index = np.array([np.sum(TF_list[i]) for i in range(num_threads)])
    sort_index = np.argsort(threads_last_index)[::-1]
    return_similar_num_list = np.full((2, np.sum(TF_list), similar_num_list.shape[3]), int_nan, dtype="float64")
    return_similar_num_list[0, : np.sum(TF_list[sort_index[0]])] = similar_num_list[
        sort_index[0], 0, TF_list[sort_index[0]]
    ]
    return_similar_num_list[1, : np.sum(TF_list[sort_index[0]])] = similar_num_list[
        sort_index[0], 1, TF_list[sort_index[0]]
    ]
    head_return_similar_num_list = np.full((np.sum(TF_list)), int_nan, dtype="float64")
    head_similar_num_list = similar_num_list[:, 0, :, 0]
    head_return_similar_num_list[: np.sum(TF_list[sort_index[0]])] = head_similar_num_list[
        sort_index[0], TF_list[sort_index[0]]
    ]
    last_index = np.sum(TF_list[sort_index[0]])
    with objmode():
        using_memory = psutil.Process().memory_info().rss / 1024**2
        print(f"         using memory : {using_memory} M bytes")
    for target_index in sort_index[1:]:
        true_index = np.arange(TF_list.shape[1])[TF_list[target_index]]
        for thread_id in prange(num_threads):
            true_index_thread = true_index[thread_id::num_threads].copy()
            for i in true_index_thread:
                for t in arg_close_arr(
                    similar_num_list[target_index, 0, i, 0],
                    head_return_similar_num_list[:last_index],
                ):
                    if isclose_arr(
                        similar_num_list[target_index, 0, i],
                        return_similar_num_list[0, t],
                    ):
                        need_calc_list[target_index, i] = False
                        if isclose_arr(
                            similar_num_list[target_index, 1, i],
                            return_similar_num_list[1, t],
                        ):
                            TF_list[target_index, i] = False
                            break
                progress_proxy.update(1)
        return_similar_num_list[0, last_index : last_index + np.sum(TF_list[target_index])] = similar_num_list[
            target_index, 0, TF_list[target_index]
        ]
        return_similar_num_list[1, last_index : last_index + np.sum(TF_list[target_index])] = similar_num_list[
            target_index, 1, TF_list[target_index]
        ]
        head_return_similar_num_list[last_index : last_index + np.sum(TF_list[target_index])] = head_similar_num_list[
            target_index, TF_list[target_index]
        ]
        last_index += np.sum(TF_list[target_index])
    return TF_list, need_calc_list


@njit(parallel=True, error_model="numpy")
def make_unique_equations_info(max_op, n_op1, TF_list, need_calc_list, random_x, progress_proxy):
    int_nan = -100
    n_op2 = max_op - 1 - n_op1
    base_dict = cache_load(max_op - 1)
    tot_mask_x = nb_permutations(np.arange(max_op + 1), max_op + 1)
    if n_op1 >= n_op2:
        ops = [-1, -2, -3, -4]
    else:
        ops = [-4]
    sum_TF = np.sum(TF_list)
    num_threads = TF_list.shape[0]
    return_similar_num_list = np.full((2, sum_TF, random_x.shape[1]), int_nan, dtype="float64")
    return_equation_list = np.full((sum_TF, 2 * max_op + 1), int_nan, dtype="int8")
    return_base_eq_id_list = np.full((sum_TF, 5), int_nan, dtype="int64")
    return_need_calc_list = np.zeros((sum_TF), dtype="bool")
    return_need_calc_change_x = np.zeros((sum_TF, tot_mask_x.shape[0]), dtype="bool")
    with objmode():
        using_memory = psutil.Process().memory_info().rss / 1024**2
        print(f"         using memory : {using_memory} M bytes")
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), int_nan, dtype="int8")
        cache_for_mask_x = np.full((tot_mask_x.shape[0], 2, random_x.shape[1]), int_nan, dtype="float64")
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
                            if arg_close_arr(similar_num[1], cache_for_mask_x[:n, 1]) != int_nan:
                                thread_need_calc_change_x[k] = False
                                # neeeeeeeed
                                continue
                            else:
                                cache_for_mask_x[n] = similar_num
                                n += 1
                        normalized_min_index = np.argmin(cache_for_mask_x[:n, 0, 0])
                        min_index = np.argmin(cache_for_mask_x[:n, 1, 0])
                        return_similar_num_list[0, last_index] = cache_for_mask_x[normalized_min_index, 0]
                        return_similar_num_list[1, last_index] = cache_for_mask_x[min_index, 1]
                        return_equation_list[last_index] = equation
                        return_need_calc_list[last_index] = need_calc_list[thread_id, counter]
                        return_base_eq_id_list[last_index, 0] = merge_op
                        return_base_eq_id_list[last_index, 1] = n_op1
                        return_base_eq_id_list[last_index, 2] = id1
                        return_base_eq_id_list[last_index, 3] = id2
                        changed_back_eq_id = make_change_x_id(
                            dict_change_x_pattern[base_back_x_max][number], max_op + 1
                        )
                        return_base_eq_id_list[last_index, 4] = changed_back_eq_id
                        return_need_calc_change_x[last_index, : mask_x.shape[0]] = thread_need_calc_change_x[
                            : mask_x.shape[0]
                        ]
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
    int_nan = -100
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
    tot_mask_x = nb_permutations(np.arange(max_op + 1), max_op + 1)
    _, loop_per_threads = loop_count(max_op, n_op1, num_threads)
    head_saved_similar_num_list = saved_similar_num_list[:, 0].copy()
    save_similar_num_list = np.full((num_threads, loop_per_threads, random_x.shape[1]), int_nan, dtype="float64")
    head_save_similar_num_list = np.full((num_threads, loop_per_threads), int_nan, dtype="float64")
    TF_list = np.zeros((num_threads, loop_per_threads), dtype="bool")
    if n_op1 >= n_op2:
        ops = [-1, -2, -3, -4]
    else:
        ops = [-4]
    with objmode():
        using_memory = psutil.Process().memory_info().rss / 1024**2
        print(f"         using memory : {using_memory} M bytes")
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), int_nan, dtype="int8")
        cache_for_mask_x = np.full((tot_mask_x.shape[0], 2, random_x.shape[1]), int_nan, dtype="float64")
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
                            if find_min_x_max(equation, random_x, random_for_find_min_x_max):
                                is_save = False
                        if is_save:
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
                            if k == 0:
                                if similar_num[0, 0] == int_nan:
                                    is_save = False
                            elif arg_close_arr(similar_num[0], cache_for_mask_x[:n, 0]) != int_nan:
                                continue
                            if is_save:
                                cache_for_mask_x[n] = similar_num
                                n += 1
                            else:  # is_save=False
                                break
                    if is_save:
                        normalized_min_index = np.argmin(cache_for_mask_x[:n, 0, 0])
                        if max_op + 1 != eq_x_max:
                            same_head_index = arg_close_arr(
                                cache_for_mask_x[normalized_min_index, 0, 0],
                                head_before_similar_num_list,
                            )
                            for t in same_head_index:
                                if isclose_arr(
                                    cache_for_mask_x[normalized_min_index, 0],
                                    before_similar_num_list[t],
                                ):
                                    is_save = False
                                    break
                    if is_save:
                        same_head_index = arg_close_arr(
                            cache_for_mask_x[normalized_min_index, 0, 0],
                            head_saved_similar_num_list,
                        )
                        for t in same_head_index:
                            if isclose_arr(
                                cache_for_mask_x[normalized_min_index, 0],
                                saved_similar_num_list[t],
                            ):
                                is_save = False
                                break
                    if is_save:
                        same_head_index = arg_close_arr(
                            cache_for_mask_x[normalized_min_index, 0, 0],
                            head_save_similar_num_list[thread_id, :counter],
                        )
                        for t in same_head_index:
                            if isclose_arr(
                                cache_for_mask_x[normalized_min_index, 0],
                                save_similar_num_list[thread_id, t],
                            ):
                                is_save = False
                                break
                    if is_save:
                        TF_list[thread_id, counter] = True
                        save_similar_num_list[thread_id, counter] = cache_for_mask_x[normalized_min_index, 0]
                        head_save_similar_num_list[thread_id, counter] = cache_for_mask_x[normalized_min_index, 0, 0]
                    counter += 1
                    progress_proxy.update(1)
    return TF_list, save_similar_num_list


@njit(parallel=True, error_model="numpy")
def dim_reduction_7(TF_list, similar_num_list, progress_proxy):
    int_nan = -100
    num_threads = similar_num_list.shape[0]
    threads_last_index = np.array([np.sum(TF_list[i]) for i in range(num_threads)])
    sort_index = np.argsort(threads_last_index)[::-1]
    return_similar_num_list = np.full((np.sum(TF_list), similar_num_list.shape[2]), int_nan, dtype="float64")
    return_similar_num_list[: np.sum(TF_list[sort_index[0]])] = similar_num_list[sort_index[0], TF_list[sort_index[0]]]
    head_return_similar_num_list = np.full((np.sum(TF_list)), int_nan, dtype="float64")
    head_similar_num_list = similar_num_list[:, :, 0]
    head_return_similar_num_list[: np.sum(TF_list[sort_index[0]])] = head_similar_num_list[
        sort_index[0], TF_list[sort_index[0]]
    ]
    last_index = np.sum(TF_list[sort_index[0]])
    with objmode():
        using_memory = psutil.Process().memory_info().rss / 1024**2
        print(f"         using memory : {using_memory} M bytes")
    for target_index in sort_index[1:]:
        true_index = np.arange(TF_list.shape[1])[TF_list[target_index]]
        for thread_id in prange(num_threads):
            true_index_thread = true_index[thread_id::num_threads].copy()
            for i in true_index_thread:
                for t in arg_close_arr(
                    similar_num_list[target_index, i, 0],
                    head_return_similar_num_list[:last_index],
                ):
                    if isclose_arr(similar_num_list[target_index, i], return_similar_num_list[t]):
                        TF_list[target_index, i] = False
                        break
                progress_proxy.update(1)
        return_similar_num_list[last_index : last_index + np.sum(TF_list[target_index])] = similar_num_list[
            target_index, TF_list[target_index]
        ]
        head_return_similar_num_list[last_index : last_index + np.sum(TF_list[target_index])] = head_similar_num_list[
            target_index, TF_list[target_index]
        ]
        last_index += np.sum(TF_list[target_index])
    return TF_list


@njit(parallel=True, error_model="numpy")
def make_unique_equations_info_7(n_op1, TF_list, random_x, progress_proxy):
    int_nan = -100
    max_op = 7
    n_op2 = max_op - 1 - n_op1
    base_dict = cache_load(max_op - 1)
    tot_mask_x = nb_permutations(np.arange(max_op + 1), max_op + 1)
    if n_op1 >= n_op2:
        ops = [-1, -2, -3, -4]
    else:
        ops = [-4]
    sum_TF = np.sum(TF_list)
    num_threads = TF_list.shape[0]
    return_similar_num_list = np.full((sum_TF, random_x.shape[1]), int_nan, dtype="float64")
    return_equation_list = np.full((sum_TF, 2 * max_op + 1), int_nan, dtype="int8")
    return_base_eq_id_list = np.full((sum_TF, 5), int_nan, dtype="int64")
    return_need_calc_change_x = np.zeros((sum_TF, tot_mask_x.shape[0]), dtype="bool")
    with objmode():
        using_memory = psutil.Process().memory_info().rss / 1024**2
        print(f"         using memory : {using_memory} M bytes")
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), int_nan, dtype="int8")
        cache_for_mask_x = np.full((tot_mask_x.shape[0], 2, random_x.shape[1]), int_nan, dtype="float64")
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
                            if arg_close_arr(similar_num[1], cache_for_mask_x[:n, 1]) != int_nan:
                                thread_need_calc_change_x[k] = False
                                # neeeeeeeed
                                continue
                            else:
                                cache_for_mask_x[n] = similar_num
                                n += 1
                        normalized_min_index = np.argmin(cache_for_mask_x[:n, 0, 0])
                        return_similar_num_list[last_index] = cache_for_mask_x[normalized_min_index, 0]
                        return_equation_list[last_index] = equation
                        return_base_eq_id_list[last_index, 0] = merge_op
                        return_base_eq_id_list[last_index, 1] = n_op1
                        return_base_eq_id_list[last_index, 2] = id1
                        return_base_eq_id_list[last_index, 3] = id2
                        changed_back_eq_id = make_change_x_id(
                            dict_change_x_pattern[base_back_x_max][number], max_op + 1
                        )
                        return_base_eq_id_list[last_index, 4] = changed_back_eq_id
                        return_need_calc_change_x[last_index, : mask_x.shape[0]] = thread_need_calc_change_x[
                            : mask_x.shape[0]
                        ]
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
    int_nan = -100
    max_op = 7
    n_op2 = max_op - 1 - n_op1
    base_dict = cache_load(max_op - 1)
    if n_op1 >= n_op2:
        ops = [-1, -2, -3, -4]
    else:
        ops = [-4]
    num_threads = TF_list.shape[0]
    return_similar_num_list = np.full((num_threads, TF_list.shape[1], random_x.shape[1]), int_nan, dtype="float64")
    with objmode():
        using_memory = psutil.Process().memory_info().rss / 1024**2
        print(f"         using memory : {using_memory} M bytes")
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), int_nan, dtype="int8")
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
                        return_similar_num_list[thread_id, counter] = save_similar_num[0]
                        progress_proxy.update(1)
                    counter += 1
    return return_similar_num_list
