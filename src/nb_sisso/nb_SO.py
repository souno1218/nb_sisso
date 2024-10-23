#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numba
from scipy.special import comb
from .log_progress import loop_log
from numba_progress import ProgressBar
import datetime, logging, multiprocessing
from numba import njit, prange, set_num_threads, objmode
from .utils import (
    thread_check,
    argmin_and_min,
    raise_and_log,
    dtype_shape_check,
    type_check,
)


def SO(
    list_x,
    y,
    model_score,
    which_arr_to_choose_from,
    combination_dim=2,
    num_threads=None,
    how_many_to_save=50,
    verbose=True,
    is_progress=False,
    log_interval=10,
    logger=None,
):
    """
    Select all combinations from the given ndarray(list_x), throw them into the model_score,
    and save the how_many_to_save pieces in the order of increasing score.

    Parameters
    ----------
    list_x : list of ndarray of shape (n_saved_in_SIS,n_samples)
        Training data.
        Even if you choose more than one (e.g., two) from a single array, please list.
        which_arr_to_choose_from sets which elements of the list are combined.
    y : ndarray of shape (n_samples)
        Target values or labels.
    model_score : callable
        func that returns a score,jit compilation by numba is required.
        Parameters
            - x : ndarray of shape (combination_dim,n_samples)
                Combination created from List_x.
            - y : ndarray of shape (n_samples)
                Target values or labels.
        Returns
            - score1 : float
                Priority is score1 > score2.
                Values for which greater is better, such as negative mean-square error or accuracies.
            - score2 : float
                Compare score2 if score1 is the same.
                Values for which greater is better, such as negative mean-square error or accuracies.
    which_arr_to_choose_from : dict
        Decide here how many to choose.
        keys set all combination_dim from 1({1:0,~~combination_dim:0).
        Key starts at 1.
        Values specify the index of list_x.
        ex) combination_dim=3 , {1:1,2:2,3:3}
    combination_dim : int, optional
        Number of equations to combine, by default 2.
    num_threads : int, optional
        Number of CPU cores used. If not set, all cpu cores are used.
    how_many_to_save : int, optional
        Specify how many combinations to save up to the top, by default 50.
    verbose : bool, optional
        Print log or, by default True.
        This will be ignored if logger is set.
    is_progress : bool, optional
        Use progress bar or, by default False.
    log_interval : int or float, optional
        Time interval for writing out progress, by default 10 seconds.
        This will be ignored if is_progress is True.
    logger : logging.Logger, optional
        A logger instance to handle logging.
        It is expected to be a standard Python `logging.Logger` instance.
        If not set, create a logger with only a StreamHandler.

    Returns
    ----------
    score_list : ndarray of shape (how_many_to_save)
        sorted score
    index_list : ndarray of shape (how_many_to_save,combination_dim)
        index in list_x corresponding to score_list
    """
    Nan_number = -100

    # logger
    if logger is None:
        logger = logging.getLogger("SO")
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

    # num_threads
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    set_num_threads(num_threads)
    if not thread_check(num_threads):
        logger.info(f"can't set thread : {num_threads}")

    # x
    type_check(logger, list_x, "list_x", list)
    n_features = list_x[0].shape[1]
    for i in range(len(list_x)):
        dtype_shape_check(
            logger,
            list_x[i],
            f"list_x[{i}]",
            dtype_=np.float64,
            ndim=2,
            dict_index_len={1: n_features},
        )

    max_x_len = np.max([i.shape[0] for i in list_x])
    arr_x = np.full((len(list_x), max_x_len, list_x[0].shape[1]), Nan_number, dtype="float64")
    for i in range(len(list_x)):
        arr_x[i, : list_x[i].shape[0]] = list_x[i]

    # y
    dtype_shape_check(logger, y, "y", ndim=1, dict_index_len={0: arr_x.shape[2]})

    # shuffle
    index = np.arange(arr_x.shape[2])
    rng = np.random.default_rng()
    rng.shuffle(index)
    arr_x = arr_x[:, :, index].copy()
    y = y[index].copy()

    # which_arr_to_choose_from
    type_check_which_arr_to_choose_from(logger, combination_dim, which_arr_to_choose_from, list_x)
    arr_which_arr_to_choose_from = np.empty(combination_dim, dtype="int64")
    for i in range(combination_dim):
        arr_which_arr_to_choose_from[i] = which_arr_to_choose_from[i + 1]

    # log
    logger.info(f"numba={numba.__version__}, numpy={np.__version__}")
    logger.info(f"OPT={numba.config.OPT}, THREADING_LAYER={numba.config.THREADING_LAYER}")
    logger.info(
        f"USING_SVML={numba.config.USING_SVML}, ENABLE_AVX={numba.config.ENABLE_AVX}, DISABLE_JIT={numba.config.DISABLE_JIT}"
    )
    logger.info("SO")
    logger.info(f"num_threads={num_threads}, how_many_to_save={how_many_to_save}, ")
    logger.info(f"combination_dim={combination_dim}, model_score={model_score.__name__}, ")
    logger.info(f"which_arr_to_choose_from={which_arr_to_choose_from}")
    repeat = loop_counter(arr_x, arr_which_arr_to_choose_from)
    logger.info(f"loop={repeat}")

    # compiling
    logger.info(f"compiling")
    compiling(num_threads, arr_which_arr_to_choose_from, arr_x, y, model_score, is_progress, logger)
    logger.info(f"END, compile")

    time0 = datetime.datetime.now()
    if is_progress:
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        with ProgressBar(total=repeat, dynamic_ncols=False, bar_format=bar_format, leave=False) as progress:
            score_list, index_list = SO_loop(
                num_threads,
                arr_x,
                y,
                how_many_to_save,
                model_score,
                arr_which_arr_to_choose_from,
                progress,
            )
    else:
        with loop_log(logger, interval=log_interval, tot_loop=repeat, header="  ") as progress:
            score_list, index_list = SO_loop(
                num_threads,
                arr_x,
                y,
                how_many_to_save,
                model_score,
                arr_which_arr_to_choose_from,
                progress,
            )
    dtime = datetime.datetime.now() - time0
    logger.info(f"  END : time={dtime}")
    logger.info(f"best : score={score_list[0]},index={index_list[0]}")
    return score_list, index_list


def type_check_which_arr_to_choose_from(logger, combination_dim, which_arr_to_choose_from, list_x):
    type_check(logger, which_arr_to_choose_from, "which_arr_to_choose_from", dict)
    keys = list(which_arr_to_choose_from.keys())
    for i in keys:
        type_check(logger, i, "which_arr_to_choose_from.keys", int)
    Expected_keys = list(range(1, combination_dim + 1))
    if sorted(keys) != Expected_keys:
        raise_and_log(
            logger, ValueError(f"Expected which_arr_to_choose_from.keys to be {Expected_keys}, but got {keys}.")
        )
    values = list(which_arr_to_choose_from.values())
    for i in values:
        type_check(logger, i, "which_arr_to_choose_from.values", int)
    if max(values) > len(list_x) - 1:
        raise_and_log(logger, ValueError("which_arr_to_choose_from.values must be less than or equal to len(list_x)-1"))


def compiling(num_threads, arr_which_arr_to_choose_from, arr_x, y, model_score, is_progress, logger):
    if is_progress:
        with ProgressBar(total=0, leave=False, disable=True) as p:
            _ = SO_loop(num_threads, arr_x[:, :5], y, 1, model_score, arr_which_arr_to_choose_from, p)
    else:
        with loop_log(logger, interval=10000, tot_loop=0, header="") as p:
            _ = SO_loop(num_threads, arr_x[:, :5], y, 1, model_score, arr_which_arr_to_choose_from, p)


def loop_counter(arr_x, arr_which_arr_to_choose_from):
    unique_arr = np.unique(arr_which_arr_to_choose_from)
    num = 1
    for i in unique_arr:
        same = np.sum(arr_which_arr_to_choose_from == i)
        num *= comb(arr_x[i].shape[0], same, exact=True)
    return num


@njit(error_model="numpy")
def make_check_list(arr_which_arr_to_choose_from):
    return_arr = np.full((arr_which_arr_to_choose_from.shape[0], 2), 10000, dtype="int64")
    unique_arr = np.unique(arr_which_arr_to_choose_from)
    arange = np.arange(arr_which_arr_to_choose_from.shape[0])
    n = 0
    for j in unique_arr:
        same_index = arange[arr_which_arr_to_choose_from == j]
        if same_index.shape[0] > 1:
            sorted_index = np.sort(same_index)
            for i in range(sorted_index.shape[0] - 1):
                return_arr[n] = sorted_index[i : i + 2]
                n += 1
    index = np.argsort(return_arr[:, 1])
    return return_arr[index]


@njit(error_model="numpy")
def make_index_arr(number, check_list, len_x_arr, arr_which_arr_to_choose_from, index_arr):
    n = 0
    for i in range(arr_which_arr_to_choose_from.shape[0]):
        index_arr[i] = number % len_x_arr[i]
        if check_list[n, 1] == i:
            if index_arr[check_list[n, 0]] <= index_arr[i]:
                return False
            else:
                n += 1
        number //= len_x_arr[i]
    return True


@njit(parallel=True, error_model="numpy")  # ,fastmath=True
def SO_loop(
    num_threads,
    arr_x,
    y,
    how_many_to_save,
    model_score,
    arr_which_arr_to_choose_from,
    progress_proxy,
):
    Nan_number = -100
    how_many_to_choose = arr_which_arr_to_choose_from.shape[0]
    len_x_arr = np.array([np.sum(arr_x[i, :, 0] != Nan_number) for i in arr_which_arr_to_choose_from])
    repeat = np.prod(len_x_arr)

    score_list_thread = np.full((num_threads, how_many_to_save, 2), np.finfo(np.float64).min, dtype="float64")
    index_list_thread = np.full((num_threads, how_many_to_save, how_many_to_choose), Nan_number, dtype="int64")
    for thread_id in prange(num_threads):
        score_list = np.full((how_many_to_save, 2), np.finfo(np.float64).min, dtype="float64")
        index_list = np.full((how_many_to_save, how_many_to_choose), Nan_number, dtype="int64")
        border1, border2 = np.finfo(np.float64).min, np.finfo(np.float64).min
        min_index = 0
        index_arr = np.zeros((how_many_to_choose), dtype="int64")
        selected_X = np.empty((how_many_to_choose, arr_x.shape[2]), dtype="float64")
        check_list = make_check_list(arr_which_arr_to_choose_from)
        for i in range(thread_id, repeat, num_threads):
            is_calc = make_index_arr(i, check_list, len_x_arr, arr_which_arr_to_choose_from, index_arr)
            if not is_calc:
                continue
            for j, k in enumerate(index_arr):
                selected_X[j] = arr_x[arr_which_arr_to_choose_from[j], k]
            score1, score2 = model_score(selected_X, y)
            if np.logical_not(np.isnan(score1)):
                if score1 > border1:
                    score_list[min_index, 0] = score1
                    score_list[min_index, 1] = score2
                    index_list[min_index] = index_arr
                    min_num1, min_num2, min_index = argmin_and_min(score_list)
                    if border1 > min_num1:
                        border1 = min_num1
                        border2 = min_num2
                    elif border1 == min_num1:
                        if border2 > min_num2:
                            border2 = min_num2
                elif score1 == border1:
                    if score2 > border2:
                        score_list[min_index, 0] = score1
                        score_list[min_index, 1] = score2
                        index_list[min_index] = index_arr
                        min_num1, min_num2, min_index = argmin_and_min(score_list)
                        if border1 > min_num1:
                            border1 = min_num1
                            border2 = min_num2
                        elif border1 == min_num1:
                            if border2 > min_num2:
                                border2 = min_num2
            progress_proxy.update(1)
        score_list_thread[thread_id] = score_list
        index_list_thread[thread_id] = index_list

    with objmode(index="int64[:]"):
        index = np.lexsort((score_list_thread[:, :, 1].ravel(), score_list_thread[:, :, 0].ravel()))[::-1][
            :how_many_to_save
        ]
    return (
        score_list_thread.reshape(-1, 2)[index],
        index_list_thread.reshape(-1, how_many_to_choose)[index],
    )
