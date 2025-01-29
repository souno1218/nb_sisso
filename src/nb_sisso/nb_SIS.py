#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from .log_progress import loop_log
from numba_progress import ProgressBar
import importlib.resources as pkg_resources
import numba, os, datetime, logging, multiprocessing
from numba import njit, prange, set_num_threads, objmode

from .utils import (
    thread_check,
    calc_RPN,
    argmin_and_min,
    raise_and_log,
    dtype_shape_check,
    type_check,
    p_upper_x,
    count_True,
)


def SIS(
    x,
    y,
    model_score,
    units=None,
    how_many_to_save=50000,
    is_use_1=False,
    max_n_op=5,
    operators_to_use=["+", "-", "*", "/"],
    num_threads=None,
    fast=False,
    fmax_max=1e15,
    fmax_min=1e-15,
    verbose=True,
    is_progress=False,
    log_interval=10,
    logger=None,
):
    """
    Select all combinations from the given initial features x and operators and create new features.
    Throw them into model_score and save the how_many_to_save features in order of increasing score.

    Parameters
    ----------
    x : ndarray of shape (n_features,n_samples)
        Training data.
    y : ndarray of shape (n_samples)
        Target values or labels.
    model_score : callable
        Must be a dedicated function for x ndim=1.
        Func that returns a score,jit compilation by numba is required.
        Parameters
            - x : ndarray of shape (n_samples)
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
    units : ndarray of shape (n_features,n_unit_types), optional
        Information on units.
        Must be int64.
        If the feature x is {0: m/s^2, 1: a.u., 2: m^2}, then set the following
            array( [[ 1, -2],  <- index 0
                    [ 0,  0],  <- index 1
                    [ 2,  0]]) <- index 2
                    [ m,  s]
        If not set, it will be np.zeros((n_features,1),dtype=int64).
    how_many_to_save : int, optional
        Specify how many combinations to save up to the top, by default 50000.
    is_use_1 : bool, optional
        Whether to add np.ones((n_samples),dtype=float64) to the initial features, by default False.
    max_n_op : int, optional
        Maximum operator quantity, by default 5.
    operators_to_use : list, optional
        Operators to be considered, by default ["+", "-", "*", "/"].
        Choose from ["+","-","*","/","**-1","**2","sqrt","| |","**3","cbrt","**6","exp","exp-","log","sin","cos","scd"].
    num_threads : int, optional
        Number of CPU cores used. If not set, all cpu cores are used.
    fast : bool, optional
        if pattern <= 4 or how_many_to_save < 10000,This will be ignored.
        When computing wiHowever, it would be in vain because in many cases good results are not concentrated in one core.
        However, it would be in vain because in many cases good results are not concentrated in one core.
        Eliminate wasteful preservation to the extent that it is safe to do so 99.99999% of the time.
    fmax_min : bool, optional
        The threshold that if the maximal absolute value of the data in a feature is smaller than fmax_min,
        it means the magnitudes of all numbers in the feature are so small that the feature will be treated as zero-feature and be discarded.
    fmax_max : bool, optional
        The threshold that if the maximal absolute value of the data in a feature is greater than fmax_max,
        it means the magnitudes of certain numbers in the feature are so large that the feature will be treated as infinity-feature and be discarded.
    verbose : bool, optional
        Print log or, by default True.
        This will be ignored if logger is set.
    is_progress : bool, optional
        Use progress bar or, by default False.
    log_interval : int, optional
        Time interval for writing out progress, by default 10 seconds.
        This will be ignored if is_progress is True.
    logger : _type_, optional
        A logger instance to handle logging.
        It is expected to be a standard Python `logging.Logger` instance.
        If not set, create a logger with only a StreamHandler.

    Returns
    ----------
    score_list : ndarray of shape (how_many_to_save)
        Sorted scores.
    eq_list : ndarray of shape (how_many_to_save,2*max_n_op+1)
        Encrypted expression corresponding to score_list.
        utils.decryption can be used to decrypt the data.
    """

    int_nan = -100

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

    # check exist cache file
    if max_n_op == 6:
        cache_path = os.fspath(pkg_resources.path("nb_sisso", "cache_folder"))
        exist_cache_6 = os.path.isfile(f"{cache_path}/cache_6.npz")
        exist_check_change_x_ones_6 = os.path.isfile(f"{cache_path}/check_change_x_ones_6.npz")
        if exist_cache_6 and exist_check_change_x_ones_6:
            broken_cache_6, broken_check_change_x_ones_6 = False, False
            try:
                np.load(f"{cache_path}/cache_6.npz")
            except:
                broken_cache_6 = True
            try:
                np.load(f"{cache_path}/check_change_x_ones_6.npz")
            except:
                broken_check_change_x_ones_6 = True
            if broken_cache_6 or broken_check_change_x_ones_6:
                corrupted_files = []
                if broken_cache_6:
                    corrupted_files.append("cache_6.npz")
                if broken_check_change_x_ones_6:
                    corrupted_files.append("check_change_x_ones_6.npz")
                file_list = ", ".join(corrupted_files)
                raise_and_log(logger, ValueError(f"The cache files {file_list} are corrupted. Please regenerate them."))
        else:
            not_founf_files = []
            if not exist_cache_6:
                not_founf_files.append("cache_6.npz")
            if not exist_check_change_x_ones_6:
                not_founf_files.append("check_change_x_ones_6.npz")
            file_list = ", ".join(not_founf_files)
            download_url = "https://drive.google.com/drive/folders/1SKGgM8iNV8-Dbn9UushDy3hPJXV6vEf6?usp=drive_link"
            txt = (
                f"The cache files {file_list} were not found. Due to GitHub file size limitations, \n"
                f"                   cache_6.npz and check_change_x_ones_6.npz cannot be downloaded directly. \n"
                f"                   Please download them from the following URL or use `make_cache.py` to generate them manually. \n"
                f"                   URL: {download_url}\n"
                f"                   Place the files in {cache_path}."
            )
            raise_and_log(logger, FileNotFoundError(txt))

    # num_threads
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    set_num_threads(num_threads)
    if not thread_check(num_threads):
        logger.info(f"can't set thread : {num_threads}")

    # x
    dtype_shape_check(logger, x, "x", dtype_=np.float64, ndim=2)

    # y
    dtype_shape_check(logger, y, "y", ndim=1, dict_index_len={0: x.shape[1]})

    # shuffle
    index = np.arange(x.shape[1])
    rng = np.random.default_rng()  # seed=123)
    rng.shuffle(index)
    x = x.T[index].T.copy()
    y = y[index].copy()

    # units
    if units is None:
        units = np.zeros((x.shape[0], 1), dtype="int64")
    else:
        dtype_shape_check(
            logger,
            units,
            "units",
            dtype_=np.int64,
            ndim=2,
            dict_index_len={0: x.shape[0]},
        )

    if "sqrt" in operators_to_use:
        units *= 2**max_n_op
    if "cbrt" in operators_to_use:
        units *= 3**max_n_op

    dict_op_str_to_num = {
        "+": -1,
        "-": -2,
        "*": -3,
        "/": -4,
        "**-1": -6,
        "**2": -7,
        "sqrt": -8,
        "| |": -9,
        "**3": -10,
        "cbrt": -11,
        "**6": -12,
        "exp": -13,
        "exp-": -14,
        "log": -15,
        "sin": -16,
        "cos": -17,
        "scd": -18,
    }

    type_check(logger, operators_to_use, "operators_to_use", list)
    for i in operators_to_use:
        type_check(logger, i, "operators_to_use[i]", str)
        if not i in dict_op_str_to_num.keys():
            raise_and_log(
                logger,
                ValueError(f"operators_to_use must be chosen from {dict_op_str_to_num.keys()}"),
            )

    # use_binary_op
    use_binary_op = []
    for i in list(dict_op_str_to_num.keys())[:4]:
        if i in operators_to_use:
            use_binary_op += [dict_op_str_to_num[i]]

    # use_unary_op
    use_unary_op = []
    for i in list(dict_op_str_to_num.keys())[4:]:
        if i in operators_to_use:
            use_unary_op += [dict_op_str_to_num[i]]

    # fast
    if fast and (num_threads > 4) and (how_many_to_save >= 10000):
        mu = how_many_to_save // num_threads
        arange = np.linspace(mu, mu * 2, 1000)
        data = np.array([p_upper_x(how_many_to_save, i, num_threads) for i in arange])
        try:
            how_many_per1c = int(arange[data < 1e-7][0]) + 1
        except:
            how_many_per1c = how_many_to_save
    else:
        how_many_per1c = how_many_to_save

    # log
    logger.info(f"numba={numba.__version__}, numpy={np.__version__}")
    logger.info(f"OPT={numba.config.OPT}, THREADING_LAYER={numba.config.THREADING_LAYER}")
    logger.info(
        f"USING_SVML={numba.config.USING_SVML}, ENABLE_AVX={numba.config.ENABLE_AVX}, DISABLE_JIT={numba.config.DISABLE_JIT}"
    )

    logger.info("SIS")
    logger.info(f"num_threads={num_threads}, how_many_to_save={how_many_to_save}, ")
    logger.info(f"how_many_to_save_per_1_core={how_many_per1c}, ")
    logger.info(f"max_n_op={max_n_op}, model_score={model_score.__name__}, ")
    logger.info(f"x.shape={x.shape}, is_use_1={is_use_1}")
    logger.info(f"use_binary_op={use_binary_op}, ")
    logger.info(f"use_unary_op={use_unary_op}")
    str_units = " , ".join([str(units[i]) for i in range(units.shape[0])])
    logger.info(f"units={str_units}")

    # compiling
    logger.info(f"compiling")
    compiling(num_threads, is_use_1, use_binary_op, use_unary_op, x, y, units, model_score, is_progress, logger)
    logger.info(f"END, compile")

    save_score_list = np.full((num_threads, how_many_per1c, 2), np.finfo(np.float64).min, dtype="float64")
    save_eq_list = np.full((num_threads, how_many_per1c, 2 * max_n_op + 1), int_nan, dtype="int8")
    min_index_list = np.zeros(num_threads, dtype="int64")
    border_list = np.full((num_threads, 2), np.finfo(np.float64).min, dtype="float64")

    used_eq_dict, used_unit_dict, used_shape_id_dict, used_info_dict = sub_loop_non_op(
        x,
        y,
        units,
        is_use_1,
        model_score,
        save_score_list,
        save_eq_list,
        min_index_list,
        border_list,
    )

    time0 = datetime.datetime.now()
    for n_op in range(1, max_n_op + 1):
        time1 = datetime.datetime.now()
        logger.info(f"  n_op={n_op}")
        for n_op1 in range(n_op - 1, -1, -1):
            n_op2 = n_op - 1 - n_op1
            loop = loop_counter_binary(use_binary_op, n_op1, n_op2, used_eq_dict)
            logger.info(f"    binary_op n_op1:n_op2 = {n_op1}:{n_op2},  loop:{loop}")
            time2 = datetime.datetime.now()
            if is_progress:
                bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                with ProgressBar(total=loop, dynamic_ncols=False, bar_format=bar_format, leave=False) as progress:
                    sub_loop_binary_op(
                        x,
                        y,
                        model_score,
                        how_many_to_save,
                        fmax_max,
                        fmax_min,
                        n_op,
                        n_op1,
                        use_binary_op,
                        save_score_list,
                        save_eq_list,
                        min_index_list,
                        border_list,
                        used_eq_dict,
                        used_unit_dict,
                        used_shape_id_dict,
                        used_info_dict,
                        progress,
                    )
            else:
                header = "      "
                with loop_log(logger, interval=log_interval, tot_loop=loop, header=header) as progress:
                    sub_loop_binary_op(
                        x,
                        y,
                        model_score,
                        how_many_to_save,
                        fmax_max,
                        fmax_min,
                        n_op,
                        n_op1,
                        use_binary_op,
                        save_score_list,
                        save_eq_list,
                        min_index_list,
                        border_list,
                        used_eq_dict,
                        used_unit_dict,
                        used_shape_id_dict,
                        used_info_dict,
                        progress,
                    )
            logger.info(f"      time : {datetime.datetime.now()-time2}")
        if len(use_unary_op) != 0:
            loop = loop_counter_unary(n_op, used_eq_dict)
            logger.info(f"    unary_op,  loop:{loop}")
            time2 = datetime.datetime.now()
            if is_progress:
                bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                with ProgressBar(total=loop, dynamic_ncols=False, bar_format=bar_format, leave=False) as progress:
                    sub_loop_unary_op(
                        x,
                        y,
                        model_score,
                        how_many_to_save,
                        fmax_max,
                        fmax_min,
                        n_op,
                        use_unary_op,
                        save_score_list,
                        save_eq_list,
                        min_index_list,
                        border_list,
                        used_eq_dict,
                        used_unit_dict,
                        used_shape_id_dict,
                        used_info_dict,
                        progress,
                    )
            else:
                header = "      "
                with loop_log(logger, interval=log_interval, tot_loop=loop, header=header) as progress:
                    sub_loop_unary_op(
                        x,
                        y,
                        model_score,
                        how_many_to_save,
                        fmax_max,
                        fmax_min,
                        n_op,
                        use_unary_op,
                        save_score_list,
                        save_eq_list,
                        min_index_list,
                        border_list,
                        used_eq_dict,
                        used_unit_dict,
                        used_shape_id_dict,
                        used_info_dict,
                        progress,
                    )
            logger.info(f"      time : {datetime.datetime.now()-time2}")
        logger.info(f"    END, time={datetime.datetime.now()-time1}")

    index = np.lexsort((save_score_list[:, :, 1].ravel(), save_score_list[:, :, 0].ravel()))[::-1][:how_many_to_save]
    return_score_list = save_score_list.reshape((-1, 2))[index]
    return_eq_list = save_eq_list.reshape(-1, 2 * max_n_op + 1)[index]
    logger.info(f"total time={datetime.datetime.now()-time0}")
    return return_score_list, return_eq_list


def compiling(num_threads, is_use_1, use_binary_op, use_unary_op, x, y, units, model_score, is_progress, logger):
    save_score_list = np.full((num_threads, 1, 2), np.finfo(np.float64).min, dtype="float64")
    save_eq_list = np.full((num_threads, 1, 2 * 1 + 1), -100, dtype="int8")
    min_index_list = np.zeros(num_threads, dtype="int64")
    border_list = np.full((num_threads, 2), np.finfo(np.float64).min, dtype="float64")
    save = (save_score_list, save_eq_list, min_index_list, border_list)
    used = sub_loop_non_op(x, y, units, is_use_1, model_score, *save)
    fmax_max, fmax_min = 1e15, 1e-15
    if is_progress:
        with ProgressBar(total=0, leave=False, disable=True) as progress:
            sub_loop_binary_op(x, y, model_score, 1, fmax_max, fmax_min, 1, 0, use_binary_op, *save, *used, progress)
        if len(use_unary_op) != 0:
            with ProgressBar(total=0, leave=False, disable=True) as progress:
                sub_loop_unary_op(x, y, model_score, 1, fmax_max, fmax_min, 1, use_unary_op, *save, *used, progress)
    else:
        with loop_log(logger, interval=10000, tot_loop=0, header="") as progress:
            sub_loop_binary_op(x, y, model_score, 1, fmax_max, fmax_min, 1, 0, use_binary_op, *save, *used, progress)
        if len(use_unary_op) != 0:
            with loop_log(logger, interval=10000, tot_loop=0, header="") as progress:
                sub_loop_unary_op(x, y, model_score, 1, fmax_max, fmax_min, 1, use_unary_op, *save, *used, progress)


@njit(error_model="numpy")
def loop_counter_binary(use_binary_op, n_op1, n_op2, used_eq_dict):
    loop = 0
    for n_binary_op1 in range(n_op1 + 1):
        if n_binary_op1 in list(used_eq_dict[n_op1].keys()):
            len_use_eq_arr1 = used_eq_dict[n_op1][n_binary_op1].shape[0]
            for n_binary_op2 in range(n_op2 + 1):
                if n_binary_op2 in list(used_eq_dict[n_op2].keys()):
                    if n_binary_op1 >= n_binary_op2:
                        loop += len(use_binary_op) * len_use_eq_arr1 * used_eq_dict[n_op2][n_binary_op2].shape[0]
                    elif -4 in use_binary_op:
                        loop += len_use_eq_arr1 * used_eq_dict[n_op2][n_binary_op2].shape[0]
    return loop


@njit(error_model="numpy")
def loop_counter_unary(n_op, used_eq_dict):
    loop = 0
    for before_n_binary_op in range(n_op):
        if before_n_binary_op in list(used_eq_dict[n_op - 1].keys()):
            loop += used_eq_dict[n_op - 1][before_n_binary_op].shape[0]
    return loop


@njit(error_model="numpy")
def make_change_x_id(mask, x_max):
    # mask -> [3,4,2] みたいな
    # x_max -> 全体としての最大値
    if count_True(mask, 5, 0) == 0:  # 5 -> lambda x: x > border
        return 0
    TF = np.ones(x_max, dtype="bool")
    return_num = 0
    for i in range(mask.shape[0]):
        return_num *= x_max - i
        return_num += count_True(TF[: mask[i] - 1], 0, 0)  # 0 -> lambda x : x
        TF[mask[i] - 1] = False
    return return_num


@njit(error_model="numpy")
def eq_to_num(eq, x_max):
    int_nan = -100
    len_eq = count_True(eq, 2, int_nan)  # 2 -> lambda x: x != border
    num = 0
    for i in range(len_eq - 1, 0, -1):
        num = num * (x_max + 19) + (18 + np.int64(eq[i]))  # -5~-18
    num = num * (x_max + 1) + eq[0]
    return num


@njit(error_model="numpy")
def load_preprocessed_arr_len():
    with objmode(preprocessed_arr_len="int64[:]"):
        cache_path = os.fspath(pkg_resources.path("nb_sisso", "cache_folder"))
        preprocessed_arr_len = np.load(f"{cache_path}/arr_len.npy")
    return preprocessed_arr_len


@njit(error_model="numpy")
def load_preprocessed_results(n_binary_op, n_binary_op1):
    int_nan = -100
    # back_eq -> make_change_x_id -> changed_back_eq_x_num
    # num_to_index         :  [changed_back_eq_x_num]->changed_back_eq_x_index
    # preprocessed_results :  [id1,id2,changed_back_eq_x_index,merge_op]->eq_id or int_nan
    # eq -> make_change_x_id -> shuffled_eq_x_num
    # check_change_x  :

    n_binary_op2 = n_binary_op - 1 - n_binary_op1

    with objmode(num_to_index="int64[:]"):
        cache_path = os.fspath(pkg_resources.path("nb_sisso", "cache_folder"))
        num_to_index = np.load(f"{cache_path}/num_to_index_{n_binary_op}.npz")[str(n_binary_op1)]
    # num_to_index={j:i for i,j in enumerate(data)}

    with objmode(data="int64[:,:]"):
        cache_path = os.fspath(pkg_resources.path("nb_sisso", "cache_folder"))
        data = np.load(f"{cache_path}/cache_{n_binary_op}.npz")["arr_0"]
    preprocessed_arr_len = load_preprocessed_arr_len()
    shape = (preprocessed_arr_len[n_binary_op1], preprocessed_arr_len[n_binary_op2], np.max(num_to_index) + 1, 4)
    preprocessed_results = np.full(shape, int_nan, dtype="int64")
    for i in range(data.shape[0]):
        if data[i, 0] == n_binary_op1:
            preprocessed_results[data[i, 1], data[i, 2], num_to_index[data[i, 3]], data[i, 4] + 4] = i

    with objmode(need_calc="bool[:]"):
        cache_path = os.fspath(pkg_resources.path("nb_sisso", "cache_folder"))
        need_calc = np.load(f"{cache_path}/need_calc_{n_binary_op}.npz")["arr_0"]

    with objmode(check_change_x="int8[:,:,:]"):
        cache_path = os.fspath(pkg_resources.path("nb_sisso", "cache_folder"))
        check_change_x = np.load(f"{cache_path}/check_change_x_ones_{n_binary_op}.npz")["arr_0"]

    return preprocessed_results, num_to_index, need_calc, check_change_x


@njit(error_model="numpy")
def make_eq_id(n_binary_op1, info):
    int_nan = -100
    n_binary_op = info.shape[0] - 1
    unique = np.unique(info)
    len_unique = unique.shape[0]
    unique_arr = np.empty(len_unique + 1, dtype="int64")
    dict_x_to_num = np.full((len_unique, 2), int_nan, dtype="int8")
    retuen_arr = np.empty(n_binary_op - n_binary_op1, dtype="int8")

    n = 1
    index = 0
    for i in info[: n_binary_op1 + 1]:
        if i != 0:
            for j in range(len_unique):
                if unique[j] == i:
                    index == j
                    break
            if dict_x_to_num[index, 0] == int_nan:
                unique_arr[n] = i
                dict_x_to_num[index, 0] = n
                n += 1
    k = 0
    for i in info[n_binary_op1 + 1 :]:
        if i != 0:
            for j in range(len_unique):
                if unique[j] == i:
                    index == j
                    break
            if dict_x_to_num[index, 1] == int_nan:
                if dict_x_to_num[index, 0] == int_nan:
                    unique_arr[n] = i
                    num = n
                    n += 1
                else:
                    num = dict_x_to_num[index, 0]
                dict_x_to_num[index, 1] = num
                retuen_arr[k] = num
                k += 1
    return unique_arr[:n], make_change_x_id(retuen_arr[:k], n_binary_op + 1)


@njit(error_model="numpy")
def sub_loop_non_op(
    x,
    y,
    units,
    is_use_1,
    model_score,
    save_score_list,
    save_eq_list,
    min_index_list,
    border_list,
):
    int_nan = -100
    used_eq_dict = dict()
    used_unit_dict = dict()
    used_shape_id_dict = dict()
    used_info_dict = dict()

    used_eq_arr = np.empty((x.shape[0] + 1, 1), dtype="int8")
    used_unit_arr = np.empty((x.shape[0] + 1, units.shape[1]), dtype="int64")
    used_shape_id_arr = np.empty((x.shape[0] + 1), dtype="int64")
    used_info_arr = np.empty((x.shape[0] + 1, 1), dtype="int64")
    last_index = 0

    border1, border2 = border_list[0]

    if is_use_1:
        used_eq_arr[last_index] = 0
        used_unit_arr[last_index] = 0
        used_shape_id_arr[last_index] = 1
        used_info_arr[last_index, 0] = 0
        last_index += 1
    for i in range(1, x.shape[0] + 1):
        score1, score2 = model_score(x[i - 1], y)
        used_eq_arr[last_index] = i
        used_unit_arr[last_index] = units[i - 1]
        used_shape_id_arr[last_index] = 0
        used_info_arr[last_index, 0] = i
        last_index += 1
        if not np.isnan(score1):
            if score1 > border1:
                save_score_list[0, min_index_list[0], 0] = score1
                save_score_list[0, min_index_list[0], 1] = score2
                save_eq_list[0, min_index_list[0], 0] = i
                min_num1, min_num2, min_index = argmin_and_min(save_score_list[0])
                min_index_list[0] = min_index
                if border1 > min_num1:
                    border1 = min_num1
                    border2 = min_num2
                elif border1 == min_num1:
                    if border2 > min_num2:
                        border2 = min_num2
            elif score1 == border1:
                if score2 > border2:
                    save_score_list[0, min_index_list[0], 0] = score1
                    save_score_list[0, min_index_list[0], 1] = score2
                    save_eq_list[0, min_index_list[0], 0] = i
                    min_num1, min_num2, min_index = argmin_and_min(save_score_list[0])
                    min_index_list[0] = min_index
                    if border1 > min_num1:
                        border1 = min_num1
                        border2 = min_num2
                    elif border1 == min_num1:
                        if border2 > min_num2:
                            border2 = min_num2
    border_list[0, 0] = border1
    border_list[0, 1] = border2
    used_eq_dict[0] = {0: used_eq_arr[:last_index]}
    used_unit_dict[0] = {0: used_unit_arr[:last_index]}
    used_shape_id_dict[0] = {0: used_shape_id_arr[:last_index]}
    used_info_dict[0] = {0: used_info_arr[:last_index]}

    return used_eq_dict, used_unit_dict, used_shape_id_dict, used_info_dict


@njit(parallel=True, error_model="numpy")  # ,fastmath=True)
def sub_loop_binary_op(
    x,
    y,
    model_score,
    how_many_to_save,
    fmax_max,
    fmax_min,
    n_op,
    n_op1,
    use_binary_op,
    save_score_list,
    save_eq_list,
    min_index_list,
    border_list,
    used_eq_dict,
    used_unit_dict,
    used_shape_id_dict,
    used_info_dict,
    progress,
):
    int_nan = -100

    num_threads = save_score_list.shape[0]
    max_n_op = (save_eq_list.shape[2] - 1) // 2
    n_op2 = n_op - 1 - n_op1

    for n_binary_op1 in range(n_op1, -1, -1):
        if not n_binary_op1 in list(used_eq_dict[n_op1].keys()):
            continue
        use_eq_arr1 = used_eq_dict[n_op1][n_binary_op1]
        use_unit_arr1 = used_unit_dict[n_op1][n_binary_op1]
        use_shape_id_arr1 = used_shape_id_dict[n_op1][n_binary_op1]
        use_info_arr1 = used_info_dict[n_op1][n_binary_op1]
        if use_eq_arr1.shape[0] == 0:
            continue
        for n_binary_op2 in range(n_op2, -1, -1):
            if not n_binary_op2 in list(used_eq_dict[n_op2].keys()):
                continue
            if n_binary_op1 >= n_binary_op2:
                now_use_binary_op = np.array(use_binary_op)
            else:
                if -4 in use_binary_op:
                    now_use_binary_op = np.array([-4])
                else:
                    continue
            use_eq_arr2 = used_eq_dict[n_op2][n_binary_op2]
            use_unit_arr2 = used_unit_dict[n_op2][n_binary_op2]
            use_shape_id_arr2 = used_shape_id_dict[n_op2][n_binary_op2]
            use_info_arr2 = used_info_dict[n_op2][n_binary_op2]
            if use_eq_arr2.shape[0] == 0:
                continue
            n_binary_op = n_binary_op1 + n_binary_op2 + 1
            preprocessed_results, num_to_index, preprocessed_need_calc, check_change_x = load_preprocessed_results(
                n_binary_op, n_binary_op1
            )
            len_units = use_unit_arr1[0].shape[0]
            len_eq_arr1 = use_eq_arr1.shape[0]
            len_eq_arr2 = use_eq_arr2.shape[0]
            if max_n_op != n_op:
                len_used_arr = len(now_use_binary_op) * ((len_eq_arr1 * len_eq_arr2) // num_threads + 1)
            else:  # max_n_op==n_op:
                len_used_arr = 0
            used_eq_arr_thread = np.full((num_threads, len_used_arr, 2 * n_op + 1), int_nan, dtype="int8")
            used_unit_arr_thread = np.full((num_threads, len_used_arr, len_units), int_nan, dtype="int64")
            used_shape_id_arr_thread = np.full((num_threads, len_used_arr), int_nan, dtype="int64")
            used_info_arr_thread = np.full((num_threads, len_used_arr, n_binary_op + 1), int_nan, dtype="int64")
            last_index_thread = np.zeros(num_threads, dtype="int64")
            loop = len_eq_arr1 * len_eq_arr2
            for thread_id in prange(num_threads):
                score_list = save_score_list[thread_id].copy()
                eq_list = save_eq_list[thread_id].copy()
                min_index = min_index_list[thread_id]
                border1, border2 = border_list[thread_id]
                last_index = 0
                equation = np.empty((2 * n_op + 1), dtype="int8")
                info = np.empty((n_binary_op + 1), dtype="int64")
                for i in range(thread_id, loop, num_threads):
                    id1 = i % len_eq_arr1
                    i //= len_eq_arr1
                    id2 = i % len_eq_arr2
                    shape_id1 = use_shape_id_arr1[id1]
                    shape_id2 = use_shape_id_arr2[id2]
                    info[: n_binary_op1 + 1] = use_info_arr1[id1]
                    info[n_binary_op1 + 1 :] = use_info_arr2[id2]
                    unique_arr, changed_back_eq_x_num = make_eq_id(n_binary_op1, info)
                    changed_back_eq_x_index = num_to_index[changed_back_eq_x_num]
                    for merge_op in now_use_binary_op:
                        eq_id = preprocessed_results[shape_id1, shape_id2, changed_back_eq_x_index, merge_op + 4]
                        if eq_id != int_nan:
                            need_save = True
                            for k in range(check_change_x.shape[1]):
                                if check_change_x[eq_id, k, 0] == int_nan:
                                    break
                                if unique_arr[check_change_x[eq_id, k, 0]] > unique_arr[check_change_x[eq_id, k, 1]]:
                                    need_save = False
                                    break
                            if need_save:
                                unit1 = use_unit_arr1[id1]
                                unit2 = use_unit_arr2[id2]
                                if merge_op in [-1, -2]:
                                    if np.any(unit1 != unit2):
                                        continue
                                eq1 = use_eq_arr1[id1]
                                eq2 = use_eq_arr2[id2]
                                len_eq1 = count_True(eq1, 2, int_nan)  # 2 -> lambda x: x != border
                                len_eq2 = count_True(eq2, 2, int_nan)  # 2 -> lambda x: x != border
                                equation[:len_eq1] = eq1[:len_eq1]
                                equation[len_eq1 : len_eq1 + len_eq2] = eq2[:len_eq2]
                                equation[len_eq1 + len_eq2] = merge_op
                                equation[len_eq1 + len_eq2 + 1 :] = int_nan
                                ans_num = calc_RPN(x, equation, fmax_max, fmax_min)
                                if not np.isnan(ans_num[0]):
                                    if max_n_op != n_op:
                                        match merge_op:
                                            case -1:  # +
                                                used_unit_arr_thread[thread_id, last_index] = unit1
                                            case -2:  # -
                                                used_unit_arr_thread[thread_id, last_index] = unit1
                                            case -3:  # *
                                                used_unit_arr_thread[thread_id, last_index] = unit1 + unit2
                                            case -4:  # /
                                                used_unit_arr_thread[thread_id, last_index] = unit1 - unit2
                                        used_eq_arr_thread[thread_id, last_index] = equation
                                        used_shape_id_arr_thread[thread_id, last_index] = eq_id
                                        used_info_arr_thread[thread_id, last_index] = info
                                        last_index += 1
                                    if preprocessed_need_calc[eq_id]:
                                        score1, score2 = model_score(ans_num, y)
                                        if not (np.isnan(score1) or np.isnan(score2)):
                                            if score1 > border1:
                                                score_list[min_index, 0] = score1
                                                score_list[min_index, 1] = score2
                                                eq_list[min_index, : 2 * n_op + 1] = equation
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
                                                    eq_list[min_index, : 2 * n_op + 1] = equation
                                                    min_num1, min_num2, min_index = argmin_and_min(score_list)
                                                    if border1 > min_num1:
                                                        border1 = min_num1
                                                        border2 = min_num2
                                                    elif border1 == min_num1:
                                                        if border2 > min_num2:
                                                            border2 = min_num2
                        progress.update(1)
                save_score_list[thread_id] = score_list
                save_eq_list[thread_id] = eq_list
                min_index_list[thread_id] = min_index
                border_list[thread_id, 0] = border1
                border_list[thread_id, 1] = border2
                if max_n_op != n_op:
                    last_index_thread[thread_id] = last_index

            if num_threads != 1:
                border1 = np.partition(save_score_list[:, :, 0].ravel(), -how_many_to_save)[-how_many_to_save]
                n_save_same_score1 = how_many_to_save - np.sum(save_score_list[:, :, 0] > border1)
                border2 = np.sort(save_score_list[:, :, 1].ravel()[save_score_list[:, :, 0].ravel() == border1])[::-1][
                    n_save_same_score1 - 1
                ]
                border_list[:, 0] = border1
                border_list[:, 1] = border2

            if max_n_op != n_op:
                sum_last_index = np.sum(last_index_thread)
                used_eq_arr = np.full((sum_last_index, 2 * n_op + 1), int_nan, dtype="int8")
                used_unit_arr = np.full((sum_last_index, len_units), int_nan, dtype="int64")
                used_shape_id_arr = np.full((sum_last_index), int_nan, dtype="int64")
                used_info_arr = np.full((sum_last_index, n_binary_op + 1), int_nan, dtype="int64")
                for thread_id in prange(num_threads):
                    before_index = np.sum(last_index_thread[:thread_id])
                    used_eq_arr[before_index : before_index + last_index_thread[thread_id]] = used_eq_arr_thread[
                        thread_id, : last_index_thread[thread_id]
                    ]
                    used_unit_arr[before_index : before_index + last_index_thread[thread_id]] = used_unit_arr_thread[
                        thread_id, : last_index_thread[thread_id]
                    ]
                    used_shape_id_arr[before_index : before_index + last_index_thread[thread_id]] = (
                        used_shape_id_arr_thread[thread_id, : last_index_thread[thread_id]]
                    )
                    used_info_arr[before_index : before_index + last_index_thread[thread_id]] = used_info_arr_thread[
                        thread_id, : last_index_thread[thread_id]
                    ]
                if n_op in used_eq_dict:
                    if n_binary_op in used_eq_dict[n_op]:
                        used_eq_dict[n_op][n_binary_op] = np.concatenate((used_eq_dict[n_op][n_binary_op], used_eq_arr))
                        used_unit_dict[n_op][n_binary_op] = np.concatenate(
                            (used_unit_dict[n_op][n_binary_op], used_unit_arr)
                        )
                        used_shape_id_dict[n_op][n_binary_op] = np.concatenate(
                            (used_shape_id_dict[n_op][n_binary_op], used_shape_id_arr)
                        )
                        used_info_dict[n_op][n_binary_op] = np.concatenate(
                            (used_info_dict[n_op][n_binary_op], used_info_arr)
                        )
                    else:
                        used_eq_dict[n_op][n_binary_op] = used_eq_arr
                        used_unit_dict[n_op][n_binary_op] = used_unit_arr
                        used_shape_id_dict[n_op][n_binary_op] = used_shape_id_arr
                        used_info_dict[n_op][n_binary_op] = used_info_arr
                else:
                    used_eq_dict[n_op] = {n_binary_op: used_eq_arr}
                    used_unit_dict[n_op] = {n_binary_op: used_unit_arr}
                    used_shape_id_dict[n_op] = {n_binary_op: used_shape_id_arr}
                    used_info_dict[n_op] = {n_binary_op: used_info_arr}


@njit(error_model="numpy")
def check_eq(eq, ban_ops):  # use exp,exp-,log,sin,cos
    for ban_op in ban_ops:
        if ban_op in eq:
            return False
    return True


@njit(parallel=True, error_model="numpy")  # ,fastmath=True)
def sub_loop_unary_op(
    x,
    y,
    model_score,
    how_many_to_save,
    fmax_max,
    fmax_min,
    n_op,
    use_unary_op,
    save_score_list,
    save_eq_list,
    min_index_list,
    border_list,
    used_eq_dict,
    used_unit_dict,
    used_shape_id_dict,
    used_info_dict,
    progress,
):
    int_nan = -100
    num_threads = save_score_list.shape[0]
    max_n_op = (save_eq_list.shape[2] - 1) // 2
    x_max = x.shape[0]
    for before_n_binary_op in range(n_op):
        if not before_n_binary_op in list(used_eq_dict[n_op - 1].keys()):
            continue
        use_eq_arr = used_eq_dict[n_op - 1][before_n_binary_op]
        use_unit_arr = used_unit_dict[n_op - 1][before_n_binary_op]
        len_units = use_unit_arr.shape[1]
        if max_n_op != n_op:
            len_used_arr = len(use_unary_op) * (use_eq_arr.shape[0] // num_threads + 1)
        else:  # max_n_op==n_op:
            len_used_arr = 0
        used_eq_arr_thread = np.full((num_threads, len_used_arr, 2 * n_op + 1), int_nan, dtype="int8")
        used_unit_arr_thread = np.full((num_threads, len_used_arr, len_units), int_nan, dtype="int64")
        used_shape_id_arr_thread = np.full((num_threads, len_used_arr), int_nan, dtype="int64")
        used_info_arr_thread = np.full((num_threads, len_used_arr, 1), int_nan, dtype="int64")
        last_index_thread = np.zeros(num_threads, dtype="int64")
        loop = use_eq_arr.shape[0]
        for thread_id in prange(num_threads):
            score_list = save_score_list[thread_id].copy()
            eq_list = save_eq_list[thread_id].copy()
            min_index = min_index_list[thread_id]
            border1, border2 = border_list[thread_id, 0], border_list[thread_id, 1]
            last_index = 0
            equation = np.empty((2 * n_op + 1), dtype="int8")
            for i in range(thread_id, loop, num_threads):
                base_eq = use_eq_arr[i]
                len_base_eq = count_True(base_eq, 2, int_nan)  # 2 -> lambda x: x != border
                base_unit = use_unit_arr[i]
                equation[:len_base_eq] = base_eq[:len_base_eq]
                equation[len_base_eq + 1 :] = int_nan
                for op in use_unary_op:
                    checked = False
                    unit = base_unit
                    match op:
                        case -5:  # *-1
                            None  # 存在意義が分からない
                        case -6:  # ^-1
                            # if equation[len_base_eq-1]!=-6:#^-1
                            # checked=True
                            # unit=-1*base_unit#units更新
                            None  # 存在意義が分からない
                        case -7:  # ^2
                            if equation[len_base_eq - 1] != -8:  # sqrt
                                if n_op != 1:
                                    checked = True
                                    unit = 2 * base_unit  # units更新
                        case -8:  # sqrt
                            if not equation[len_base_eq - 1] in [-7, -12]:  # ^2,^6
                                checked = True
                                unit = base_unit // 2  # units更新
                        case -9:  # | |
                            checked = True
                            unit = base_unit  # units更新なし
                        case -10:  # ^3
                            if equation[len_base_eq - 1] != -11:  # cbrt
                                checked = True
                                unit = 3 * base_unit  # units更新
                        case -11:  # cbrt
                            if not equation[len_base_eq - 1] in [-10, -12]:  # ^3,^6
                                checked = True
                                unit = base_unit // 3  # units更新
                        case -12:  # ^6
                            if not equation[len_base_eq - 1] in [-8, -11]:  # sqrt,cbrt
                                checked = True
                                unit = 6 * base_unit  # units更新
                        case -13:  # exp
                            # exp,exp-,log,sin,cos
                            if check_eq(base_eq, np.array([-13, -14, -15, -16, -17])):
                                if np.all(base_unit == 0):  # 無次元に制限
                                    checked = True
                                    # unit=base_unit
                        case -14:  # exp-
                            # exp,exp-,log,sin,cos
                            if check_eq(base_eq, np.array([-13, -14, -15, -16, -17])):
                                if np.all(base_unit == 0):  # 無次元に制限
                                    checked = True
                                    # unit=base_unit
                        case -15:  # log
                            # exp,exp-,log,sin,cos
                            if check_eq(base_eq, np.array([-13, -14, -15, -16, -17])):
                                if np.all(base_unit == 0):  # 無次元に制限
                                    checked = True
                                    # unit=base_unit
                        case -16:  # sin
                            # exp,exp-,log,sin,cos
                            if check_eq(base_eq, np.array([-13, -14, -15, -16, -17])):
                                if np.all(base_unit == 0):  # 無次元に制限
                                    checked = True
                                    # unit=base_unit
                        case -17:  # cos
                            # exp,exp-,log,sin,cos
                            if check_eq(base_eq, np.array([-13, -14, -15, -16, -17])):
                                if np.all(base_unit == 0):  # 無次元に制限
                                    checked = True
                                    # unit=base_unit
                        case -18:  # scd  #わっかんね
                            if np.all(base_unit == 0):  # 無次元に制限????
                                checked = True
                                # unit=base_unit#units更新????
                    if checked:
                        equation[len_base_eq] = op
                        ans_num = calc_RPN(x, equation, fmax_max, fmax_min)
                        if not np.isnan(ans_num[0]):
                            if max_n_op != n_op:
                                used_unit_arr_thread[thread_id, last_index] = unit
                                used_eq_arr_thread[thread_id, last_index] = equation
                                used_shape_id_arr_thread[thread_id, last_index] = 0
                                used_info_arr_thread[thread_id, last_index] = eq_to_num(equation, x_max)
                                last_index += 1
                            score1, score2 = model_score(ans_num, y)
                            if not (np.isnan(score1) or np.isnan(score2)):
                                if score1 > border1:
                                    score_list[min_index, 0] = score1
                                    score_list[min_index, 1] = score2
                                    eq_list[min_index, : 2 * n_op + 1] = equation
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
                                        eq_list[min_index, : 2 * n_op + 1] = equation
                                        min_num1, min_num2, min_index = argmin_and_min(score_list)
                                        if border1 > min_num1:
                                            border1 = min_num1
                                            border2 = min_num2
                                        elif border1 == min_num1:
                                            if border2 > min_num2:
                                                border2 = min_num2
                progress.update(1)
            save_score_list[thread_id] = score_list
            save_eq_list[thread_id] = eq_list
            min_index_list[thread_id] = min_index
            border_list[thread_id, 0] = border1
            border_list[thread_id, 1] = border2
            if max_n_op != n_op:
                last_index_thread[thread_id] = last_index
        if num_threads != 1:
            border1 = np.partition(save_score_list[:, :, 0].ravel(), -how_many_to_save)[-how_many_to_save]
            n_save_same_score1 = how_many_to_save - np.sum(save_score_list[:, :, 0] > border1)
            border2 = np.sort(save_score_list[:, :, 1].ravel()[save_score_list[:, :, 0].ravel() == border1])[::-1][
                n_save_same_score1 - 1
            ]
            border_list[:, 0] = border1
            border_list[:, 1] = border2

        if max_n_op != n_op:
            sum_last_index = np.sum(last_index_thread)
            used_eq_arr = np.full((sum_last_index, 2 * n_op + 1), int_nan, dtype="int8")
            used_unit_arr = np.full((sum_last_index, len_units), int_nan, dtype="int64")
            used_shape_id_arr = np.full((sum_last_index), int_nan, dtype="int64")
            used_info_arr = np.full((sum_last_index, 1), int_nan, dtype="int64")
            for thread_id in prange(num_threads):
                before_index = np.sum(last_index_thread[:thread_id])
                used_eq_arr[before_index : before_index + last_index_thread[thread_id]] = used_eq_arr_thread[
                    thread_id, : last_index_thread[thread_id]
                ]
                used_unit_arr[before_index : before_index + last_index_thread[thread_id]] = used_unit_arr_thread[
                    thread_id, : last_index_thread[thread_id]
                ]
                used_shape_id_arr[before_index : before_index + last_index_thread[thread_id]] = (
                    used_shape_id_arr_thread[thread_id, : last_index_thread[thread_id]]
                )
                used_info_arr[before_index : before_index + last_index_thread[thread_id]] = used_info_arr_thread[
                    thread_id, : last_index_thread[thread_id]
                ]
            if 0 in used_eq_dict:
                if 0 in used_eq_dict[n_op]:
                    used_eq_dict[n_op][0] = np.concatenate((used_eq_dict[n_op][0], used_eq_arr))
                    used_unit_dict[n_op][0] = np.concatenate((used_unit_dict[n_op][0], used_unit_arr))
                    used_shape_id_dict[n_op][0] = np.concatenate((used_shape_id_dict[n_op][0], used_shape_id_arr))
                    used_info_dict[n_op][0] = np.concatenate((used_info_dict[n_op][0], used_info_arr))
                else:
                    used_eq_dict[n_op][0] = used_eq_arr
                    used_unit_dict[n_op][0] = used_unit_arr
                    used_shape_id_dict[n_op][0] = used_shape_id_arr
                    used_info_dict[n_op][0] = used_info_arr
            else:
                used_eq_dict[n_op] = {0: used_eq_arr}
                used_unit_dict[n_op] = {0: used_unit_arr}
                used_shape_id_dict[n_op] = {0: used_shape_id_arr}
                used_info_dict[n_op] = {0: used_info_arr}
