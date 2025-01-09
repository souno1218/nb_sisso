import numpy as np
import numba, os, datetime, logging, multiprocessing
from numba import njit, prange, set_num_threads, objmode, get_num_threads

from .utils_for_make_cache import *
from .log_progress import loop_log


def main(max_op, num_threads, len_x=5, log_interval=300, verbose=True, logger=None):
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

    # num_threads
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    set_num_threads(num_threads)
    if not thread_check(num_threads):
        logger.info(f"can't set thread : {num_threads}")

    # log
    logger.info(f"numba={numba.__version__}, numpy={np.__version__}")
    logger.info(f"OPT={numba.config.OPT}, THREADING_LAYER={numba.config.THREADING_LAYER}")
    logger.info(
        f"USING_SVML={numba.config.USING_SVML}, ENABLE_AVX={numba.config.ENABLE_AVX}, DISABLE_JIT={numba.config.DISABLE_JIT}"
    )
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
    time1, time2 = datetime.datetime.now(), datetime.datetime.now()
    header = "      "
    with loop_log(logger, interval=log_interval, tot_loop=loop, header=header) as progress:
        before_similar_num_list = make_before_similar_num_list(
            num_threads, max_op, random_x, before_equations, progress
        )
    time1, time2 = time2, datetime.datetime.now()
    dTime = time2 - time1
    logger.info(f"   time : {dTime}")
    logger.info(f"make_unique_equations")
    time1, time2 = time2, datetime.datetime.now()
    make_unique_equations(max_op, num_threads, random_x, before_similar_num_list, log_interval, logger)
    time1, time2 = time2, datetime.datetime.now()
    dTime = time2 - time1
    logger.info(f"   time : {dTime}")
    logger.info(f"END")
    dTime = time2 - time0
    logger.info(f"total time : {dTime}")


def make_final_cache(
    max_op,
    equation_list,
    need_calc_list,
    base_eq_id_list,
    check_change_x_tot,
    check_change_x_ones,
    logger,
):
    # base_eq_id_list = merge_op,n_op1,id1,id2,changed_back_eq_id
    int_nan = -100
    base = np.load("arr_len.npy")
    arr_len = np.zeros((max(base.shape[0], max_op + 1)), dtype="int64")
    arr_len[: base.shape[0]] = base
    arr_len[max_op] = base_eq_id_list.shape[0]
    logger.info(f"   tot             , {list(base)} -> {list(arr_len)}")
    np.save("arr_len", arr_len)
    return_np_index_to_num = dict()
    cache = dict()
    for n_op1 in range(max_op // 2, max_op):
        n_op2 = max_op - 1 - n_op1
        unique_num = np.empty((0), dtype="int64")
        for i in (1, n_op2 + 2):
            all_pattern = nb_permutations(np.arange(1, max_op + 2), i)
            all_num = np.array([make_change_x_id(all_pattern[j], max_op + 1) for j in range(all_pattern.shape[0])])
            unique_num = np.unique(np.concatenate((unique_num, all_num)))
        dict_num_to_index = {j: i for i, j in enumerate(unique_num)}
        return_np_index_to_num[n_op1] = unique_num
        preprocessed_len1 = arr_len[n_op1]
        preprocessed_len2 = arr_len[n_op2]
        index = np.arange(base_eq_id_list.shape[0])[base_eq_id_list[:, 0] == n_op1]
        cache_n_op1 = np.empty((index.shape[0], 3), dtype="int64")
        for i in range(index.shape[0]):
            id = dict_num_to_index[base_eq_id_list[index[i], 3]] * preprocessed_len2 + base_eq_id_list[index[i], 2]
            id = id * preprocessed_len1 + base_eq_id_list[index[i], 1]
            cache_n_op1[i, 0] = id
            cache_n_op1[i, 1] = index[i]
            cache_n_op1[i, 2] = base_eq_id_list[index[i], 4]
        cache[n_op1] = cache_n_op1
    np.savez(
        f"cache_{max_op}",
        **{str(n_op1): cache[n_op1] for n_op1 in range(max_op // 2, max_op)},
    )
    np.savez(
        f"num_to_index_{max_op}",
        **{str(n_op1): return_np_index_to_num[n_op1] for n_op1 in range(max_op // 2, max_op)},
    )

    np.savez(f"need_calc_{max_op}", need_calc_list)

    len_check_change_x_tot = 0
    for j in range(check_change_x_tot.shape[1]):
        if np.any(check_change_x_tot[:, j, 0] != int_nan):
            len_check_change_x_tot = j + 1
    np.savez(f"check_change_x_tot_{max_op}", check_change_x_tot[:, :len_check_change_x_tot])

    len_check_change_x_ones = 0
    for j in range(check_change_x_ones.shape[1]):
        if np.any(check_change_x_ones[:, j, 0] != int_nan):
            len_check_change_x_ones = j + 1
    np.savez(f"check_change_x_ones_{max_op}", check_change_x_ones[:, :len_check_change_x_ones])
    # savez_compressed
    np.save(f"operator_{max_op}", equation_list)


def make_final_cache_7(base_eq_id_list, check_change_x_ones, logger):
    # base_eq_id_list = merge_op,n_op1,id1,id2,changed_back_eq_id
    max_op = 7
    int_nan = -100
    base = np.load("arr_len.npy")
    arr_len = np.zeros((max(base.shape[0], max_op + 1)), dtype="int64")
    arr_len[: base.shape[0]] = base
    arr_len[max_op] = base_eq_id_list.shape[0]
    logger.info(f"   tot             , {list(base)} -> {list(arr_len)}")
    np.save("arr_len", arr_len)
    return_np_index_to_num = dict()
    cache = dict()
    for n_op1 in range(max_op // 2, max_op):
        n_op2 = max_op - 1 - n_op1
        unique_num = np.empty((0), dtype="int64")
        for i in (1, n_op2 + 2):
            all_pattern = nb_permutations(np.arange(1, max_op + 2), i)
            all_num = np.array([make_change_x_id(all_pattern[j], max_op + 1) for j in range(all_pattern.shape[0])])
            unique_num = np.unique(np.concatenate((unique_num, all_num)))
        dict_num_to_index = {j: i for i, j in enumerate(unique_num)}
        return_np_index_to_num[n_op1] = unique_num
        preprocessed_len1 = arr_len[n_op1]
        preprocessed_len2 = arr_len[n_op2]
        index = np.arange(base_eq_id_list.shape[0])[base_eq_id_list[:, 0] == n_op1]
        cache_n_op1 = np.empty((index.shape[0], 3), dtype="int64")
        for i in range(index.shape[0]):
            id = dict_num_to_index[base_eq_id_list[index[i], 3]] * preprocessed_len2 + base_eq_id_list[index[i], 2]
            id = id * preprocessed_len1 + base_eq_id_list[index[i], 1]
            cache_n_op1[i, 0] = id
            cache_n_op1[i, 1] = index[i]
            cache_n_op1[i, 2] = base_eq_id_list[index[i], 4]
        cache[n_op1] = cache_n_op1
    np.savez(
        f"cache_{max_op}",
        **{str(n_op1): cache[n_op1] for n_op1 in range(max_op // 2, max_op)},
    )
    np.savez(
        f"num_to_index_{max_op}",
        **{str(n_op1): return_np_index_to_num[n_op1] for n_op1 in range(max_op // 2, max_op)},
    )

    np.savez(f"need_calc_{max_op}", np.ones(base_eq_id_list.shape[0], dtype="bool"))

    len_check_change_x_ones = 0
    for j in range(check_change_x_ones.shape[1]):
        if np.any(check_change_x_ones[:, j, 0] != int_nan):
            len_check_change_x_ones = j + 1
    np.savez(f"check_change_x_ones_{max_op}", check_change_x_ones[:, :len_check_change_x_ones])


def make_unique_equations(max_op, num_threads, random_x, before_similar_num_list, log_interval, logger):
    num_threads = int(num_threads)
    int_nan = -100
    saved_equation_list = np.empty((0, 2 * max_op + 1), dtype="int8")
    saved_base_eq_id_list = np.empty((0, 5), dtype="int64")
    saved_check_change_x_tot = np.empty((0, (max_op * (max_op + 1)) // 2, 2), dtype="int8")
    saved_check_change_x_ones = np.empty((0, (max_op * (max_op + 1)) // 2, 2), dtype="int8")
    saved_check_exist_eq_list = np.empty((0, 2 * max_op + 1), dtype="int8")
    saved_check_exist_id_list = np.empty((0, 5), dtype="int64")
    saved_check_exist_change_x_tot_list = np.empty((0, (max_op * (max_op + 1)) // 2, 2), dtype="int8")
    saved_check_exist_change_x_ones_list = np.empty((0, (max_op * (max_op + 1)) // 2, 2), dtype="int8")
    if max_op != 7:
        saved_need_calc_list = np.empty((0), dtype="bool")
        saved_similar_num_list = np.empty((0, 2, random_x.shape[1]), dtype="float64")
        saved_check_exist_num_list = np.empty((0, 2, random_x.shape[1]), dtype="float64")
        time1, time2 = datetime.datetime.now(), datetime.datetime.now()
        for n_op1 in range(max_op - 1, (max_op - 2) // 2, -1):
            n_op2 = max_op - 1 - n_op1
            logger.info(f"   n_op1 = {n_op1}, n_op2 = {n_op2}")
            how_loop, loop_per_threads = loop_count(max_op, n_op1, num_threads)
            logger.info(f"      make_unique_equations_thread")
            mem_size_per_1data = (
                2 * random_x.shape[1] * 8
                + 8
                + 1
                + 1
                + 1
                + (2 * max_op + 1)
                + 5 * 8
                + (max_op * (max_op + 1)) * 2  # Byte
            )
            mem_size = ((mem_size_per_1data * loop_per_threads * num_threads) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} Bytes, loop={how_loop})"
            )
            time1, time2 = time2, datetime.datetime.now()
            header = "         "
            with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
                (
                    TF_list,
                    similar_num_list,
                    need_calc_list,
                    equation_list,
                    base_eq_id_list,
                    check_change_x_tot,
                    check_change_x_ones,
                    check_exist_TF,
                ) = make_unique_equations_thread(
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
            how_loop = int(np.sum(np.sort(np.sum(TF_list, axis=1))[::-1][1:]))
            mem_size_per_1data = 2 * random_x.shape[1] * 8 + 8  # Byte
            mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
            loop = np.sum(TF_list)
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} Bytes, loop={loop})"
            )
            time1, time2 = time2, datetime.datetime.now()
            header = "         "
            with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
                TF_list, need_calc_list = dim_reduction(TF_list, similar_num_list, need_calc_list, progress)
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")

            logger.info(f"      make_check_exist_info")

            sum_check_exist = int(np.sum(check_exist_TF))
            mem_size_per_1data = (
                2 * random_x.shape[1] * 8
                + (2 * max_op + 1)
                + 5 * 8
                + check_change_x_tot.shape[2] * 2
                + check_change_x_ones.shape[2] * 2
            )  # Bite
            mem_size = ((mem_size_per_1data * sum_check_exist) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} Bytes, loop={sum_check_exist})"
            )
            check_exist_num, check_exist_eq, check_exist_id, check_exist_change_x_tot, check_exist_change_x_ones = (
                make_check_exist_info(
                    check_exist_TF,
                    similar_num_list,
                    equation_list,
                    base_eq_id_list,
                    check_change_x_tot,
                    check_change_x_ones,
                )
            )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")

            logger.info(f"      dim_reduction_info")
            how_loop = int(np.sum(TF_list))
            mem_size_per_1data = (
                2 * random_x.shape[1] * 8
                + (2 * max_op + 1)
                + 5 * 8
                + 1
                + check_change_x_tot.shape[2] * 2
                + check_change_x_ones.shape[2] * 2
            )  # Bite
            mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
            )
            (
                equation_list,
                similar_num_list,
                need_calc_list,
                base_eq_id_list,
                check_change_x_tot,
                check_change_x_ones,
            ) = dim_reduction_info(
                TF_list,
                similar_num_list,
                need_calc_list,
                equation_list,
                base_eq_id_list,
                check_change_x_tot,
                check_change_x_ones,
            )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")
            saved_equation_list = np.concatenate((saved_equation_list, equation_list))
            saved_similar_num_list = np.concatenate((saved_similar_num_list, similar_num_list))
            saved_similar_num_list = saved_similar_num_list[np.argsort(saved_similar_num_list[:, 0, 0])].copy()
            saved_need_calc_list = np.concatenate((saved_need_calc_list, need_calc_list))
            saved_base_eq_id_list = np.concatenate((saved_base_eq_id_list, base_eq_id_list))
            saved_check_change_x_tot = np.concatenate((saved_check_change_x_tot, check_change_x_tot))
            saved_check_change_x_ones = np.concatenate((saved_check_change_x_ones, check_change_x_ones))
            saved_check_exist_num_list = np.concatenate((saved_check_exist_num_list, check_exist_num))
            saved_check_exist_eq_list = np.concatenate((saved_check_exist_eq_list, check_exist_eq))
            saved_check_exist_id_list = np.concatenate((saved_check_exist_id_list, check_exist_id))
            saved_check_exist_change_x_tot_list = np.concatenate(
                (saved_check_exist_change_x_tot_list, check_exist_change_x_tot)
            )
            saved_check_exist_change_x_ones_list = np.concatenate(
                (saved_check_exist_change_x_ones_list, check_exist_change_x_ones)
            )

        logger.info(f"   check_exist_step1")
        how_loop = int(saved_check_exist_num_list.shape[0])
        mem_size_per_1data = 2  # Bite
        mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
            check_exist_need_calc, indexes = check_exist_step1(
                saved_similar_num_list, saved_check_exist_num_list, saved_check_exist_change_x_tot_list, progress
            )
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")
        saved_check_exist_num_list = saved_check_exist_num_list[indexes]
        saved_check_exist_eq_list = saved_check_exist_eq_list[indexes]
        saved_check_exist_id_list = saved_check_exist_id_list[indexes]
        saved_check_exist_change_x_tot_list = saved_check_exist_change_x_tot_list[indexes]
        saved_check_exist_change_x_ones_list = saved_check_exist_change_x_ones_list[indexes]
        saved_check_exist_need_calc_list = check_exist_need_calc[indexes]

        logger.info(f"   check_exist_step2")
        how_loop = int(saved_check_exist_num_list.shape[0])
        mem_size_per_1data = 1  # Bite
        mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop - 1, header=header) as progress:
            same = check_exist_step2(saved_check_exist_num_list, progress)
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")

        logger.info(f"   check_exist_step3")
        how_loop = 0
        for i in range(np.max(same) + 1 if same.shape[0] != 0 else 0):
            how_loop += int(np.sum(same == i) - 1)
        mem_size_per_1data = 1 + 8  # Bite
        mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
            indexes, saved_check_exist_need_calc_list = check_exist_step3(
                max_op,
                random_x,
                same,
                saved_check_exist_eq_list,
                saved_check_exist_need_calc_list,
                saved_check_exist_change_x_tot_list,
                progress,
            )
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")

        saved_check_exist_eq_list = saved_check_exist_eq_list[indexes]
        saved_check_exist_id_list = saved_check_exist_id_list[indexes]
        saved_check_exist_change_x_tot_list = saved_check_exist_change_x_tot_list[indexes]
        saved_check_exist_change_x_ones_list = saved_check_exist_change_x_ones_list[indexes]
        saved_check_exist_need_calc_list = saved_check_exist_need_calc_list[indexes]

        saved_equation_list = np.concatenate((saved_equation_list, saved_check_exist_eq_list))
        saved_need_calc_list = np.concatenate((saved_need_calc_list, saved_check_exist_need_calc_list))
        saved_base_eq_id_list = np.concatenate((saved_base_eq_id_list, saved_check_exist_id_list))
        saved_check_change_x_tot = np.concatenate((saved_check_change_x_tot, saved_check_exist_change_x_tot_list))
        saved_check_change_x_ones = np.concatenate((saved_check_change_x_ones, saved_check_exist_change_x_ones_list))

        sort_index = np.argsort(saved_need_calc_list.astype("int8"))[::-1]

        saved_equation_list = saved_equation_list[sort_index]
        saved_need_calc_list = saved_need_calc_list[sort_index]
        saved_base_eq_id_list = saved_base_eq_id_list[sort_index]
        saved_check_change_x_tot = saved_check_change_x_tot[sort_index]
        saved_check_change_x_ones = saved_check_change_x_ones[sort_index]
        make_final_cache(
            max_op,
            saved_equation_list,
            saved_need_calc_list,
            saved_base_eq_id_list,
            saved_check_change_x_tot,
            saved_check_change_x_ones,
            logger,
        )
        Tcount = np.sum(saved_need_calc_list)
        Fcount = np.sum(~saved_need_calc_list)
        logger.info(f"need calc  {Tcount}:{Fcount}")
    else:  # max_op==7
        before_similar_num_list = before_similar_num_list[:, 0, :].copy()
        saved_similar_num_list = np.empty((0, random_x.shape[1]), dtype="float64")
        saved_check_exist_num_list = np.empty((0, random_x.shape[1]), dtype="float64")
        time1, time2 = datetime.datetime.now(), datetime.datetime.now()
        for n_op1 in range(max_op - 1, (max_op - 2) // 2, -1):
            n_op2 = max_op - 1 - n_op1
            logger.info(f"   n_op1 = {n_op1}, n_op2 = {n_op2}")
            how_loop, loop_per_threads = loop_count(max_op, n_op1, num_threads)
            logger.info(f"      make_unique_equations_thread")
            mem_size_per_1data = (
                random_x.shape[1] * 8 + 8 + 1 + 1 + (2 * max_op + 1) + 5 * 8 + (max_op * (max_op + 1)) * 2  # Byte
            )
            mem_size = ((mem_size_per_1data * loop_per_threads * num_threads) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} Bytes, loop={how_loop})"
            )
            time1, time2 = time2, datetime.datetime.now()
            header = "         "
            with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
                (
                    TF_list,
                    similar_num_list,
                    equation_list,
                    base_eq_id_list,
                    check_change_x_tot,
                    check_change_x_ones,
                    check_exist_TF,
                ) = make_unique_equations_thread_7(
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
            how_loop = int(np.sum(np.sort(np.sum(TF_list, axis=1))[::-1][1:]))
            mem_size_per_1data = random_x.shape[1] * 8 + 8  # Byte
            mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
            loop = np.sum(TF_list)
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} Bytes, loop={loop})"
            )
            time1, time2 = time2, datetime.datetime.now()
            header = "         "
            with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
                TF_list = dim_reduction_7(TF_list, similar_num_list, progress)
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")

            logger.info(f"      make_check_exist_info")

            sum_check_exist = int(np.sum(check_exist_TF))
            mem_size_per_1data = (
                random_x.shape[1] * 8
                + (2 * max_op + 1)
                + 5 * 8
                + check_change_x_tot.shape[2] * 2
                + check_change_x_ones.shape[2] * 2
            )  # Bite
            mem_size = ((mem_size_per_1data * sum_check_exist) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} Bytes, loop={sum_check_exist})"
            )
            check_exist_num, check_exist_eq, check_exist_id, check_exist_change_x_tot, check_exist_change_x_ones = (
                make_check_exist_info_7(
                    check_exist_TF,
                    similar_num_list,
                    equation_list,
                    base_eq_id_list,
                    check_change_x_tot,
                    check_change_x_ones,
                )
            )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")

            logger.info(f"      dim_reduction_info")
            how_loop = int(np.sum(TF_list))
            mem_size_per_1data = (
                random_x.shape[1] * 8
                + (2 * max_op + 1)
                + 5 * 8
                + check_change_x_tot.shape[2] * 2
                + check_change_x_ones.shape[2] * 2
            )  # Bite
            mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
            )
            equation_list, similar_num_list, base_eq_id_list, check_change_x_tot, check_change_x_ones = (
                dim_reduction_info_7(
                    TF_list, similar_num_list, equation_list, base_eq_id_list, check_change_x_tot, check_change_x_ones
                )
            )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")
            saved_similar_num_list = np.concatenate((saved_similar_num_list, similar_num_list))
            saved_similar_num_list = saved_similar_num_list[np.argsort(saved_similar_num_list[:, 0])].copy()
            saved_base_eq_id_list = np.concatenate((saved_base_eq_id_list, base_eq_id_list))
            saved_check_change_x_ones = np.concatenate((saved_check_change_x_ones, check_change_x_ones))
            saved_check_exist_num_list = np.concatenate((saved_check_exist_num_list, check_exist_num))
            saved_check_exist_eq_list = np.concatenate((saved_check_exist_eq_list, check_exist_eq))
            saved_check_exist_id_list = np.concatenate((saved_check_exist_id_list, check_exist_id))
            saved_check_exist_change_x_tot_list = np.concatenate(
                (saved_check_exist_change_x_tot_list, check_exist_change_x_tot)
            )
            saved_check_exist_change_x_ones_list = np.concatenate(
                (saved_check_exist_change_x_ones_list, check_exist_change_x_ones)
            )
        logger.info(f"   check_exist_step1")
        how_loop = int(saved_check_exist_num_list.shape[0])
        mem_size_per_1data = 1  # Bite
        mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
            indexes = check_exist_step1_7(
                saved_similar_num_list, saved_check_exist_num_list, saved_check_exist_change_x_tot_list, progress
            )
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")
        saved_check_exist_num_list = saved_check_exist_num_list[indexes]
        saved_check_exist_eq_list = saved_check_exist_eq_list[indexes]
        saved_check_exist_id_list = saved_check_exist_id_list[indexes]
        saved_check_exist_change_x_tot_list = saved_check_exist_change_x_tot_list[indexes]
        saved_check_exist_change_x_ones_list = saved_check_exist_change_x_ones_list[indexes]

        logger.info(f"   check_exist_step2")
        how_loop = int(saved_check_exist_num_list.shape[0])
        mem_size_per_1data = 1  # Bite
        mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop - 1, header=header) as progress:
            same = check_exist_step2_7(saved_check_exist_num_list, progress)
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")

        logger.info(f"   check_exist_step3")
        how_loop = 0
        for i in range(np.max(same) + 1 if same.shape[0] != 0 else 0):
            how_loop += int(np.sum(same == i) - 1)
        mem_size_per_1data = 1 + 8  # Bite
        mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, loop={how_loop})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
            indexes = check_exist_step3_7(
                max_op,
                random_x,
                same,
                saved_check_exist_eq_list,
                saved_check_exist_change_x_tot_list,
                progress,
            )
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")
        saved_check_exist_id_list = saved_check_exist_id_list[indexes]
        saved_check_exist_change_x_ones_list = saved_check_exist_change_x_ones_list[indexes]

        saved_base_eq_id_list = np.concatenate((saved_base_eq_id_list, saved_check_exist_id_list))
        saved_check_change_x_ones = np.concatenate((saved_check_change_x_ones, saved_check_exist_change_x_ones_list))
        make_final_cache_7(saved_base_eq_id_list, saved_check_change_x_ones, logger)


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
    tot_mask_x = nb_permutations(np.arange(max_op + 1), max_op + 1)
    head_before_similar_num_list = before_similar_num_list[:, 0, 0].copy()
    dict_change_x_pattern, dict_max_loop = make_dict_change_x_pattern(max_op)
    before_check_change_x_dict = cache_check_change_x_load(max_op - 1)
    dict_mask_x = make_dict_mask_x(max_op + 1)
    base_dict = cache_load(max_op - 1)
    base_eq_arr1 = base_dict[n_op1]
    base_eq_arr2 = base_dict[n_op2]
    len_base_eq1 = base_eq_arr1.shape[1]
    len_base_eq2 = base_eq_arr2.shape[1]
    before_check_change_x_list1 = before_check_change_x_dict[n_op1]
    before_check_change_x_list2 = before_check_change_x_dict[n_op2]
    _, loop_per_threads = loop_count(max_op, n_op1, num_threads)
    head_saved_similar_num_list = saved_similar_num_list[:, 0, 0].copy()
    save_similar_num_list = np.full(
        (num_threads, loop_per_threads, 2, random_x.shape[1]),
        int_nan,
        dtype="float64",
    )
    head_save_similar_num_list = np.full((num_threads, loop_per_threads), int_nan, dtype="float64")
    TF_list = np.zeros((num_threads, loop_per_threads), dtype="bool")
    check_exist_TF = np.zeros((num_threads, loop_per_threads), dtype="bool")
    save_need_calc_list = np.zeros((num_threads, loop_per_threads), dtype="bool")
    save_equation_list = np.full((num_threads, loop_per_threads, 2 * max_op + 1), int_nan, dtype="int8")
    save_base_eq_id_list = np.full((num_threads, loop_per_threads, 5), int_nan, dtype="int64")
    save_check_change_x_tot = np.full(
        (num_threads, loop_per_threads, (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    save_check_change_x_ones = np.full(
        (num_threads, loop_per_threads, (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    ops = np.array([-1, -2, -3, -4, -5])
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), int_nan, dtype="int8")
        cache_for_mask_x = np.zeros((tot_mask_x.shape[0]), dtype="bool")
        added_check_change_x = np.full(((max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8")
        reduce_check_change_x = np.full(((max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8")
        arange = np.arange(tot_mask_x.shape[0])
        counter = 0
        len_base_eq_arr1 = base_eq_arr1.shape[0]
        len_base_eq_arr2 = base_eq_arr2.shape[0]
        loop = len_base_eq_arr1 * len_base_eq_arr2
        for i in range(thread_id, loop, num_threads):
            id1 = i % len_base_eq_arr1
            i //= len_base_eq_arr1
            id2 = i % len_base_eq_arr2
            front_equation = base_eq_arr1[id1]
            equation[:len_base_eq1] = front_equation
            flont_x_max = np.max(front_equation)
            base_back_equation = base_eq_arr2[id2]
            base_back_x_max = np.max(base_back_equation)
            for merge_op in ops:
                equation[len_base_eq1 + len_base_eq2] = merge_op
                for number in range(dict_max_loop[base_back_x_max, flont_x_max]):
                    is_save, is_calc = True, True
                    succession = True
                    back_change_pattern = dict_change_x_pattern[base_back_x_max][number]
                    equation[len_base_eq1 : len_base_eq1 + len_base_eq2] = make_eq(
                        base_back_equation, back_change_pattern
                    )
                    if find_min_x_max(equation, random_x, random_for_find_min_x_max):
                        is_save, is_calc = False, False
                    if is_save:
                        save_similar_num = nb_calc_RPN(random_x, equation)
                        base_similar_num = save_similar_num[1].copy()
                        if save_similar_num[0, 0] == int_nan:  # any_isinf, all_const
                            if count_True(equation, 5, 0) != 0:  # likely (a-a), not (1+1) ,  lambda x: x > border
                                is_save, is_calc = False, False
                            elif is_all_zero(save_similar_num[1], atol=0):  # all_zero
                                is_save, is_calc = False, False
                            else:  # (1+1)
                                is_calc = False
                        # if is_save:
                        eq_x_max = np.max(equation)
                        mask_x = dict_mask_x[eq_x_max]
                        cache_for_mask_x[0] = False
                        for k in range(1, mask_x.shape[0]):
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
                            same = False
                            if isclose(base_similar_num[0], similar_num[1, 0]):
                                if isclose_arr(base_similar_num, similar_num[1]):
                                    same = True
                            cache_for_mask_x[k] = same
                            if save_similar_num[0, 0] > similar_num[0, 0]:  # min
                                save_similar_num[0] = similar_num[0]
                            if save_similar_num[1, 0] > similar_num[1, 0]:  # min
                                save_similar_num[1] = similar_num[1]
                        cache_for_mask_x[mask_x.shape[0] :] = False
                        check_change_x = make_check_change_x(mask_x[arange[cache_for_mask_x]])
                        before_check_change_x1 = before_check_change_x_list1[id1]
                        len_before_check_change_x1 = count_True(
                            before_check_change_x1[:, 0], 2, int_nan
                        )  # 2 -> lambda x: x != border
                        before_check_change_x2 = before_check_change_x_list2[id2]
                        changed_before_check_change_x2 = make_eq(before_check_change_x2, back_change_pattern)
                        len_before_check_change_x2 = count_True(
                            before_check_change_x2[:, 0], 2, int_nan
                        )  # 2 -> lambda x: x != border
                        for k in range(len_before_check_change_x1):
                            for t in range(len_before_check_change_x2):
                                if before_check_change_x1[k, 0] == changed_before_check_change_x2[t, 1]:
                                    if before_check_change_x1[k, 1] == changed_before_check_change_x2[t, 0]:
                                        is_save = False
                                        break
                    if is_save:
                        last_added, last_reduced = 0, 0
                        for k in range(len_before_check_change_x1):
                            found = False
                            for t in range(check_change_x.shape[0]):
                                if before_check_change_x1[k, 0] == check_change_x[t, 0]:
                                    if before_check_change_x1[k, 1] == check_change_x[t, 1]:
                                        found = True
                                        break
                            if not found:
                                added_check_change_x[last_added] = before_check_change_x1[k]
                                last_added += 1
                                succession = False
                        for k in range(len_before_check_change_x2):
                            found = False
                            for t in range(check_change_x.shape[0]):
                                if changed_before_check_change_x2[k, 0] == check_change_x[t, 0]:
                                    if changed_before_check_change_x2[k, 1] == check_change_x[t, 1]:
                                        found = True
                                        break
                            if not found:
                                added_check_change_x[last_added] = changed_before_check_change_x2[k]
                                last_added += 1
                                succession = False
                        if not succession:
                            added_check_change_x[last_added:] = int_nan

                        len_check_change_x = count_True(check_change_x[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
                        for k in range(len_check_change_x):
                            found = False
                            for t in range(len_before_check_change_x1):
                                if before_check_change_x1[t, 0] == check_change_x[k, 0]:
                                    if before_check_change_x1[t, 1] == check_change_x[k, 1]:
                                        found = True
                                        break
                            if not found:
                                for t in range(len_before_check_change_x2):
                                    if changed_before_check_change_x2[t, 0] == check_change_x[k, 0]:
                                        if changed_before_check_change_x2[t, 1] == check_change_x[k, 1]:
                                            found = True
                                            break
                            if not found:
                                reduce_check_change_x[last_reduced] = check_change_x[k]
                                last_reduced += 1
                        reduce_check_change_x[last_reduced:] = int_nan
                    if is_save:
                        if max_op + 1 != eq_x_max:  # except when using x of (number of operators + 1) type
                            for j in range(head_before_similar_num_list.shape[0]):
                                if isclose(save_similar_num[0, 0], head_before_similar_num_list[j]):
                                    if isclose_arr(
                                        save_similar_num[0],
                                        before_similar_num_list[j, 0],
                                    ):
                                        is_calc = False
                                        if isclose_arr(
                                            save_similar_num[1],
                                            before_similar_num_list[j, 1],
                                        ):
                                            is_save = False
                                            break
                                elif save_similar_num[0, 0] < head_before_similar_num_list[j]:
                                    break
                    if is_save:
                        if not succession:
                            check_exist_TF[thread_id, counter] = True
                            save_similar_num_list[thread_id, counter] = save_similar_num
                            save_equation_list[thread_id, counter] = equation
                            save_base_eq_id_list[thread_id, counter, 0] = n_op1
                            save_base_eq_id_list[thread_id, counter, 1] = id1
                            save_base_eq_id_list[thread_id, counter, 2] = id2
                            save_base_eq_id_list[thread_id, counter, 3] = make_change_x_id(
                                back_change_pattern, max_op + 1
                            )
                            save_base_eq_id_list[thread_id, counter, 4] = merge_op
                            save_check_change_x_tot[thread_id, counter] = added_check_change_x
                            save_check_change_x_ones[thread_id, counter] = reduce_check_change_x
                            is_save = False
                    if is_save:
                        for j in range(head_saved_similar_num_list.shape[0]):
                            if isclose(save_similar_num[0, 0], head_saved_similar_num_list[j]):
                                if isclose_arr(
                                    save_similar_num[0],
                                    saved_similar_num_list[j, 0],
                                ):
                                    is_calc = False
                                    if isclose_arr(
                                        save_similar_num[1],
                                        saved_similar_num_list[j, 1],
                                    ):
                                        is_save = False
                                        break
                            elif save_similar_num[0, 0] < head_saved_similar_num_list[j]:
                                break
                    if is_save:
                        for j in range(counter):
                            if head_save_similar_num_list[thread_id, j] != int_nan:
                                if isclose(save_similar_num[0, 0], head_save_similar_num_list[thread_id, j]):
                                    if isclose_arr(
                                        save_similar_num[0],
                                        save_similar_num_list[thread_id, j, 0],
                                    ):
                                        is_calc = False
                                        if isclose_arr(
                                            save_similar_num[1],
                                            save_similar_num_list[thread_id, j, 1],
                                        ):
                                            is_save = False
                                            break
                    if is_save:
                        TF_list[thread_id, counter] = True
                        save_similar_num_list[thread_id, counter] = save_similar_num
                        head_save_similar_num_list[thread_id, counter] = save_similar_num[0, 0]
                        save_need_calc_list[thread_id, counter] = is_calc
                        save_equation_list[thread_id, counter] = equation
                        save_base_eq_id_list[thread_id, counter, 0] = n_op1
                        save_base_eq_id_list[thread_id, counter, 1] = id1
                        save_base_eq_id_list[thread_id, counter, 2] = id2
                        save_base_eq_id_list[thread_id, counter, 3] = make_change_x_id(back_change_pattern, max_op + 1)
                        save_base_eq_id_list[thread_id, counter, 4] = merge_op
                        save_check_change_x_tot[thread_id, counter, : check_change_x.shape[0]] = check_change_x
                        save_check_change_x_ones[thread_id, counter] = reduce_check_change_x
                    counter += 1
                    progress_proxy.update(1)
    return (
        TF_list,
        save_similar_num_list,
        save_need_calc_list,
        save_equation_list,
        save_base_eq_id_list,
        save_check_change_x_tot,
        save_check_change_x_ones,
        check_exist_TF,
    )


@njit(parallel=True, error_model="numpy")
def dim_reduction(TF_list, similar_num_list, need_calc_list, progress_proxy):
    int_nan = -100
    num_threads = similar_num_list.shape[0]
    threads_last_index = np.array([np.sum(TF_list[i]) for i in range(num_threads)])
    sort_index = np.argsort(threads_last_index)
    return_similar_num_list = np.full((np.sum(TF_list), 2, similar_num_list.shape[3]), int_nan, dtype="float64")
    return_similar_num_list[: np.sum(TF_list[sort_index[0]])] = similar_num_list[sort_index[0], TF_list[sort_index[0]]]
    head_return_similar_num_list = np.full((np.sum(TF_list)), int_nan, dtype="float64")
    head_similar_num_list = similar_num_list[:, :, 0, 0].copy()
    head_return_similar_num_list[: np.sum(TF_list[sort_index[0]])] = head_similar_num_list[
        sort_index[0], TF_list[sort_index[0]]
    ]
    last_index = np.sum(TF_list[sort_index[0]])
    for target_index in sort_index[1:]:
        true_index = np.random.permutation(np.arange(TF_list.shape[1])[TF_list[target_index]])
        for thread_id in prange(num_threads):
            true_index_thread = true_index[thread_id::num_threads].copy()
            for i in true_index_thread:
                head_target = similar_num_list[target_index, i, 0, 0]
                for j in range(last_index):
                    if isclose(head_target, head_return_similar_num_list[j]):
                        if isclose_arr(
                            similar_num_list[target_index, i, 0],
                            return_similar_num_list[j, 0],
                        ):
                            need_calc_list[target_index, i] = False
                            if isclose_arr(
                                similar_num_list[target_index, i, 1],
                                return_similar_num_list[j, 1],
                            ):
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
    return TF_list, need_calc_list


@njit(parallel=True, error_model="numpy")
def dim_reduction_info(
    TF_list, similar_num_list, need_calc_list, equation_list, base_eq_id_list, check_change_x_tot, check_change_x_ones
):
    int_nan = -100
    sum_TF = np.sum(TF_list)
    num_threads = TF_list.shape[0]
    return_similar_num_list = np.full((sum_TF, 2, similar_num_list.shape[3]), int_nan, dtype="float64")
    return_equation_list = np.full((sum_TF, equation_list.shape[2]), int_nan, dtype="int8")
    return_base_eq_id_list = np.full((sum_TF, 5), int_nan, dtype="int64")
    return_need_calc_list = np.zeros((sum_TF), dtype="bool")
    return_check_change_x_tot = np.full((sum_TF, check_change_x_tot.shape[2], 2), int_nan, dtype="int8")
    return_check_change_x_ones = np.full((sum_TF, check_change_x_ones.shape[2], 2), int_nan, dtype="int8")
    last_index = 0
    for thread_id in range(num_threads):
        indexes = np.arange(TF_list.shape[1])[TF_list[thread_id]]
        return_similar_num_list[last_index : last_index + indexes.shape[0]] = similar_num_list[thread_id, indexes]
        return_equation_list[last_index : last_index + indexes.shape[0]] = equation_list[thread_id, indexes]
        return_base_eq_id_list[last_index : last_index + indexes.shape[0]] = base_eq_id_list[thread_id, indexes]
        return_need_calc_list[last_index : last_index + indexes.shape[0]] = need_calc_list[thread_id, indexes]
        return_check_change_x_tot[last_index : last_index + indexes.shape[0]] = check_change_x_tot[thread_id, indexes]
        return_check_change_x_ones[last_index : last_index + indexes.shape[0]] = check_change_x_ones[thread_id, indexes]
        last_index += indexes.shape[0]
    return (
        return_equation_list,
        return_similar_num_list,
        return_need_calc_list,
        return_base_eq_id_list,
        return_check_change_x_tot,
        return_check_change_x_ones,
    )


@njit(parallel=True, error_model="numpy")
def make_check_exist_info(
    check_exist_TF, similar_num_list, equation_list, base_eq_id_list, check_change_x_tot, check_change_x_ones
):
    int_nan = -100
    num_threads = check_exist_TF.shape[0]
    sum_check_exist = np.sum(check_exist_TF)
    return_check_exist_num = np.full((sum_check_exist, 2, similar_num_list.shape[3]), int_nan, dtype="float64")
    return_check_exist_eq = np.full((sum_check_exist, equation_list.shape[2]), int_nan, dtype="int8")
    return_check_exist_id = np.full((sum_check_exist, 5), int_nan, dtype="int64")
    return_check_exist_change_x_tot = np.full((sum_check_exist, check_change_x_tot.shape[2], 2), int_nan, dtype="int8")
    return_check_exist_change_x_ones = np.full(
        (sum_check_exist, check_change_x_ones.shape[2], 2), int_nan, dtype="int8"
    )
    last_index = 0
    for thread_id in range(num_threads):
        indexes = np.arange(check_exist_TF.shape[1])[check_exist_TF[thread_id]]
        return_check_exist_num[last_index : last_index + indexes.shape[0]] = similar_num_list[thread_id, indexes]
        return_check_exist_eq[last_index : last_index + indexes.shape[0]] = equation_list[thread_id, indexes]
        return_check_exist_id[last_index : last_index + indexes.shape[0]] = base_eq_id_list[thread_id, indexes]
        return_check_exist_change_x_tot[last_index : last_index + indexes.shape[0]] = check_change_x_tot[
            thread_id, indexes
        ]
        return_check_exist_change_x_ones[last_index : last_index + indexes.shape[0]] = check_change_x_ones[
            thread_id, indexes
        ]
        last_index += indexes.shape[0]
    return (
        return_check_exist_num[:last_index],
        return_check_exist_eq[:last_index],
        return_check_exist_id[:last_index],
        return_check_exist_change_x_tot[:last_index],
        return_check_exist_change_x_ones[:last_index],
    )


@njit(parallel=True, error_model="numpy")
def check_exist_step1(similar_num_list, check_exist_num_arr, check_exist_change_x, progress_proxy):
    int_nan = -100
    head_saved_similar_num_list = similar_num_list[:, 0, 0].copy()
    len_return = check_exist_num_arr.shape[0]
    check_exist_need_calc = np.zeros((len_return), dtype="bool")
    TF = np.zeros((len_return), dtype="bool")
    for i in prange(check_exist_num_arr.shape[0]):
        is_calc, is_save = True, True
        for j in range(head_saved_similar_num_list.shape[0]):
            if isclose(check_exist_num_arr[i, 0, 0], head_saved_similar_num_list[j]):
                if isclose_arr(check_exist_num_arr[i, 0], similar_num_list[j, 0]):
                    is_calc = False
                    if isclose_arr(check_exist_num_arr[i, 1], similar_num_list[j, 1]):
                        is_save = False
                        break
            elif check_exist_num_arr[i, 0, 0] < head_saved_similar_num_list[j]:
                break
        check_exist_need_calc[i] = is_calc
        TF[i] = is_save
        progress_proxy.update(1)
    index = np.arange(len_return)[TF]
    count_change_x = np.array(
        [count_True(check_exist_change_x[i, :, 0], 2, int_nan) for i in index]
    )  # 2 -> lambda x: x != border
    index = index[np.argsort(count_change_x)]
    return check_exist_need_calc, index


@njit(parallel=True, error_model="numpy")
def check_exist_step2(check_exist_num_arr, progress_proxy):
    int_nan = -100
    n_check_exist = check_exist_num_arr.shape[0]
    same = np.full((n_check_exist), int_nan, dtype="int8")
    count = 0
    for i in range(1, n_check_exist):
        if same[i] == int_nan:
            for j in prange(i + 1, n_check_exist):
                if same[j] == int_nan:
                    if isclose(check_exist_num_arr[i, 0, 0], check_exist_num_arr[j, 0, 0]):
                        if isclose_arr(check_exist_num_arr[i, 0], check_exist_num_arr[j, 0]):
                            same[j] = count
            same[i] = count
            count += 1
        progress_proxy.update(1)
    return same


@njit(parallel=True, error_model="numpy")
def check_exist_step3(max_op, x, same, check_exist_eq, check_exist_need_calc, check_exist_change_x, progress_proxy):
    int_nan = -100
    dict_mask_x = make_dict_mask_x(max_op + 1)
    n_check_exist = check_exist_eq.shape[0]
    TF = np.ones((n_check_exist), dtype="bool")
    arange = np.arange(n_check_exist)
    count = np.max(same) + 1 if same.shape[0] != 0 else 0
    for n in prange(count):
        index = arange[same == n]
        i = index[0]
        for j in index[1:]:
            target_equation = check_exist_eq[i]
            target_mask_x = dict_mask_x[np.max(target_equation)]
            equation = check_exist_eq[j]
            mask_x = dict_mask_x[np.max(equation)]
            is_save = True
            for k in range(target_mask_x.shape[0]):
                if is_save:
                    need_calc = True
                    for l in range(count_True(check_exist_change_x[i, :, 0], 2, int_nan)):  # 2 -> lambda x: x != border
                        a = target_mask_x[k, check_exist_change_x[i, l]]
                        if a[0] > a[1]:
                            need_calc = False
                            break
                    if need_calc:
                        target_num_arr = nb_calc_RPN(x[target_mask_x[k]], target_equation)
                        for l in range(mask_x.shape[0]):
                            if is_save:
                                need_calc = True
                                len_m = count_True(
                                    check_exist_change_x[j, :, 0], 2, int_nan
                                )  # 2 -> lambda x: x != border
                                for m in range(len_m):
                                    a = mask_x[l, check_exist_change_x[j, m]]
                                    if a[0] > a[1]:
                                        need_calc = False
                                        break
                                if need_calc:
                                    nums = nb_calc_RPN(x[mask_x[l]], equation)
                                    if isclose(target_num_arr[0, 0], nums[0, 0]):
                                        if isclose_arr(target_num_arr[0], nums[0]):
                                            check_exist_need_calc[j] = False
                                            if isclose(target_num_arr[1, 0], nums[1, 0]):
                                                if isclose_arr(target_num_arr[1], nums[1]):
                                                    is_save = False
                                                    TF[j] = False
                                                    break
            progress_proxy.update(1)
    index = np.arange(n_check_exist)[TF]
    return index, check_exist_need_calc


# max_op = 7


@njit(parallel=True, error_model="numpy")
def make_unique_equations_thread_7(
    n_op1,
    num_threads,
    random_x,
    before_similar_num_list,
    saved_similar_num_list,
    progress_proxy,
):
    max_op = 7
    int_nan = -100
    n_op2 = max_op - 1 - n_op1
    random_for_find_min_x_max = np.random.random(random_x.shape[1])
    tot_mask_x = nb_permutations(np.arange(max_op + 1), max_op + 1)
    head_before_similar_num_list = before_similar_num_list[:, 0].copy()
    dict_change_x_pattern, dict_max_loop = make_dict_change_x_pattern(max_op)
    before_check_change_x_dict = cache_check_change_x_load(max_op - 1)
    dict_mask_x = make_dict_mask_x(max_op + 1)
    base_dict = cache_load(max_op - 1)
    base_eq_arr1 = base_dict[n_op1]
    base_eq_arr2 = base_dict[n_op2]
    len_base_eq1 = base_eq_arr1.shape[1]
    len_base_eq2 = base_eq_arr2.shape[1]
    before_check_change_x_list1 = before_check_change_x_dict[n_op1]
    before_check_change_x_list2 = before_check_change_x_dict[n_op2]
    _, loop_per_threads = loop_count(max_op, n_op1, num_threads)
    head_saved_similar_num_list = saved_similar_num_list[:, 0].copy()
    save_similar_num_list = np.full(
        (num_threads, loop_per_threads, random_x.shape[1]),
        int_nan,
        dtype="float64",
    )
    head_save_similar_num_list = np.full((num_threads, loop_per_threads), int_nan, dtype="float64")
    TF_list = np.zeros((num_threads, loop_per_threads), dtype="bool")
    check_exist_TF = np.zeros((num_threads, loop_per_threads), dtype="bool")
    save_equation_list = np.full((num_threads, loop_per_threads, 2 * max_op + 1), int_nan, dtype="int8")
    save_base_eq_id_list = np.full((num_threads, loop_per_threads, 5), int_nan, dtype="int64")
    save_check_change_x_tot = np.full(
        (num_threads, loop_per_threads, (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    save_check_change_x_ones = np.full(
        (num_threads, loop_per_threads, (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    ops = np.array([-1, -2, -3, -4, -5])
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), int_nan, dtype="int8")
        cache_for_mask_x = np.zeros((tot_mask_x.shape[0]), dtype="bool")
        added_check_change_x = np.full(((max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8")
        reduce_check_change_x = np.full(((max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8")
        arange = np.arange(tot_mask_x.shape[0])
        counter = 0
        len_base_eq_arr1 = base_eq_arr1.shape[0]
        len_base_eq_arr2 = base_eq_arr2.shape[0]
        loop = len_base_eq_arr1 * len_base_eq_arr2
        for i in range(thread_id, loop, num_threads):
            id1 = i % len_base_eq_arr1
            i //= len_base_eq_arr1
            id2 = i % len_base_eq_arr2
            front_equation = base_eq_arr1[id1]
            equation[:len_base_eq1] = front_equation
            flont_x_max = np.max(front_equation)
            base_back_equation = base_eq_arr2[id2]
            base_back_x_max = np.max(base_back_equation)
            for merge_op in ops:
                equation[len_base_eq1 + len_base_eq2] = merge_op
                for number in range(dict_max_loop[base_back_x_max, flont_x_max]):
                    is_save = True
                    succession = True
                    back_change_pattern = dict_change_x_pattern[base_back_x_max][number]
                    equation[len_base_eq1 : len_base_eq1 + len_base_eq2] = make_eq(
                        base_back_equation, back_change_pattern
                    )
                    if find_min_x_max(equation, random_x, random_for_find_min_x_max):
                        is_save = False
                    if is_save:
                        save_similar_num = nb_calc_RPN(random_x, equation)[0]
                        base_similar_num = save_similar_num.copy()
                        if save_similar_num[0] == int_nan:  # any_isinf, all_const
                            is_save = False
                        # if is_save:
                        eq_x_max = np.max(equation)
                        mask_x = dict_mask_x[eq_x_max]
                        cache_for_mask_x[0] = False
                        for k in range(1, mask_x.shape[0]):
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)[0]
                            same = False
                            if isclose(base_similar_num[0], similar_num[0]):
                                if isclose_arr(base_similar_num, similar_num):
                                    same = True
                            cache_for_mask_x[k] = same
                            if save_similar_num[0] > similar_num[0]:  # min
                                save_similar_num[:] = similar_num
                        cache_for_mask_x[mask_x.shape[0] :] = False
                        check_change_x = make_check_change_x(mask_x[arange[cache_for_mask_x]])
                        before_check_change_x1 = before_check_change_x_list1[id1]
                        len_before_check_change_x1 = count_True(
                            before_check_change_x1[:, 0], 2, int_nan
                        )  # 2 -> lambda x: x != border
                        before_check_change_x2 = before_check_change_x_list2[id2]
                        changed_before_check_change_x2 = make_eq(before_check_change_x2, back_change_pattern)
                        len_before_check_change_x2 = count_True(
                            before_check_change_x2[:, 0], 2, int_nan
                        )  # 2 -> lambda x: x != border
                        for k in range(len_before_check_change_x1):
                            for t in range(len_before_check_change_x2):
                                if before_check_change_x1[k, 0] == changed_before_check_change_x2[t, 1]:
                                    if before_check_change_x1[k, 1] == changed_before_check_change_x2[t, 0]:
                                        is_save = False
                                        break
                    if is_save:
                        last_added, last_reduced = 0, 0
                        for k in range(len_before_check_change_x1):
                            found = False
                            for t in range(check_change_x.shape[0]):
                                if before_check_change_x1[k, 0] == check_change_x[t, 0]:
                                    if before_check_change_x1[k, 1] == check_change_x[t, 1]:
                                        found = True
                            if not found:
                                added_check_change_x[last_added] = before_check_change_x1[k]
                                last_added += 1
                                succession = False
                        for k in range(len_before_check_change_x2):
                            found = False
                            for t in range(check_change_x.shape[0]):
                                if changed_before_check_change_x2[k, 0] == check_change_x[t, 0]:
                                    if changed_before_check_change_x2[k, 1] == check_change_x[t, 1]:
                                        found = True
                            if not found:
                                added_check_change_x[last_added] = changed_before_check_change_x2[k]
                                last_added += 1
                                succession = False
                        if not succession:
                            added_check_change_x[last_added:] = int_nan

                        len_check_change_x = count_True(check_change_x[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
                        for k in range(len_check_change_x):
                            found = False
                            for t in range(len_before_check_change_x1):
                                if before_check_change_x1[t, 0] == check_change_x[k, 0]:
                                    if before_check_change_x1[t, 1] == check_change_x[k, 1]:
                                        found = True
                                        break
                            if not found:
                                for t in range(len_before_check_change_x2):
                                    if changed_before_check_change_x2[t, 0] == check_change_x[k, 0]:
                                        if changed_before_check_change_x2[t, 1] == check_change_x[k, 1]:
                                            found = True
                                            break
                            if not found:
                                reduce_check_change_x[last_reduced] = check_change_x[k]
                                last_reduced += 1
                        reduce_check_change_x[last_reduced:] = int_nan
                    if is_save:
                        if max_op + 1 != eq_x_max:  # except when using x of (number of operators + 1) type
                            for j in range(head_before_similar_num_list.shape[0]):
                                if isclose(save_similar_num[0], head_before_similar_num_list[j]):
                                    if isclose_arr(save_similar_num, before_similar_num_list[j]):
                                        is_save = False
                                        break
                                elif save_similar_num[0] < head_before_similar_num_list[j]:
                                    break
                    if is_save:
                        if not succession:
                            check_exist_TF[thread_id, counter] = True
                            save_similar_num_list[thread_id, counter] = save_similar_num
                            save_equation_list[thread_id, counter] = equation
                            save_base_eq_id_list[thread_id, counter, 0] = n_op1
                            save_base_eq_id_list[thread_id, counter, 1] = id1
                            save_base_eq_id_list[thread_id, counter, 2] = id2
                            save_base_eq_id_list[thread_id, counter, 3] = make_change_x_id(
                                back_change_pattern, max_op + 1
                            )
                            save_base_eq_id_list[thread_id, counter, 4] = merge_op
                            save_check_change_x_tot[thread_id, counter] = added_check_change_x
                            save_check_change_x_ones[thread_id, counter] = reduce_check_change_x
                            is_save = False
                    if is_save:
                        for j in range(head_saved_similar_num_list.shape[0]):
                            if isclose(save_similar_num[0], head_saved_similar_num_list[j]):
                                if isclose_arr(save_similar_num, saved_similar_num_list[j]):
                                    is_save = False
                                    break
                            elif save_similar_num[0] < head_saved_similar_num_list[j]:
                                break
                    if is_save:
                        for j in range(counter):
                            if head_save_similar_num_list[thread_id, j] != int_nan:
                                if isclose(save_similar_num[0], head_save_similar_num_list[thread_id, j]):
                                    if isclose_arr(save_similar_num, save_similar_num_list[thread_id, j]):
                                        is_save = False
                                        break
                    if is_save:
                        TF_list[thread_id, counter] = True
                        save_similar_num_list[thread_id, counter] = save_similar_num
                        head_save_similar_num_list[thread_id, counter] = save_similar_num[0]
                        save_equation_list[thread_id, counter] = equation
                        save_base_eq_id_list[thread_id, counter, 0] = n_op1
                        save_base_eq_id_list[thread_id, counter, 1] = id1
                        save_base_eq_id_list[thread_id, counter, 2] = id2
                        save_base_eq_id_list[thread_id, counter, 3] = make_change_x_id(back_change_pattern, max_op + 1)
                        save_base_eq_id_list[thread_id, counter, 4] = merge_op
                        save_check_change_x_tot[thread_id, counter, : check_change_x.shape[0]] = check_change_x
                        save_check_change_x_ones[thread_id, counter] = reduce_check_change_x
                    counter += 1
                    progress_proxy.update(1)
    return (
        TF_list,
        save_similar_num_list,
        save_equation_list,
        save_base_eq_id_list,
        save_check_change_x_tot,
        save_check_change_x_ones,
        check_exist_TF,
    )


@njit(parallel=True, error_model="numpy")
def dim_reduction_7(TF_list, similar_num_list, progress_proxy):
    int_nan = -100
    num_threads = similar_num_list.shape[0]
    threads_last_index = np.array([np.sum(TF_list[i]) for i in range(num_threads)])
    sort_index = np.argsort(threads_last_index)
    return_similar_num_list = np.full((np.sum(TF_list), similar_num_list.shape[2]), int_nan, dtype="float64")
    return_similar_num_list[: np.sum(TF_list[sort_index[0]])] = similar_num_list[sort_index[0], TF_list[sort_index[0]]]
    head_return_similar_num_list = np.full((np.sum(TF_list)), int_nan, dtype="float64")
    head_similar_num_list = similar_num_list[:, :, 0].copy()
    head_return_similar_num_list[: np.sum(TF_list[sort_index[0]])] = head_similar_num_list[
        sort_index[0], TF_list[sort_index[0]]
    ]
    last_index = np.sum(TF_list[sort_index[0]])
    for target_index in sort_index[1:]:
        true_index = np.random.permutation(np.arange(TF_list.shape[1])[TF_list[target_index]])
        for thread_id in prange(num_threads):
            true_index_thread = true_index[thread_id::num_threads].copy()
            for i in true_index_thread:
                head_target = similar_num_list[target_index, i, 0]
                for j in range(last_index):
                    if isclose(head_target, head_return_similar_num_list[j]):
                        if isclose_arr(similar_num_list[target_index, i], return_similar_num_list[j]):
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
def dim_reduction_info_7(
    TF_list, similar_num_list, equation_list, base_eq_id_list, check_change_x_tot, check_change_x_ones
):
    int_nan = -100
    sum_TF = np.sum(TF_list)
    num_threads = TF_list.shape[0]
    return_similar_num_list = np.full((sum_TF, similar_num_list.shape[2]), int_nan, dtype="float64")
    return_equation_list = np.full((sum_TF, equation_list.shape[2]), int_nan, dtype="int8")
    return_base_eq_id_list = np.full((sum_TF, 5), int_nan, dtype="int64")
    return_check_change_x_tot = np.full((sum_TF, check_change_x_tot.shape[2], 2), int_nan, dtype="int8")
    return_check_change_x_ones = np.full((sum_TF, check_change_x_ones.shape[2], 2), int_nan, dtype="int8")
    last_index = 0
    for thread_id in range(num_threads):
        indexes = np.arange(TF_list.shape[1])[TF_list[thread_id]]
        return_similar_num_list[last_index : last_index + indexes.shape[0]] = similar_num_list[thread_id, indexes]
        return_equation_list[last_index : last_index + indexes.shape[0]] = equation_list[thread_id, indexes]
        return_base_eq_id_list[last_index : last_index + indexes.shape[0]] = base_eq_id_list[thread_id, indexes]
        return_check_change_x_tot[last_index : last_index + indexes.shape[0]] = check_change_x_tot[thread_id, indexes]
        return_check_change_x_ones[last_index : last_index + indexes.shape[0]] = check_change_x_ones[thread_id, indexes]
        last_index += indexes.shape[0]
    return (
        return_equation_list,
        return_similar_num_list,
        return_base_eq_id_list,
        return_check_change_x_tot,
        return_check_change_x_ones,
    )


@njit(parallel=True, error_model="numpy")
def make_check_exist_info_7(
    check_exist_TF, similar_num_list, equation_list, base_eq_id_list, check_change_x_tot, check_change_x_ones
):
    int_nan = -100
    num_threads = check_exist_TF.shape[0]
    sum_check_exist = np.sum(check_exist_TF)
    return_check_exist_num = np.full((sum_check_exist, similar_num_list.shape[2]), int_nan, dtype="float64")
    return_check_exist_eq = np.full((sum_check_exist, equation_list.shape[2]), int_nan, dtype="int8")
    return_check_exist_id = np.full((sum_check_exist, 5), int_nan, dtype="int64")
    return_check_exist_change_x_tot = np.full((sum_check_exist, check_change_x_tot.shape[2], 2), int_nan, dtype="int8")
    return_check_exist_change_x_ones = np.full(
        (sum_check_exist, check_change_x_ones.shape[2], 2), int_nan, dtype="int8"
    )
    last_index = 0
    for thread_id in range(num_threads):
        indexes = np.arange(check_exist_TF.shape[1])[check_exist_TF[thread_id]]
        return_check_exist_num[last_index : last_index + indexes.shape[0]] = similar_num_list[thread_id, indexes]
        return_check_exist_eq[last_index : last_index + indexes.shape[0]] = equation_list[thread_id, indexes]
        return_check_exist_id[last_index : last_index + indexes.shape[0]] = base_eq_id_list[thread_id, indexes]
        return_check_exist_change_x_tot[last_index : last_index + indexes.shape[0]] = check_change_x_tot[
            thread_id, indexes
        ]
        return_check_exist_change_x_ones[last_index : last_index + indexes.shape[0]] = check_change_x_ones[
            thread_id, indexes
        ]
        last_index += indexes.shape[0]
    return (
        return_check_exist_num[:last_index],
        return_check_exist_eq[:last_index],
        return_check_exist_id[:last_index],
        return_check_exist_change_x_tot[:last_index],
        return_check_exist_change_x_ones[:last_index],
    )


@njit(parallel=True, error_model="numpy")
def check_exist_step1_7(similar_num_list, check_exist_num_arr, check_exist_change_x, progress_proxy):
    int_nan = -100
    head_saved_similar_num_list = similar_num_list[:, 0].copy()
    len_return = check_exist_num_arr.shape[0]
    TF = np.zeros((len_return), dtype="bool")
    for i in prange(len_return):
        is_save = True
        for j in range(head_saved_similar_num_list.shape[0]):
            if isclose(check_exist_num_arr[i, 0], head_saved_similar_num_list[j]):
                if isclose_arr(check_exist_num_arr[i], similar_num_list[j]):
                    is_save = False
                    break
            elif check_exist_num_arr[i, 0] < head_saved_similar_num_list[j]:
                break
        TF[i] = is_save
        progress_proxy.update(1)
    index = np.arange(len_return)[TF]
    count_change_x = np.array(
        [count_True(check_exist_change_x[i, :, 0], 2, int_nan) for i in index]
    )  # 2 -> lambda x: x != border
    index = index[np.argsort(count_change_x)]
    return index


@njit(parallel=True, error_model="numpy")
def check_exist_step2_7(check_exist_num_arr, progress_proxy):
    int_nan = -100
    n_check_exist = check_exist_num_arr.shape[0]
    same = np.full((n_check_exist), int_nan, dtype="int8")
    count = 0
    for i in range(1, n_check_exist):
        if same[i] == int_nan:
            for j in prange(i + 1, n_check_exist):
                if same[j] == int_nan:
                    if isclose(check_exist_num_arr[i, 0], check_exist_num_arr[j, 0]):
                        if isclose_arr(check_exist_num_arr[i], check_exist_num_arr[j]):
                            same[j] = count
            same[i] = count
            count += 1
        progress_proxy.update(1)
    return same


@njit(parallel=True, error_model="numpy")
def check_exist_step3_7(max_op, x, same, check_exist_eq, check_exist_change_x, progress_proxy):
    int_nan = -100
    dict_mask_x = make_dict_mask_x(max_op + 1)
    n_check_exist = check_exist_eq.shape[0]
    TF = np.ones((n_check_exist), dtype="bool")
    arange = np.arange(n_check_exist)
    count = np.max(same) + 1 if same.shape[0] != 0 else 0
    for n in prange(count):
        index = arange[same == n]
        i = index[0]
        for j in index[1:]:
            target_equation = check_exist_eq[i]
            target_mask_x = dict_mask_x[np.max(target_equation)]
            equation = check_exist_eq[j]
            mask_x = dict_mask_x[np.max(equation)]
            is_save = True
            for k in range(target_mask_x.shape[0]):
                if is_save:
                    need_calc = True
                    for l in range(count_True(check_exist_change_x[i, :, 0], 2, int_nan)):  # 2 -> lambda x: x != border
                        a = target_mask_x[k, check_exist_change_x[i, l]]
                        if a[0] > a[1]:
                            need_calc = False
                            break
                    if need_calc:
                        target_num_arr = nb_calc_RPN(x[target_mask_x[k]], target_equation)[0]
                        for l in range(mask_x.shape[0]):
                            if is_save:
                                need_calc = True
                                len_m = count_True(
                                    check_exist_change_x[j, :, 0], 2, int_nan
                                )  # 2 -> lambda x: x != border
                                for m in range(len_m):
                                    a = mask_x[l, check_exist_change_x[j, m]]
                                    if a[0] > a[1]:
                                        need_calc = False
                                        break
                                if need_calc:
                                    nums = nb_calc_RPN(x[mask_x[l]], equation)[0]
                                    if isclose(target_num_arr[0], nums[0]):
                                        if isclose_arr(target_num_arr, nums):
                                            is_save = False
                                            TF[j] = False
                                            break
            progress_proxy.update(1)
    index = np.arange(n_check_exist)[TF]
    return index
