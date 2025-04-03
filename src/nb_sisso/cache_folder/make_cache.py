import jedi
import numpy as np
import numba, os, datetime, logging, multiprocessing, gc
from numba import njit, prange, set_num_threads, objmode, get_num_threads

from utils_for_make_cache import *
from log_progress import loop_log


def main(max_op, num_threads, len_x=40, log_interval=300, verbose=True, logger=None, seed=None):
    int_nan = -100
    if 30 > len_x:
        raise ValueError("len_x need >= 30")
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
            _format = "%(asctime)s : %(message)s"
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

    logger.info("making random_x")
    if seed is None:
        seed = np.random.randint(100000)
    loop = 100000
    upper = 0.5
    lower = 1.5
    logger.info(f"   seed = {seed}, loop = {loop}, upper = {upper}, lower = {lower}")
    save_corrcoef, random_x = make_random_x(max_op, len_x, seed=seed, loop=loop, upper=upper, lower=lower)
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
    logger.info(f"   using Memory size =  {str_using_mem()}")
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
    one_eq_calc_check_change_x,
    logger,
):
    # base_eq_id_list = merge_op,n_op1,id1,id2,changed_back_eq_id
    int_nan = -100
    base = np.load("arr_len.npy")
    arr_len = np.zeros((max(base.shape[0], max_op + 1)), dtype="int64")
    arr_len[: base.shape[0]] = base
    arr_len[max_op] = base_eq_id_list.shape[0]
    logger.info(f"   tot             , {[int(i) for i in base]} -> {[int(i) for i in arr_len]}")
    np.save("arr_len", arr_len)
    return_np_index_to_num = dict()
    for n_op1 in range(max_op):
        n_op2 = max_op - 1 - n_op1
        n = 0
        max_num = make_change_x_id(np.arange(n_op1 + 2, max_op + 2), max_op + 1)
        dict_num_to_index = np.full((max_num + 1), int_nan, dtype="int64")
        dict_change_x_pattern, dict_max_loop = make_dict_change_x_pattern(max_op)
        for flont_x_max in range(n_op1 + 2):
            for base_back_x_max in range(n_op2 + 2):
                for number in range(dict_max_loop[base_back_x_max, flont_x_max]):
                    back_change_pattern = dict_change_x_pattern[base_back_x_max][number]
                    change_x_id = make_change_x_id(back_change_pattern, max_op + 1)
                    if dict_num_to_index[change_x_id] == int_nan:
                        dict_num_to_index[change_x_id] = n
                        n += 1
        return_np_index_to_num[n_op1] = dict_num_to_index

    np.savez(
        f"num_to_index_{max_op}",
        **{str(n_op1): return_np_index_to_num[n_op1] for n_op1 in range(max_op)},
    )
    # base_eq_id_list = n_op1,id1,id2,change_x_id,merge_op
    np.savez(f"cache_{max_op}", base_eq_id_list)

    np.savez(f"need_calc_{max_op}", need_calc_list)

    len_check_change_x_tot = 0
    for j in range(check_change_x_tot.shape[1]):
        if np.any(check_change_x_tot[:, j, 0] != int_nan):
            len_check_change_x_tot = j + 1
    np.savez(f"check_change_x_tot_{max_op}", check_change_x_tot[:, :len_check_change_x_tot])

    len_one_eq_calc_check_change_x = 0
    for j in range(one_eq_calc_check_change_x.shape[1]):
        if np.any(one_eq_calc_check_change_x[:, j, 0] != int_nan):
            len_one_eq_calc_check_change_x = j + 1
    np.savez(f"one_eq_calc_check_change_x_{max_op}", one_eq_calc_check_change_x[:, :len_one_eq_calc_check_change_x])

    len_check_change_x_ones = 0
    for j in range(check_change_x_ones.shape[1]):
        if np.any(check_change_x_ones[:, j, 0] != int_nan):
            len_check_change_x_ones = j + 1
    np.savez(f"check_change_x_ones_{max_op}", check_change_x_ones[:, :len_check_change_x_ones])
    # savez_compressed
    np.save(f"operator_{max_op}", equation_list)


def make_final_cache_7(
    equation_list,
    base_eq_id_list,
    check_change_x_tot,
    check_change_x_ones,
    logger,
):
    max_op = 7
    # base_eq_id_list = merge_op,n_op1,id1,id2,changed_back_eq_id
    int_nan = -100
    base = np.load("arr_len.npy")
    arr_len = np.zeros((max(base.shape[0], max_op + 1)), dtype="int64")
    arr_len[: base.shape[0]] = base
    arr_len[max_op] = base_eq_id_list.shape[0]
    logger.info(f"   tot             , {[int(i) for i in base]} -> {[int(i) for i in arr_len]}")
    np.save("arr_len", arr_len)
    return_np_index_to_num = dict()
    for n_op1 in range(max_op):
        n_op2 = max_op - 1 - n_op1
        n = 0
        max_num = make_change_x_id(np.arange(n_op1 + 2, max_op + 2), max_op + 1)
        dict_num_to_index = np.full((max_num + 1), int_nan, dtype="int64")
        dict_change_x_pattern, dict_max_loop = make_dict_change_x_pattern(max_op)
        for flont_x_max in range(n_op1 + 2):
            for base_back_x_max in range(n_op2 + 2):
                for number in range(dict_max_loop[base_back_x_max, flont_x_max]):
                    back_change_pattern = dict_change_x_pattern[base_back_x_max][number]
                    change_x_id = make_change_x_id(back_change_pattern, max_op + 1)
                    if dict_num_to_index[change_x_id] == int_nan:
                        dict_num_to_index[change_x_id] = n
                        n += 1
        return_np_index_to_num[n_op1] = dict_num_to_index

    np.savez(
        f"num_to_index_{max_op}",
        **{str(n_op1): return_np_index_to_num[n_op1] for n_op1 in range(max_op)},
    )
    # base_eq_id_list = n_op1,id1,id2,change_x_id,merge_op
    np.savez(f"cache_{max_op}", base_eq_id_list)

    np.savez(f"need_calc_{max_op}", np.ones((equation_list.shape[0]), dtype="bool"))

    len_check_change_x_tot = 0
    for j in range(check_change_x_tot.shape[1]):
        if np.any(check_change_x_tot[:, j, 0] != int_nan):
            len_check_change_x_tot = j + 1
    np.savez(f"check_change_x_tot_{max_op}", check_change_x_tot[:, :len_check_change_x_tot])

    one_eq_calc_check_change_x = np.full((equation_list.shape[0], 0, 2), int_nan, dtype="int8")
    np.savez(f"one_eq_calc_check_change_x_{max_op}", one_eq_calc_check_change_x)

    len_check_change_x_ones = 0
    for j in range(check_change_x_ones.shape[1]):
        if np.any(check_change_x_ones[:, j, 0] != int_nan):
            len_check_change_x_ones = j + 1
    np.savez(f"check_change_x_ones_{max_op}", check_change_x_ones[:, :len_check_change_x_ones])
    # savez_compressed
    np.save(f"operator_{max_op}", equation_list)


def make_unique_equations(max_op, num_threads, random_x, before_similar_num_list, log_interval, logger):
    num_threads = int(num_threads)
    int_nan = -100
    saved_equation_list = np.empty((0, 2 * max_op + 1), dtype="int8")
    saved_base_eq_id_list = np.empty((0, 5), dtype="int64")
    saved_check_change_x_tot = np.empty((0, (max_op * (max_op + 1)) // 2, 2), dtype="int8")
    saved_check_change_x_ones = np.empty((0, (max_op * (max_op + 1)) // 2, 2), dtype="int8")
    saved_check_exist_eq_list = np.empty((0, 2 * max_op + 1), dtype="int8")
    saved_check_exist_id_list = np.empty((0, 5), dtype="int64")
    saved_back_change_pattern = np.empty((0, max_op), dtype="int8")
    saved_one_eq_calc_check_change_x = np.empty((0, (max_op * (max_op + 1)) // 2, 2), dtype="int8")
    if max_op != 7:
        saved_need_calc_list = np.empty((0), dtype="bool")
        saved_similar_num_list = np.empty((0, 2, random_x.shape[1]), dtype="float64")
        saved_check_exist_num_list = np.empty((0, 2, random_x.shape[1]), dtype="float64")
        time1, time2 = datetime.datetime.now(), datetime.datetime.now()
        for n_op1 in range(max_op - 1, -1, -1):
            n_op2 = max_op - 1 - n_op1
            logger.info(f"   n_op1 = {n_op1}, n_op2 = {n_op2}")
            how_loop, loop_per_threads = loop_count(max_op, n_op1, num_threads)
            logger.info(f"      make_unique_equations_thread")
            logger.info(f"         using Memory size =  {str_using_mem()}")
            size_arr_for_mem = how_loop
            mem_size_per_1data = (  # Byte
                2 * random_x.shape[1] * 8  # save_similar_num_list
                + 8  # head_save_similar_num_list
                + 1  # TF_list
                + 1  # check_exist_TF
                + 1  # save_need_calc_list
                + (2 * max_op + 1)  # save_equation_list
                + 5 * 8  # save_base_eq_id_list
                + (max_op * (max_op + 1))  # save_check_change_x_tot
                + (max_op * (max_op + 1))  # save_check_change_x_ones
                + (max_op * (max_op + 1))  # save_one_eq_calc_check_change_x
            )
            mem_size = ((mem_size_per_1data * loop_per_threads * num_threads) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
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
                    back_change_pattern,
                    one_eq_calc_check_change_x,
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
            logger.info(f"         using Memory size =  {str_using_mem()}")
            how_loop = int(np.sum(TF_list[1:]))
            mem_size_per_1data = (  # Byte
                2 * similar_num_list.shape[3] * 8  # return_similar_num_list, # 2 * random_x.shape[1] * 8
                + 8  # head_return_similar_num_list
            )
            size_arr_for_mem = np.sum(TF_list)
            mem_size = ((mem_size_per_1data * size_arr_for_mem) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
            )
            time1, time2 = time2, datetime.datetime.now()
            header = "         "
            with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
                dim_reduction(TF_list, similar_num_list, need_calc_list, progress)
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")

            logger.info(f"      make_check_exist_info")
            logger.info(f"         using Memory size =  {str_using_mem()}")
            size_arr_for_mem = int(np.sum(check_exist_TF))
            mem_size_per_1data = (  # Bite
                2 * similar_num_list.shape[3] * 8  # return_check_exist_num, # 2 * random_x.shape[1] * 8
                + equation_list.shape[2]  # return_check_exist_eq, # + (2 * max_op + 1)
                + 5 * 8  # return_check_exist_id
                + back_change_pattern.shape[2]  # return_back_change_pattern
            )
            mem_size = ((mem_size_per_1data * size_arr_for_mem) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
            )
            check_exist_num, check_exist_eq, check_exist_id, check_exist_back_change_pattern = make_check_exist_info(
                check_exist_TF, similar_num_list, equation_list, base_eq_id_list, back_change_pattern
            )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")

            logger.info(f"      dim_reduction_info")
            logger.info(f"         using Memory size =  {str_using_mem()}")
            how_loop = int(np.sum(TF_list))
            size_arr_for_mem = how_loop
            mem_size_per_1data = (  # Byte
                2 * similar_num_list.shape[3] * 8  # return_similar_num_list, # 2 * random_x.shape[1] * 8
                + equation_list.shape[2]  # return_equation_list, + (2 * max_op + 1)
                + 5 * 8  # return_base_eq_id_list
                + 1  # return_need_calc_list
                + check_change_x_tot.shape[2] * 2  # return_check_change_x_tot
                + check_change_x_ones.shape[2] * 2  # return_check_change_x_ones
                + one_eq_calc_check_change_x.shape[2] * 2  # return_one_eq_calc_check_change_x
            )
            mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
            )
            (
                return_equation_list,
                return_similar_num_list,
                return_need_calc_list,
                return_base_eq_id_list,
                return_check_change_x_tot,
                return_check_change_x_ones,
                return_one_eq_calc_check_change_x,
            ) = dim_reduction_info(
                TF_list,
                similar_num_list,
                need_calc_list,
                equation_list,
                base_eq_id_list,
                check_change_x_tot,
                check_change_x_ones,
                one_eq_calc_check_change_x,
            )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")

            logger.info(f"         using Memory size =  {str_using_mem()}")

            saved_equation_list = del_concatenate(saved_equation_list, return_equation_list)
            _saved_similar_num_list = del_concatenate(saved_similar_num_list, return_similar_num_list)
            saved_similar_num_list = _saved_similar_num_list[np.argsort(_saved_similar_num_list[:, 0, 0])].copy()
            saved_need_calc_list = del_concatenate(saved_need_calc_list, return_need_calc_list)
            saved_base_eq_id_list = del_concatenate(saved_base_eq_id_list, return_base_eq_id_list)
            saved_check_change_x_tot = del_concatenate(saved_check_change_x_tot, return_check_change_x_tot)
            saved_check_change_x_ones = del_concatenate(saved_check_change_x_ones, return_check_change_x_ones)
            saved_one_eq_calc_check_change_x = del_concatenate(
                saved_one_eq_calc_check_change_x, return_one_eq_calc_check_change_x
            )
            saved_check_exist_num_list = del_concatenate(saved_check_exist_num_list, check_exist_num)
            saved_check_exist_eq_list = del_concatenate(saved_check_exist_eq_list, check_exist_eq)
            saved_check_exist_id_list = del_concatenate(saved_check_exist_id_list, check_exist_id)
            saved_back_change_pattern = del_concatenate(saved_back_change_pattern, check_exist_back_change_pattern)
            logger.info(f"         using Memory size =  {str_using_mem()}")
            del (
                return_equation_list,
                return_similar_num_list,
                return_need_calc_list,
                return_base_eq_id_list,
                return_check_change_x_tot,
                return_check_change_x_ones,
                return_one_eq_calc_check_change_x,
                _saved_similar_num_list,
                TF_list,
                similar_num_list,
                need_calc_list,
                equation_list,
                base_eq_id_list,
                check_change_x_tot,
                check_change_x_ones,
                one_eq_calc_check_change_x,
                check_exist_num,
                check_exist_eq,
                check_exist_id,
                check_exist_back_change_pattern,
            )
            gc.collect()

            logger.info(f"         using Memory size =  {str_using_mem()}")
            sum_memory_size = (
                saved_equation_list.__sizeof__()
                + saved_similar_num_list.__sizeof__()
                + saved_need_calc_list.__sizeof__()
                + saved_base_eq_id_list.__sizeof__()
                + saved_check_change_x_tot.__sizeof__()
                + saved_check_change_x_ones.__sizeof__()
                + saved_one_eq_calc_check_change_x.__sizeof__()
                + saved_check_exist_num_list.__sizeof__()
                + saved_check_exist_eq_list.__sizeof__()
                + saved_check_exist_id_list.__sizeof__()
                + saved_back_change_pattern.__sizeof__()
            )
            logger.info(f"         sum_memory_size =  {sum_memory_size//100000/10} MB")

        logger.info(f"   check_exist_step1")
        logger.info(f"      using Memory size =  {str_using_mem()}")
        how_loop = int(saved_check_exist_num_list.shape[0])
        size_arr_for_mem = how_loop
        mem_size_per_1data = 1 + 1  # Bite  # check_exist_need_calc  # TF
        mem_size = ((mem_size_per_1data * size_arr_for_mem) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
            check_exist_need_calc, TF = check_exist_step1(
                saved_similar_num_list, before_similar_num_list, saved_check_exist_num_list, progress
            )
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")
        saved_check_exist_num_list = saved_check_exist_num_list[TF]
        saved_check_exist_eq_list = saved_check_exist_eq_list[TF]
        saved_check_exist_id_list = saved_check_exist_id_list[TF]
        saved_check_exist_need_calc_list = check_exist_need_calc[TF]
        saved_back_change_pattern = saved_back_change_pattern[TF]

        logger.info(f"   check_exist_step2")
        logger.info(f"      using Memory size =  {str_using_mem()}")
        how_loop = int(saved_check_exist_num_list.shape[0])
        size_arr_for_mem = how_loop
        mem_size_per_1data = 1  # Bite  # same
        mem_size = ((mem_size_per_1data * size_arr_for_mem) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop - 1, header=header) as progress:
            same = check_exist_step2(num_threads, saved_check_exist_num_list, progress)
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")

        logger.info(f"   check_exist_step3")
        logger.info(f"      using Memory size =  {str_using_mem()}")
        how_loop = int(np.max(same)) + 1 if same.shape[0] != 0 else 0
        size_arr_for_mem = saved_check_exist_eq_list.shape[0]
        mem_size_per_1data = 1 + (2 * 2 * (max_op * (max_op + 1)) // 2)  # Bite
        mem_size = ((mem_size_per_1data * size_arr_for_mem) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
            (
                TF,
                check_change_x_ones,
                check_change_x_tot,
                saved_check_exist_need_calc_list,
                one_eq_calc_check_change_x,
            ) = check_exist_step3(
                max_op,
                num_threads,
                random_x,
                same,
                saved_check_exist_eq_list,
                saved_check_exist_id_list,
                saved_back_change_pattern,
                saved_check_exist_need_calc_list,
                progress,
            )

        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")

        saved_check_exist_num_list = saved_check_exist_num_list[TF].copy()
        saved_check_exist_eq_list = saved_check_exist_eq_list[TF].copy()
        saved_check_exist_id_list = saved_check_exist_id_list[TF].copy()
        check_change_x_ones = check_change_x_ones[TF].copy()
        check_change_x_tot = check_change_x_tot[TF].copy()
        saved_check_exist_need_calc_list = saved_check_exist_need_calc_list[TF].copy()
        one_eq_calc_check_change_x = one_eq_calc_check_change_x[TF].copy()
        same = same[TF].copy()

        logger.info(f"   check_exist_step4")
        logger.info(f"      using Memory size =  {str_using_mem()}")
        how_loop = int(np.max(same)) + 1 if same.shape[0] != 0 else 0
        size_arr_for_mem = how_loop
        mem_size_per_1data = 8 * 3  # Bite
        mem_size = ((mem_size_per_1data * how_loop) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
            saved_check_exist_need_calc_list = check_exist_step4(
                same, saved_check_exist_num_list, saved_check_exist_need_calc_list, progress
            )
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")

        TF = one_eq_calc_check_change_x[:, 0, 0] != int_nan
        TF &= ~saved_check_exist_need_calc_list
        one_eq_calc_check_change_x[TF] = int_nan

        saved_equation_list = del_concatenate(saved_equation_list, saved_check_exist_eq_list)
        saved_need_calc_list = del_concatenate(saved_need_calc_list, saved_check_exist_need_calc_list)
        saved_base_eq_id_list = del_concatenate(saved_base_eq_id_list, saved_check_exist_id_list)
        saved_check_change_x_tot = del_concatenate(saved_check_change_x_tot, check_change_x_tot)
        saved_check_change_x_ones = del_concatenate(saved_check_change_x_ones, check_change_x_ones)
        saved_one_eq_calc_check_change_x = del_concatenate(saved_one_eq_calc_check_change_x, one_eq_calc_check_change_x)

        sort_index = np.empty((saved_need_calc_list.shape[0]), dtype="int64")
        sort_index[: np.sum(saved_need_calc_list)] = np.arange(saved_need_calc_list.shape[0])[saved_need_calc_list]
        sort_index[np.sum(saved_need_calc_list) :] = np.arange(saved_need_calc_list.shape[0])[~saved_need_calc_list]

        saved_equation_list = saved_equation_list[sort_index]
        saved_need_calc_list = saved_need_calc_list[sort_index]
        saved_base_eq_id_list = saved_base_eq_id_list[sort_index]
        saved_check_change_x_tot = saved_check_change_x_tot[sort_index]
        saved_check_change_x_ones = saved_check_change_x_ones[sort_index]
        saved_one_eq_calc_check_change_x = saved_one_eq_calc_check_change_x[sort_index]

        make_final_cache(
            max_op,
            saved_equation_list,
            saved_need_calc_list,
            saved_base_eq_id_list,
            saved_check_change_x_tot,
            saved_check_change_x_ones,
            saved_one_eq_calc_check_change_x,
            logger,
        )
        Tcount = np.sum(saved_need_calc_list)
        Fcount = np.sum(~saved_need_calc_list)
        logger.info(f"need calc  {Tcount}:{Fcount}")

    else:  # max_op==7
        saved_similar_num_list = np.empty((0, random_x.shape[1]), dtype="float64")
        saved_check_exist_num_list = np.empty((0, random_x.shape[1]), dtype="float64")
        time1, time2 = datetime.datetime.now(), datetime.datetime.now()
        for n_op1 in range(max_op - 1, -1, -1):
            n_op2 = max_op - 1 - n_op1
            logger.info(f"   n_op1 = {n_op1}, n_op2 = {n_op2}")
            how_loop, loop_per_threads = loop_count(max_op, n_op1, num_threads)
            logger.info(f"      make_unique_equations_thread")
            logger.info(f"         using Memory size =  {str_using_mem()}")
            size_arr_for_mem = how_loop
            mem_size_per_1data = (  # Byte
                random_x.shape[1] * 8  # save_similar_num_list
                + 1 * 8  # head_save_similar_num_list
                + 1  # TF_list
                + 1  # check_exist_TF
                + (2 * max_op + 1)  # save_equation_list
                + 5 * 8  # save_base_eq_id_list
                + max_op  # save_back_change_pattern
                + (max_op * (max_op + 1))  # save_check_change_x_tot
                + (max_op * (max_op + 1))  # save_check_change_x_ones
            )
            mem_size = ((mem_size_per_1data * loop_per_threads * num_threads) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
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
                    back_change_pattern,
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
            logger.info(f"         using Memory size =  {str_using_mem()}")
            how_loop = int(np.sum(TF_list[1:]))
            mem_size_per_1data = (  # Byte
                similar_num_list.shape[2] * 8  # return_similar_num_list, # random_x.shape[1] * 8
                + 8  # head_return_similar_num_list
            )
            mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
            size_arr_for_mem = np.sum(TF_list)
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
            )
            time1, time2 = time2, datetime.datetime.now()
            header = "         "
            with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
                dim_reduction_7(TF_list, similar_num_list, progress)
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")

            logger.info(f"      make_check_exist_info")
            logger.info(f"         using Memory size =  {str_using_mem()}")
            size_arr_for_mem = int(np.sum(check_exist_TF))
            mem_size_per_1data = (  # Bite
                similar_num_list.shape[2] * 8  # return_check_exist_num, # random_x.shape[1] * 8
                + equation_list.shape[2]  # return_check_exist_eq, # (2 * max_op + 1)
                + 5 * 8  # return_check_exist_id
                + back_change_pattern.shape[2]  # return_back_change_pattern
            )
            mem_size = ((mem_size_per_1data * size_arr_for_mem) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
            )
            check_exist_num, check_exist_eq, check_exist_id, check_exist_back_change_pattern = make_check_exist_info_7(
                check_exist_TF, similar_num_list, equation_list, base_eq_id_list, back_change_pattern
            )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")

            logger.info(f"      dim_reduction_info")
            logger.info(f"         using Memory size =  {str_using_mem()}")
            how_loop = int(np.sum(TF_list))
            size_arr_for_mem = how_loop
            mem_size_per_1data = (
                similar_num_list.shape[2] * 8  # return_similar_num_list, # random_x.shape[1] * 8
                + equation_list.shape[2]  # return_equation_list, # + (2 * max_op + 1)
                + 5 * 8  # return_base_eq_id_list
                + check_change_x_tot.shape[2] * 2  # return_check_change_x_tot
                + check_change_x_ones.shape[2] * 2  # return_check_change_x_ones
            )  # Bite
            mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
            logger.info(
                f"         Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
            )
            (
                return_equation_list,
                return_similar_num_list,
                return_base_eq_id_list,
                return_check_change_x_tot,
                check_change_x_ones,
            ) = dim_reduction_info_7(
                TF_list,
                similar_num_list,
                equation_list,
                base_eq_id_list,
                check_change_x_tot,
                check_change_x_ones,
            )
            time1, time2 = time2, datetime.datetime.now()
            logger.info(f"         time : {time2-time1}")

            saved_equation_list = del_concatenate(saved_equation_list, return_equation_list)
            _saved_similar_num_list = del_concatenate(saved_similar_num_list, return_similar_num_list)
            saved_similar_num_list = _saved_similar_num_list[np.argsort(_saved_similar_num_list[:, 0])].copy()
            saved_base_eq_id_list = del_concatenate(saved_base_eq_id_list, return_base_eq_id_list)
            saved_check_change_x_tot = del_concatenate(saved_check_change_x_tot, return_check_change_x_tot)
            saved_check_change_x_ones = del_concatenate(saved_check_change_x_ones, return_check_change_x_ones)
            saved_check_exist_num_list = del_concatenate(saved_check_exist_num_list, check_exist_num)
            saved_check_exist_eq_list = del_concatenate(saved_check_exist_eq_list, check_exist_eq)
            saved_check_exist_id_list = del_concatenate(saved_check_exist_id_list, check_exist_id)
            saved_back_change_pattern = del_concatenate(saved_back_change_pattern, check_exist_back_change_pattern)
            del (
                return_equation_list,
                return_similar_num_list,
                return_base_eq_id_list,
                return_check_change_x_tot,
                return_check_change_x_ones,
                _saved_similar_num_list,
                TF_list,
                similar_num_list,
                equation_list,
                base_eq_id_list,
                check_change_x_tot,
                check_change_x_ones,
                check_exist_num,
                check_exist_eq,
                check_exist_id,
                check_exist_back_change_pattern,
            )
            gc.collect()
        logger.info(f"   check_exist_step1")
        logger.info(f"      using Memory size =  {str_using_mem()}")
        how_loop = int(saved_check_exist_num_list.shape[0])
        size_arr_for_mem = how_loop
        mem_size_per_1data = 2  # Bite
        mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
            TF = check_exist_step1_7(
                saved_similar_num_list, before_similar_num_list, saved_check_exist_num_list, progress
            )
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")
        saved_check_exist_num_list = saved_check_exist_num_list[TF]
        saved_check_exist_eq_list = saved_check_exist_eq_list[TF]
        saved_check_exist_id_list = saved_check_exist_id_list[TF]
        saved_back_change_pattern = saved_back_change_pattern[TF]

        logger.info(f"   check_exist_step2")
        logger.info(f"      using Memory size =  {str_using_mem()}")
        how_loop = int(saved_check_exist_num_list.shape[0])
        size_arr_for_mem = how_loop
        mem_size_per_1data = 1  # Bite
        mem_size = ((mem_size_per_1data * np.sum(TF_list)) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop - 1, header=header) as progress:
            same = check_exist_step2_7(num_threads, saved_check_exist_num_list, progress)
        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")

        logger.info(f"   check_exist_step3")
        logger.info(f"      using Memory size =  {str_using_mem()}")
        how_loop = int(np.max(same)) + 1 if same.shape[0] != 0 else 0
        size_arr_for_mem = saved_check_exist_eq_list.shape[0]
        mem_size_per_1data = 1 + (2 * 2 * (max_op * (max_op + 1)) // 2)  # Bite
        mem_size = ((mem_size_per_1data * size_arr_for_mem) // 100000) / 10
        logger.info(
            f"      Memory size of numpy array = {mem_size} M bytes +alpha (1data={mem_size_per_1data} bytes, size_arr_for_mem={size_arr_for_mem})"
        )
        header = "      "
        time1, time2 = time2, datetime.datetime.now()
        with loop_log(logger, interval=log_interval, tot_loop=how_loop, header=header) as progress:
            TF, check_change_x_ones, check_change_x_tot = check_exist_step3_7(
                num_threads,
                random_x,
                same,
                saved_check_exist_eq_list,
                saved_check_exist_id_list,
                saved_back_change_pattern,
                progress,
            )

        time1, time2 = time2, datetime.datetime.now()
        logger.info(f"      time : {time2-time1}")

        saved_check_exist_num_list = saved_check_exist_num_list[TF]
        saved_check_exist_eq_list = saved_check_exist_eq_list[TF]
        saved_check_exist_id_list = saved_check_exist_id_list[TF]
        check_change_x_ones = check_change_x_ones[TF]
        check_change_x_tot = check_change_x_tot[TF]
        same = same[TF]

        saved_equation_list = del_concatenate(saved_equation_list, saved_check_exist_eq_list)
        saved_base_eq_id_list = del_concatenate(saved_base_eq_id_list, saved_check_exist_id_list)
        saved_check_change_x_tot = del_concatenate(saved_check_change_x_tot, check_change_x_tot)
        saved_check_change_x_ones = del_concatenate(saved_check_change_x_ones, check_change_x_ones)

        make_final_cache_7(
            saved_equation_list,
            saved_base_eq_id_list,
            saved_check_change_x_tot,
            saved_check_change_x_ones,
            logger,
        )
        Tcount = saved_equation_list.shape[0]
        logger.info(f"need calc  {Tcount}:0")


# @njit(error_model="numpy")
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
    save_back_change_pattern = np.full((num_threads, loop_per_threads, max_op), int_nan, dtype="int8")
    save_check_change_x_tot = np.full(
        (num_threads, loop_per_threads, (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    save_check_change_x_ones = np.full(
        (num_threads, loop_per_threads, (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    save_one_eq_calc_check_change_x = np.full(
        (num_threads, loop_per_threads, (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    if n_op1 >= n_op2:
        ops = np.array([-1, -2, -3, -4])
    else:
        ops = np.array([-4])
    len_base_eq_arr1 = base_eq_arr1.shape[0]
    len_base_eq_arr2 = base_eq_arr2.shape[0]
    loop = len_base_eq_arr1 * len_base_eq_arr2
    split_indexes = np.array_split(np.arange(loop), num_threads)
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), int_nan, dtype="int8")
        cache_for_mask_x = np.zeros((tot_mask_x.shape[0], 2, random_x.shape[1]), dtype="float64")
        TF_mask_x = np.ones(tot_mask_x.shape[0], dtype="bool")
        same_for_mask_x = np.zeros((tot_mask_x.shape[0]), dtype="int64")
        norm_same_for_mask_x = np.zeros((tot_mask_x.shape[0]), dtype="int64")
        counter = 0
        for i in split_indexes[thread_id]:
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
                    is_save, is_calc, all_covered = True, True, True
                    back_change_pattern = dict_change_x_pattern[base_back_x_max][number]
                    equation[len_base_eq1 : len_base_eq1 + len_base_eq2] = make_eq(
                        base_back_equation, back_change_pattern
                    )
                    if find_min_x_max(equation, random_x, random_for_find_min_x_max):
                        is_save, is_calc = False, False
                    if is_save:
                        before_check_change_x1 = before_check_change_x_list1[id1]
                        len_before_check_change_x1 = count_True(
                            before_check_change_x1[:, 0], 2, int_nan
                        )  # 2 -> lambda x: x != border
                        before_check_change_x2 = before_check_change_x_list2[id2]
                        changed_before_check_change_x2 = make_eq(before_check_change_x2, back_change_pattern)
                        len_before_check_change_x2 = count_True(
                            changed_before_check_change_x2[:, 0], 2, int_nan
                        )  # 2 -> lambda x: x != border

                        save_similar_num = nb_calc_RPN(random_x, equation)
                        if save_similar_num[0, 0] == int_nan:  # any_isinf, all_const
                            if count_True(equation, 5, 0) != 0:  # likely (a-a), not (1+1) ,  lambda x: x > border
                                is_save, is_calc = False, False
                            elif is_all_zero(save_similar_num[1], atol=0):  # all_zero
                                is_save, is_calc = False, False
                            else:  # (1+1)
                                is_calc = False
                    if is_save:
                        eq_x_max = np.max(equation)
                        mask_x = dict_mask_x[eq_x_max]
                        c, norm_c = 0, 0
                        TF_mask_x[: mask_x.shape[0]] = True
                        for k in range(mask_x.shape[0]):
                            for l in range(len_before_check_change_x1):
                                if mask_x[k, before_check_change_x1[l, 0]] > mask_x[k, before_check_change_x1[l, 1]]:
                                    TF_mask_x[k] = False
                                    break
                            if TF_mask_x[k]:
                                for l in range(len_before_check_change_x2):
                                    if (
                                        mask_x[k, changed_before_check_change_x2[l, 0]]
                                        > mask_x[k, changed_before_check_change_x2[l, 1]]
                                    ):
                                        TF_mask_x[k] = False
                                        break
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
                            same, norm_same = False, False
                            for l in range(norm_c):
                                if isclose(cache_for_mask_x[l, 0, 0], similar_num[0, 0]):
                                    if isclose_arr(cache_for_mask_x[l, 0], similar_num[0]):
                                        norm_same_for_mask_x[k] = l
                                        norm_same = True
                            if not norm_same:
                                norm_same_for_mask_x[k] = norm_c
                                cache_for_mask_x[norm_c, 0] = similar_num[0]
                                norm_c += 1
                            for l in range(c):
                                if isclose(cache_for_mask_x[l, 1, 0], similar_num[1, 0]):
                                    if isclose_arr(cache_for_mask_x[l, 1], similar_num[1]):
                                        same_for_mask_x[k] = l
                                        same = True
                            if not same:
                                same_for_mask_x[k] = c
                                cache_for_mask_x[c, 1] = similar_num[1]
                                c += 1
                            if save_similar_num[0, 0] > similar_num[0, 0]:  # min
                                save_similar_num[0] = similar_num[0]
                            if save_similar_num[1, 0] > similar_num[1, 0]:  # min
                                save_similar_num[1] = similar_num[1]
                        if not np.any(TF_mask_x[: mask_x.shape[0]]):
                            is_save = False
                        if is_save and max_op + 1 != eq_x_max:
                            # except when using x of (number of operators + 1) type
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
                            can_use, all_covered, add_check_change_x = make_check_change_x(
                                mask_x, same_for_mask_x[: mask_x.shape[0]], TF_mask_x[: mask_x.shape[0]]
                            )
                            if not can_use:
                                is_save = False
                                # print("drop : make_check_change_x")
                                # print(equation)
                                # print(same_for_mask_x[: mask_x.shape[0]])
                                # print("arr_similar_num_head : ", arr_similar_num_head[: mask_x.shape[0]])
                                # print(TF_mask_x[: mask_x.shape[0]])
                                # print(can_use, all_covered, add_check_change_x)
                            if can_use and all_covered:
                                can_use_norm, one_eq_calc_check_change_x = make_check_change_x_norm(
                                    mask_x,
                                    norm_same_for_mask_x[: mask_x.shape[0]],
                                    TF_mask_x[: mask_x.shape[0]],
                                    add_check_change_x[0],
                                )
                                if not can_use_norm:
                                    is_save = False
                                    # print("drop : make_check_change_x_norm")
                                    # print(equation)
                                    # print(same_for_mask_x[: mask_x.shape[0]])
                                    # print(norm_same_for_mask_x[: mask_x.shape[0]])
                                    # print("arr_similar_num_head : ", arr_similar_num_head[: mask_x.shape[0]])
                                    # print(TF_mask_x[: mask_x.shape[0]])
                                    # print(can_use, all_covered, add_check_change_x, one_eq_calc_check_change_x)
                            if is_save:
                                if not all_covered:
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
                                    save_back_change_pattern[thread_id, counter, : back_change_pattern.shape[0]] = (
                                        back_change_pattern
                                    )
                                    is_save = False
                        if is_save:
                            TF_list[thread_id, counter] = True
                            save_similar_num_list[thread_id, counter] = save_similar_num
                            head_save_similar_num_list[thread_id, counter] = save_similar_num[0, 0]
                            save_need_calc_list[thread_id, counter] = is_calc
                            save_equation_list[thread_id, counter] = equation
                            save_base_eq_id_list[thread_id, counter, 0] = n_op1
                            save_base_eq_id_list[thread_id, counter, 1] = id1
                            save_base_eq_id_list[thread_id, counter, 2] = id2
                            save_base_eq_id_list[thread_id, counter, 3] = make_change_x_id(
                                back_change_pattern, max_op + 1
                            )
                            save_base_eq_id_list[thread_id, counter, 4] = merge_op
                            save_one_eq_calc_check_change_x[
                                thread_id, counter, : one_eq_calc_check_change_x.shape[0]
                            ] = one_eq_calc_check_change_x
                            n = 0
                            for k in range(len_before_check_change_x1):
                                save_check_change_x_tot[thread_id, counter, n] = before_check_change_x1[k]
                                n += 1
                            for k in range(len_before_check_change_x2):
                                unique = True
                                for l in range(len_before_check_change_x1):
                                    if changed_before_check_change_x2[k, 0] == before_check_change_x1[l, 0]:
                                        if changed_before_check_change_x2[k, 1] == before_check_change_x1[l, 1]:
                                            unique = False
                                            break
                                if unique:
                                    save_check_change_x_tot[thread_id, counter, n] = changed_before_check_change_x2[k]
                                    n += 1
                            for k in range(add_check_change_x.shape[1]):
                                if add_check_change_x[0, k, 0] != int_nan:
                                    save_check_change_x_tot[thread_id, counter, n] = add_check_change_x[0, k]
                                    save_check_change_x_ones[thread_id, counter, k] = add_check_change_x[0, k]
                                    n += 1
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
        save_back_change_pattern,
        save_one_eq_calc_check_change_x,
    )


@njit(parallel=True, error_model="numpy")
def dim_reduction(TF_list, similar_num_list, need_calc_list, progress_proxy):
    int_nan = -100
    num_threads = similar_num_list.shape[0]
    return_similar_num_list = np.full((np.sum(TF_list), 2, similar_num_list.shape[3]), int_nan, dtype="float64")
    return_similar_num_list[: np.sum(TF_list[0])] = similar_num_list[0, TF_list[0]]
    head_return_similar_num_list = np.full((np.sum(TF_list)), int_nan, dtype="float64")
    head_similar_num_list = similar_num_list[:, :, 0, 0].copy()
    head_return_similar_num_list[: np.sum(TF_list[0])] = head_similar_num_list[0, TF_list[0]]
    last_index = np.sum(TF_list[0])
    for target_index in range(1, num_threads):
        true_index = np.arange(TF_list.shape[1])[TF_list[target_index]]
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
        next_index = last_index + np.sum(TF_list[target_index])
        return_similar_num_list[last_index:next_index] = similar_num_list[target_index, TF_list[target_index]]
        head_return_similar_num_list[last_index:next_index] = head_similar_num_list[target_index, TF_list[target_index]]
        last_index = next_index
    # return TF_list, need_calc_list


@njit(parallel=True, error_model="numpy")
def dim_reduction_info(
    TF_list,
    similar_num_list,
    need_calc_list,
    equation_list,
    base_eq_id_list,
    check_change_x_tot,
    check_change_x_ones,
    one_eq_calc_check_change_x,
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
    return_one_eq_calc_check_change_x = np.full((sum_TF, one_eq_calc_check_change_x.shape[2], 2), int_nan, dtype="int8")
    last_index = 0
    for i in range(num_threads):
        indexes = np.arange(TF_list.shape[1])[TF_list[i]]
        return_similar_num_list[last_index : last_index + indexes.shape[0]] = similar_num_list[i, indexes]
        return_equation_list[last_index : last_index + indexes.shape[0]] = equation_list[i, indexes]
        return_base_eq_id_list[last_index : last_index + indexes.shape[0]] = base_eq_id_list[i, indexes]
        return_need_calc_list[last_index : last_index + indexes.shape[0]] = need_calc_list[i, indexes]
        return_check_change_x_tot[last_index : last_index + indexes.shape[0]] = check_change_x_tot[i, indexes]
        return_check_change_x_ones[last_index : last_index + indexes.shape[0]] = check_change_x_ones[i, indexes]
        return_one_eq_calc_check_change_x[last_index : last_index + indexes.shape[0]] = one_eq_calc_check_change_x[
            i, indexes
        ]
        last_index += indexes.shape[0]
    return (
        return_equation_list,
        return_similar_num_list,
        return_need_calc_list,
        return_base_eq_id_list,
        return_check_change_x_tot,
        return_check_change_x_ones,
        return_one_eq_calc_check_change_x,
    )


@njit(parallel=True, error_model="numpy")
def make_check_exist_info(check_exist_TF, similar_num_list, equation_list, base_eq_id_list, back_change_pattern):
    int_nan = -100
    num_threads = check_exist_TF.shape[0]
    sum_check_exist = np.sum(check_exist_TF)
    return_check_exist_num = np.full((sum_check_exist, 2, similar_num_list.shape[3]), int_nan, dtype="float64")
    return_check_exist_eq = np.full((sum_check_exist, equation_list.shape[2]), int_nan, dtype="int8")
    return_check_exist_id = np.full((sum_check_exist, 5), int_nan, dtype="int64")
    return_back_change_pattern = np.full((sum_check_exist, back_change_pattern.shape[2]), int_nan, dtype="int8")
    last_index = 0
    for i in range(num_threads):
        indexes = np.arange(check_exist_TF.shape[1])[check_exist_TF[i]]
        return_check_exist_num[last_index : last_index + indexes.shape[0]] = similar_num_list[i, indexes]
        return_check_exist_eq[last_index : last_index + indexes.shape[0]] = equation_list[i, indexes]
        return_check_exist_id[last_index : last_index + indexes.shape[0]] = base_eq_id_list[i, indexes]
        return_back_change_pattern[last_index : last_index + indexes.shape[0]] = back_change_pattern[i, indexes]
        last_index += indexes.shape[0]
    return (
        return_check_exist_num[:last_index],
        return_check_exist_eq[:last_index],
        return_check_exist_id[:last_index],
        return_back_change_pattern[:last_index],
    )


@njit(parallel=True, error_model="numpy")
def check_exist_step1(similar_num_list, before_similar_num_list, check_exist_num_arr, progress_proxy):
    int_nan = -100
    head_saved_similar_num_list = similar_num_list[:, 0, 0].copy()
    head_before_similar_num_list = before_similar_num_list[:, 0, 0].copy()
    len_return = check_exist_num_arr.shape[0]
    check_exist_need_calc = np.zeros((len_return), dtype="bool")
    TF = np.zeros((len_return), dtype="bool")
    for i in prange(check_exist_num_arr.shape[0]):
        is_calc, is_save = True, True
        for j in range(head_before_similar_num_list.shape[0]):
            if isclose(check_exist_num_arr[i, 0, 0], head_before_similar_num_list[j]):
                if isclose_arr(check_exist_num_arr[i, 0], before_similar_num_list[j, 0]):
                    is_calc = False
                    if isclose_arr(check_exist_num_arr[i, 1], before_similar_num_list[j, 1]):
                        is_save = False
                        break
            elif check_exist_num_arr[i, 0, 0] < head_before_similar_num_list[j]:
                break
        if is_save:
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
    return check_exist_need_calc, TF


@njit(parallel=True, error_model="numpy")
def check_exist_step2(num_threads, check_exist_num_arr, progress_proxy):
    int_nan = -100
    n_check_exist = check_exist_num_arr.shape[0]
    arg_index = np.argsort(check_exist_num_arr[:, 1, 0])
    same = np.full((n_check_exist), int_nan, dtype="int64")
    count = 0
    filled_count = np.zeros(num_threads, dtype="int64")
    for i in range(n_check_exist):
        if same[arg_index[i]] == int_nan:
            same[arg_index[i]] = count
            filled_count[:] = 0
            for thread_id in prange(num_threads):
                for j in range(i + 1 + thread_id, n_check_exist, num_threads):
                    if same[arg_index[j]] == int_nan:
                        if isclose(check_exist_num_arr[arg_index[i], 1, 0], check_exist_num_arr[arg_index[j], 1, 0]):
                            if isclose_arr(check_exist_num_arr[arg_index[i], 1], check_exist_num_arr[arg_index[j], 1]):
                                same[arg_index[j]] = count
                                filled_count[thread_id] += 1
                        else:
                            break
            count += 1
            progress_proxy.update(np.sum(filled_count) + 1)
    change_nums = np.full(count, int_nan, dtype="int64")
    base_same = same.copy()
    n = 0
    for i in range(n_check_exist):
        if change_nums[base_same[i]] == int_nan:
            change_nums[base_same[i]] = n
            same[i] = n
            n += 1
        else:
            same[i] = change_nums[base_same[i]]
    return same


# @njit(error_model="numpy")
@njit(parallel=True, error_model="numpy")
def check_exist_step3(
    max_op,
    num_threads,
    random_x,
    arr_same,
    check_exist_eq,
    check_exist_id,
    arr_back_change_pattern,
    check_exist_need_calc,
    progress_proxy,
):
    dict_mask_x = make_dict_mask_x(max_op + 1)
    before_check_change_x_dict = cache_check_change_x_load(max_op - 1)
    arange = np.arange(check_exist_eq.shape[0])
    int_nan = -100
    if arr_same.shape[0] == 0:
        loop = 0
    else:
        loop = np.max(arr_same) + 1
    TF = np.zeros(check_exist_eq.shape[0], dtype="bool")
    save_check_change_x_tot = np.full((check_exist_eq.shape[0], (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8")
    save_check_change_x_ones = np.full(
        (check_exist_eq.shape[0], (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    save_one_eq_calc_check_change_x = np.full(
        (check_exist_eq.shape[0], (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    for i in prange(loop):
        indexes = arange[arr_same == i]
        mat_use, dict_check_change_x, calc, dict_one_eq_calc_check_change_x = sub_check_exist_step3(
            max_op,
            num_threads,
            random_x,
            check_exist_eq,
            check_exist_id,
            arr_back_change_pattern,
            dict_mask_x,
            before_check_change_x_dict,
            indexes,
        )
        len_calc = count_True(calc, 2, int_nan)  # 2 -> lambda x: x != border
        len_mat_use = count_True(mat_use[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
        for m in range(len_mat_use):
            if check_exist_need_calc[indexes[mat_use[m, 0]]]:
                if not m in calc[:len_calc]:
                    # print("drop : ", check_exist_eq[indexes[mat_use[m, 0]]])
                    check_exist_need_calc[indexes[mat_use[m, 0]]] = False
                else:
                    one_eq_calc_check_change_x = dict_one_eq_calc_check_change_x[mat_use[m, 0]]
                    save_one_eq_calc_check_change_x[indexes[mat_use[m, 0]], : one_eq_calc_check_change_x.shape[0]] = (
                        one_eq_calc_check_change_x
                    )
            TF[indexes[mat_use[m, 0]]] = True
            add = dict_check_change_x[mat_use[m, 0]][mat_use[m, 1]]
            len_add = count_True(add[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
            n_op1 = check_exist_id[indexes[mat_use[m, 0]], 0]
            id1 = check_exist_id[indexes[mat_use[m, 0]], 1]
            n_op2 = max_op - 1 - n_op1
            id2 = check_exist_id[indexes[mat_use[m, 0]], 2]
            back_change_pattern = arr_back_change_pattern[indexes[mat_use[m, 0]]]
            before_check_change_x1 = before_check_change_x_dict[n_op1][id1]
            len_before1 = count_True(before_check_change_x1[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
            before_check_change_x2 = before_check_change_x_dict[n_op2][id2]
            changed_before_check_change_x2 = make_eq(before_check_change_x2, back_change_pattern)
            len_before2 = count_True(changed_before_check_change_x2[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
            save_check_change_x_ones[indexes[mat_use[m, 0]], :len_add] = add[:len_add]
            save_check_change_x_tot[indexes[mat_use[m, 0]], :len_before1] = before_check_change_x1[:len_before1]
            save_check_change_x_tot[indexes[mat_use[m, 0]], len_before1 : len_before1 + len_before2] = (
                changed_before_check_change_x2[:len_before2]
            )
            save_check_change_x_tot[
                indexes[mat_use[m, 0]],
                len_before1 + len_before2 : len_before1 + len_before2 + len_add,
            ] = add[:len_add]
        progress_proxy.update(1)
    return (
        TF,
        save_check_change_x_ones,
        save_check_change_x_tot,
        check_exist_need_calc,
        save_one_eq_calc_check_change_x,
    )


# @njit(parallel=True, error_model="numpy")
@njit(error_model="numpy")
def sub_check_exist_step3(
    max_op,
    num_threads,
    random_x,
    check_exist_eq,
    check_exist_id,
    arr_back_change_pattern,
    dict_mask_x,
    before_check_change_x_dict,
    indexes,
):
    print_counter = 0
    lim_print_counter = 100000000
    printed = False

    int_nan = -100
    arange = np.arange(indexes.shape[0])
    equation = check_exist_eq[indexes[0]]
    eq_x_max = np.max(equation)
    mask_x = dict_mask_x[eq_x_max]
    mat_use = np.empty((num_threads, indexes.shape[0], 3), dtype="int64")
    return_mat_use = np.full((indexes.shape[0], 2), int_nan, dtype="int64")
    cache_for_mask_x = np.zeros((mask_x.shape[0], random_x.shape[1]), dtype="float64")
    norm_cache_for_mask_x = np.zeros((mask_x.shape[0], random_x.shape[1]), dtype="float64")
    TF_mask_x = np.ones((indexes.shape[0], mask_x.shape[0]), dtype="bool")
    same_for_mask_x = np.zeros((mask_x.shape[0]), dtype="int64")
    norm_same_for_mask_x = np.zeros((mask_x.shape[0]), dtype="int64")
    dict_check_change_x = dict()
    dict_norm_check_change_x = dict()
    calc = np.empty((indexes.shape[0]), dtype="int64")
    calc_pattern = np.empty((indexes.shape[0]), dtype="int64")
    n, norm_n = 0, 0
    for k in range(mask_x.shape[0]):
        similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
        norm_same, same = False, False
        for l in range(norm_n):
            if isclose(norm_cache_for_mask_x[l, 0], similar_num[0, 0]):
                if isclose_arr(norm_cache_for_mask_x[l], similar_num[0]):
                    norm_same = True
                    break
        if not norm_same:
            norm_cache_for_mask_x[norm_n] = similar_num[0]
            norm_n += 1
        for l in range(n):
            if isclose(cache_for_mask_x[l, 0], similar_num[1, 0]):
                if isclose_arr(cache_for_mask_x[l], similar_num[1]):
                    same = True
                    break
        if not same:
            cache_for_mask_x[n] = similar_num[1]
            n += 1
    cache_for_mask_x = cache_for_mask_x[:n]
    norm_cache_for_mask_x = norm_cache_for_mask_x[:norm_n]
    len_norm_cache_for_mask_x = norm_n
    len_cache_for_mask_x = n
    dict_one_eq_calc_check_change_x = np.full((indexes.shape[0], max_op, 2), int_nan, dtype="int8")

    same_num_index = np.empty((indexes.shape[0], mask_x.shape[0]), dtype="int64")
    same_norm_num_index = np.empty((indexes.shape[0], mask_x.shape[0]), dtype="int64")
    patterns = np.empty((indexes.shape[0]), dtype="int64")
    use = np.full((indexes.shape[0]), int_nan, dtype="int64")
    n_max_mat_covered_num = np.empty((indexes.shape[0]), dtype="int64")
    n_min_mat_covered_num = np.empty((indexes.shape[0]), dtype="int64")

    for j in range(indexes.shape[0]):
        equation = check_exist_eq[indexes[j]]
        n_op1 = check_exist_id[indexes[j], 0]
        id1 = check_exist_id[indexes[j], 1]
        n_op2 = max_op - 1 - n_op1
        id2 = check_exist_id[indexes[j], 2]
        back_change_pattern = arr_back_change_pattern[indexes[j]]
        before_check_change_x1 = before_check_change_x_dict[n_op1][id1]
        before_check_change_x2 = before_check_change_x_dict[n_op2][id2]
        changed_before_check_change_x2 = make_eq(before_check_change_x2, back_change_pattern)
        len_before_check_change_x1 = count_True(before_check_change_x1[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
        len_before_check_change_x2 = count_True(
            changed_before_check_change_x2[:, 0], 2, int_nan
        )  # 2 -> lambda x: x != border
        TF_mask_x[j] = True
        for k in range(mask_x.shape[0]):
            for l in range(len_before_check_change_x1):
                if mask_x[k, before_check_change_x1[l, 0]] > mask_x[k, before_check_change_x1[l, 1]]:
                    TF_mask_x[j, k] = False
                    break
            if TF_mask_x[j, k]:
                for l in range(len_before_check_change_x2):
                    if (
                        mask_x[k, changed_before_check_change_x2[l, 0]]
                        > mask_x[k, changed_before_check_change_x2[l, 1]]
                    ):
                        TF_mask_x[j, k] = False
                        break
            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)
            for l in range(len_norm_cache_for_mask_x):
                if isclose(norm_cache_for_mask_x[l, 0], similar_num[0, 0]):
                    if isclose_arr(norm_cache_for_mask_x[l], similar_num[0]):
                        norm_same_for_mask_x[k] = l
                        break
            for l in range(len_cache_for_mask_x):
                if isclose(cache_for_mask_x[l, 0], similar_num[1, 0]):
                    if isclose_arr(cache_for_mask_x[l], similar_num[1]):
                        same_for_mask_x[k] = l
                        break
        same_num_index[j] = same_for_mask_x
        same_norm_num_index[j] = norm_same_for_mask_x
        arr_check_change_x = make_check_change_x(mask_x, same_for_mask_x, TF_mask_x[j])[2]
        dict_check_change_x[j] = arr_check_change_x
        patterns[j] = arr_check_change_x.shape[0]
    n_all_pattern = np.max(patterns)

    mat_covered_num = np.zeros((indexes.shape[0], n_all_pattern, len_cache_for_mask_x), dtype="bool")
    for j in range(indexes.shape[0]):
        arr_check_change_x = dict_check_change_x[j]
        for k in range(patterns[j]):
            for l in range(mask_x.shape[0]):
                if TF_mask_x[j, l]:
                    check = True
                    for m in range(arr_check_change_x.shape[1]):
                        if int_nan == arr_check_change_x[k, m, 0]:
                            break
                        elif mask_x[l, arr_check_change_x[k, m, 0]] > mask_x[l, arr_check_change_x[k, m, 1]]:
                            check = False
                            break
                    if check:
                        if mat_covered_num[j, k, same_num_index[j, l]]:
                            print("ERROR : mat_covered_num")
                            print(check_exist_eq[indexes[j]])
                            print(same_num_index[j], same_norm_num_index[j], TF_mask_x[j])
                            print(make_check_change_x(mask_x, same_num_index[j], TF_mask_x[j]))
                        else:  # if not mat_covered_num[j, k, same_num_index[j, l]]:
                            mat_covered_num[j, k, same_num_index[j, l]] = True
        _arr = np.array([np.sum(mat_covered_num[j, k]) for k in range(patterns[j])])
        n_max_mat_covered_num[j] = np.max(_arr)
        n_min_mat_covered_num[j] = np.min(_arr)

    # same eqs (1 op is safe)
    same_eq_shape_S = np.full((indexes.shape[0]), int_nan, dtype="int64")
    c = 0
    for j in range(indexes.shape[0]):
        max_c = 0
        for k in range(j):
            if same_eq_shape_S[k] == max_c:
                is_same = True
                for l in range(check_exist_eq.shape[1]):
                    if check_exist_eq[indexes[j], l] <= 0:
                        if check_exist_eq[indexes[j], l] != check_exist_eq[indexes[k], l]:
                            is_same = False
                            break
                    elif check_exist_eq[indexes[k], l] <= 0:
                        is_same = False
                        break
                if is_same:
                    same_eq_shape_S[j] = same_eq_shape_S[k]
                    break
                else:
                    max_c += 1
        if same_eq_shape_S[j] == int_nan:
            same_eq_shape_S[j] = c
            c += 1
    same_eq_shape_L = np.full((indexes.shape[0]), int_nan, dtype="int64")
    c = 0
    for j in range(indexes.shape[0]):
        max_c = 0
        for k in range(j):
            if same_eq_shape_L[k] == max_c:
                is_same = True
                for l in range(check_exist_eq.shape[1]):
                    if check_exist_eq[indexes[j], l] == 0:
                        if check_exist_eq[indexes[k], l] != 0:
                            is_same = False
                            break
                    elif check_exist_eq[indexes[k], l] == 0:
                        is_same = False
                        break
                    elif check_exist_eq[indexes[j], l] in [-1, -2]:
                        if not check_exist_eq[indexes[k], l] in [-1, -2]:
                            is_same = False
                            break
                    elif check_exist_eq[indexes[k], l] in [-1, -2]:
                        is_same = False
                        break
                    elif check_exist_eq[indexes[j], l] in [-3, -4]:
                        if check_exist_eq[indexes[j], l] != check_exist_eq[indexes[k], l]:
                            is_same = False
                            break
                    elif check_exist_eq[indexes[k], l] in [-3, -4]:
                        is_same = False
                        break
                if is_same:
                    same_eq_shape_L[j] = same_eq_shape_L[k]
                    break
                else:
                    max_c += 1
        if same_eq_shape_L[j] == int_nan:
            same_eq_shape_L[j] = c
            c += 1
    len_n_same_eq_shape_L = c

    arr_n_min = np.empty(len_n_same_eq_shape_L, dtype="int64")
    arr_n_max = np.empty(len_n_same_eq_shape_L, dtype="int64")
    for i in range(len_n_same_eq_shape_L):
        selected_indexes = arange[same_eq_shape_L == i]
        n_selected_indexes = selected_indexes.shape[0]
        selected_n_max_mat_covered_num = np.sort(n_max_mat_covered_num[selected_indexes])[::-1]
        selected_n_min_mat_covered_num = np.sort(n_min_mat_covered_num[selected_indexes])
        sum_ = 0
        arr_n_min[i] = n_selected_indexes + 1
        for j in range(n_selected_indexes):
            sum_ += selected_n_max_mat_covered_num[j]
            if sum_ >= len_cache_for_mask_x:
                arr_n_min[i] = j + 1
                break
        sum_ = 0
        arr_n_max[i] = n_selected_indexes
        for j in range(n_selected_indexes):
            sum_ += selected_n_min_mat_covered_num[j]
            if sum_ == len_cache_for_mask_x:
                arr_n_max[i] = j + 1
                break
            elif sum_ > len_cache_for_mask_x:
                arr_n_max[i] = j
                break

    mat_use = np.full((len_n_same_eq_shape_L, indexes.shape[0], 2), int_nan, dtype="int64")
    arr_coverd = np.empty(len_cache_for_mask_x, dtype="bool")
    for i in range(len_n_same_eq_shape_L):
        base_selected_indexes = arange[same_eq_shape_L == i]
        selected_indexes = np.empty_like(base_selected_indexes)
        n_selected_indexes = 0
        for j in range(np.max(same_eq_shape_S[base_selected_indexes]) + 1):
            TF = same_eq_shape_S[base_selected_indexes] == j
            selected_indexes[n_selected_indexes : n_selected_indexes + np.sum(TF)] = base_selected_indexes[TF]
            n_selected_indexes += np.sum(TF)
        len_mat_n_covered_num = 0
        for j in selected_indexes:
            len_mat_n_covered_num += patterns[j]
        base_mat_n_covered_num = np.empty((len_mat_n_covered_num, 3), dtype="int64")
        c_mat_n_covered_num = 0
        for j in selected_indexes:
            for k in range(patterns[j]):
                base_mat_n_covered_num[c_mat_n_covered_num, 0] = j
                base_mat_n_covered_num[c_mat_n_covered_num, 1] = k
                base_mat_n_covered_num[c_mat_n_covered_num, 2] = np.sum(mat_covered_num[j, k])
                c_mat_n_covered_num += 1
        mat_n_covered_num = np.empty((len_mat_n_covered_num, 3), dtype="int64")
        c_mat_n_covered_num = 0
        for j in range(np.max(base_mat_n_covered_num[:, 2]), -1, -1):
            same_n_covered_num_indexes = np.arange(len_mat_n_covered_num)[base_mat_n_covered_num[:, 2] == j]
            mat_n_covered_num[c_mat_n_covered_num : c_mat_n_covered_num + same_n_covered_num_indexes.shape[0]] = (
                base_mat_n_covered_num[same_n_covered_num_indexes]
            )
            c_mat_n_covered_num += same_n_covered_num_indexes.shape[0]
        for start_index in range(len_mat_n_covered_num):
            arr_coverd[:] = mat_covered_num[mat_n_covered_num[start_index, 0], mat_n_covered_num[start_index, 1]]
            mat_use[i, 0, 0] = mat_n_covered_num[start_index, 0]
            mat_use[i, 0, 1] = mat_n_covered_num[start_index, 1]
            c_mat_use = 1
            for j in range(start_index + 1, len_mat_n_covered_num):
                if np.all(arr_coverd):
                    break
                if not mat_n_covered_num[j, 0] in mat_use[i, :c_mat_use, 0]:
                    for k in range(len_cache_for_mask_x):
                        if arr_coverd[k]:
                            if mat_covered_num[mat_n_covered_num[j, 0], mat_n_covered_num[j, 1], k]:
                                break
                    else:
                        for k in range(len_cache_for_mask_x):
                            if not arr_coverd[k]:
                                if mat_covered_num[mat_n_covered_num[j, 0], mat_n_covered_num[j, 1], k]:
                                    arr_coverd[k] = True
                        mat_use[i, c_mat_use, 0] = mat_n_covered_num[j, 0]
                        mat_use[i, c_mat_use, 1] = mat_n_covered_num[j, 1]
                        c_mat_use += 1
            if not np.all(arr_coverd):
                mat_use[i] = int_nan
            else:
                break
    if np.any(mat_use[:, 0, 0] != int_nan):
        found_index = np.arange(len_n_same_eq_shape_L)[mat_use[:, 0, 0] != int_nan]
        use_index = found_index[0]
        n_use = np.sum(mat_use[use_index, :, 0] != int_nan)
        for i in found_index[1:]:
            if n_use > np.sum(mat_use[i, :, 0] != int_nan):
                use_index = i
                n_use = np.sum(mat_use[i, :, 0] != int_nan)
        for i in range(n_use):
            return_mat_use[i, 0] = mat_use[use_index, i, 0]
            return_mat_use[i, 1] = mat_use[use_index, i, 1]
        norm_check = len_cache_for_mask_x == len_norm_cache_for_mask_x
        same_norm_num_index = same_norm_num_index[return_mat_use[:n_use, 0]]
        norm_patterns = np.empty((n_use), dtype="int64")
        norm_TF_mask_x = np.ones((n_use, mask_x.shape[0]), dtype="bool")
        for j in range(n_use):
            check_change_x = dict_check_change_x[return_mat_use[j, 0]][return_mat_use[j, 1]]
            len_check_change_x = count_True(check_change_x[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
            norm_TF_mask_x[j] = TF_mask_x[return_mat_use[j, 0]]
            for k in range(mask_x.shape[0]):
                for l in range(len_check_change_x):
                    if mask_x[k, check_change_x[l, 0]] > mask_x[k, check_change_x[l, 1]]:
                        norm_TF_mask_x[j, k] = False
                        break
            arr_norm_check_change_x = make_check_change_x(mask_x, same_norm_num_index[j], norm_TF_mask_x[j])[2]
            dict_norm_check_change_x[j] = arr_norm_check_change_x
            norm_patterns[j] = arr_norm_check_change_x.shape[0]
        n_all_norm_pattern = np.max(norm_patterns)
        mat_covered_norm_num = np.zeros((n_use, n_all_norm_pattern, len_norm_cache_for_mask_x), dtype="bool")

        if norm_check:
            for k in range(n_use):
                calc[k] = k
            calc[n_use:] = int_nan
            calc_pattern[:] = 0
        else:
            for j in range(n_use):
                arr_norm_check_change_x = dict_norm_check_change_x[j]
                for k in range(norm_patterns[j]):
                    for l in range(mask_x.shape[0]):
                        if norm_TF_mask_x[j, l]:
                            check = True
                            for m in range(arr_norm_check_change_x.shape[1]):
                                if int_nan == arr_norm_check_change_x[k, m, 0]:
                                    break
                                elif (
                                    mask_x[l, arr_norm_check_change_x[k, m, 0]]
                                    > mask_x[l, arr_norm_check_change_x[k, m, 1]]
                                ):
                                    check = False
                                    break
                            if check:
                                if mat_covered_norm_num[j, k, same_norm_num_index[j, l]]:
                                    print("ERROR : mat_covered_norm_num")
                                    print(check_exist_eq[indexes[return_mat_use[j, 0]]])
                                    # print(same_num_index[j], same_norm_num_index[j], TF_mask_x[j])
                                    # print(make_check_change_x(mask_x, same_num_index[j], TF_mask_x[j]))
                                else:
                                    mat_covered_norm_num[j, k, same_norm_num_index[j, l]] = True
            for k in range(1, n_use + 1):
                for l in range(k):
                    calc[l] = l
                calc[k:] = int_nan
                while True:
                    calc_pattern[:] = 0
                    while True:
                        coverd = True
                        for l in range(len_norm_cache_for_mask_x):
                            one_coverd = False
                            for m in range(k):
                                if mat_covered_norm_num[calc[m], calc_pattern[m], l]:
                                    if one_coverd:
                                        coverd = False
                                        break
                                    else:
                                        one_coverd = True
                            if not one_coverd:
                                coverd = False
                            if not coverd:
                                break
                        if coverd:
                            norm_check = True
                            break
                        else:
                            done_plus_one = False
                            for l in range(k - 1, -1, -1):
                                if calc_pattern[l] != norm_patterns[calc[l]] - 1:
                                    calc_pattern[l] += 1
                                    done_plus_one = True
                                    break
                                else:
                                    calc_pattern[l] = 0
                            if not done_plus_one:
                                break
                    if norm_check:
                        break
                    else:
                        done_plus_one = False
                        for l in range(k - 1, -1, -1):
                            if calc[l] != n_use - (k - l):
                                calc[l] += 1
                                done_plus_one = True
                                break
                            else:
                                for m in range(l - 1, -1, -1):
                                    if calc[m] != n_use - (k - m):
                                        calc[l] = calc[m] + 1 + (l - m)
                                        break
                        if not done_plus_one:
                            break
                if norm_check:
                    break
        # """# print
        if not norm_check:
            print("not norm_check found")
            print("len_cache_for_mask_x, len_norm_cache_for_mask_x : ", len_cache_for_mask_x, len_norm_cache_for_mask_x)
            for i in range(n_use):
                print("eq : ", check_exist_eq[indexes[return_mat_use[i, 0]]])
                print(mat_covered_norm_num[return_mat_use[i, 0], return_mat_use[i, 1]])
        # """  # print
        if norm_check:
            for m in range(n_use):
                if m in calc:
                    one_eq_calc_check_change_x = dict_norm_check_change_x[m][calc_pattern[m]]
                    _len = one_eq_calc_check_change_x.shape[0]
                    dict_one_eq_calc_check_change_x[return_mat_use[m, 0], :_len] = one_eq_calc_check_change_x
        return return_mat_use, dict_check_change_x, calc, dict_one_eq_calc_check_change_x

    print("not found aming")
    one_found = False
    for i in range(2, indexes.shape[0] + 1):
        for j in range(len_n_same_eq_shape_L):
            if (arr_n_min[j] <= i) and (i <= arr_n_max[j]):
                if not printed and (print_counter >= lim_print_counter):
                    printed = True
                    # """  # print
                    print("   arr_n_min : ", arr_n_min)
                    print("   arr_n_max : ", arr_n_max)
                    for j in range(indexes.shape[0]):
                        print(j)
                        print("   eq : ", check_exist_eq[indexes[j]])
                        print("   same_eq_shape L, S : ", same_eq_shape_L[j], same_eq_shape_S[j])
                        print("   n_max,min mat_covered_num : ", n_max_mat_covered_num[j], n_min_mat_covered_num[j])
                        print("   mat_covered_num[0] : ", mat_covered_num[j, 0])
                        print()
                    print()
                    # """  # print
                base_selected_indexes = arange[same_eq_shape_L == j]
                selected_indexes = np.empty_like(base_selected_indexes)
                n_selected_indexes = 0
                for k in range(np.max(same_eq_shape_S[base_selected_indexes]) + 1):
                    TF = same_eq_shape_S[base_selected_indexes] == k
                    selected_indexes[n_selected_indexes : n_selected_indexes + np.sum(TF)] = base_selected_indexes[TF]
                    n_selected_indexes += np.sum(TF)
                # if n_selected_indexes != selected_indexes.shape[0]:
                #    print("error : n_selected_indexes")
                use = np.arange(i)
                use_pattern = np.empty(i, dtype="int64")
                tot_done_plus = True
                while True:
                    if np.sum(n_max_mat_covered_num[selected_indexes[use]]) >= len_cache_for_mask_x:
                        if len_cache_for_mask_x >= np.sum(n_min_mat_covered_num[selected_indexes[use]]):
                            use_pattern[:] = 0
                            while True:
                                print_counter += 1
                                coverd = True
                                for k in range(len_cache_for_mask_x):
                                    one_coverd = False
                                    for l in range(i):
                                        if mat_covered_num[selected_indexes[use[l]], use_pattern[l], k]:
                                            if one_coverd:
                                                coverd = False
                                                break
                                            else:
                                                one_coverd = True
                                    if not one_coverd:
                                        coverd = False
                                    if not coverd:
                                        break
                                if coverd:
                                    for k in range(i):
                                        mat_use[0, k, 0] = selected_indexes[use[k]]
                                        mat_use[0, k, 1] = use_pattern[k]
                                        mat_use[0, k, 2] = use[k]
                                    mat_use[0, i:] = int_nan
                                    one_found = True
                                    break
                                else:
                                    done_plus_one = False
                                    for l in range(i - 1, -1, -1):
                                        if use_pattern[l] != patterns[selected_indexes[use[l]]] - 1:
                                            use_pattern[l] += 1
                                            done_plus_one = True
                                            break
                                        else:
                                            use_pattern[l] = 0
                                    if not done_plus_one:
                                        break
                    if one_found:
                        break
                    else:
                        tot_done_plus = True
                        for _ in range(num_threads):
                            done_plus = False
                            for l in range(i - 1, -1, -1):
                                if use[l] != n_selected_indexes - (i - l):
                                    use[l] += 1
                                    done_plus = True
                                    break
                                else:
                                    for m in range(l - 1, -1, -1):
                                        if use[m] != n_selected_indexes - (i - m):
                                            use[l] = use[m] + 1 + (l - m)
                                            break
                            if not done_plus:
                                tot_done_plus = False
                                break
                        if not tot_done_plus:
                            break
            if one_found:
                break
        if one_found:
            break
    if one_found:
        n_use = np.sum(mat_use[0, :, 0] != int_nan)
        for m in range(n_use):
            return_mat_use[m, 0] = mat_use[0, m, 0]
            return_mat_use[m, 1] = mat_use[0, m, 1]
        norm_check = len_cache_for_mask_x == len_norm_cache_for_mask_x
        same_norm_num_index = same_norm_num_index[return_mat_use[:n_use, 0]]
        norm_patterns = np.empty((n_use), dtype="int64")
        norm_TF_mask_x = np.ones((n_use, mask_x.shape[0]), dtype="bool")
        for j in range(n_use):
            check_change_x = dict_check_change_x[return_mat_use[j, 0]][return_mat_use[j, 1]]
            len_check_change_x = count_True(check_change_x[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
            norm_TF_mask_x[j] = TF_mask_x[return_mat_use[j, 0]]
            for k in range(mask_x.shape[0]):
                for l in range(len_check_change_x):
                    if mask_x[k, check_change_x[l, 0]] > mask_x[k, check_change_x[l, 1]]:
                        norm_TF_mask_x[j, k] = False
                        break
            arr_norm_check_change_x = make_check_change_x(mask_x, same_norm_num_index[j], norm_TF_mask_x[j])[2]
            dict_norm_check_change_x[j] = arr_norm_check_change_x
            norm_patterns[j] = arr_norm_check_change_x.shape[0]
        n_all_norm_pattern = np.max(norm_patterns)
        mat_covered_norm_num = np.zeros((n_use, n_all_norm_pattern, len_norm_cache_for_mask_x), dtype="bool")

        if norm_check:
            for k in range(n_use):
                calc[k] = k
            calc[n_use:] = int_nan
            calc_pattern[:] = 0
        else:
            for j in range(n_use):
                arr_norm_check_change_x = dict_norm_check_change_x[j]
                for k in range(norm_patterns[j]):
                    for l in range(mask_x.shape[0]):
                        if norm_TF_mask_x[j, l]:
                            check = True
                            for m in range(arr_norm_check_change_x.shape[1]):
                                if int_nan == arr_norm_check_change_x[k, m, 0]:
                                    break
                                elif (
                                    mask_x[l, arr_norm_check_change_x[k, m, 0]]
                                    > mask_x[l, arr_norm_check_change_x[k, m, 1]]
                                ):
                                    check = False
                                    break
                            if check:
                                if mat_covered_norm_num[j, k, same_norm_num_index[j, l]]:
                                    print("ERROR : mat_covered_norm_num")
                                    print(check_exist_eq[indexes[return_mat_use[j, 0]]])
                                    # print(same_num_index[j], same_norm_num_index[j], TF_mask_x[j])
                                    # print(make_check_change_x(mask_x, same_num_index[j], TF_mask_x[j]))
                                else:
                                    mat_covered_norm_num[j, k, same_norm_num_index[j, l]] = True
            for k in range(1, n_use + 1):
                for l in range(k):
                    calc[l] = l
                calc[k:] = int_nan
                while True:
                    calc_pattern[:] = 0
                    while True:
                        coverd = True
                        for l in range(len_norm_cache_for_mask_x):
                            one_coverd = False
                            for m in range(k):
                                if mat_covered_norm_num[calc[m], calc_pattern[m], l]:
                                    if one_coverd:
                                        coverd = False
                                        break
                                    else:
                                        one_coverd = True
                            if not one_coverd:
                                coverd = False
                            if not coverd:
                                break
                        if coverd:
                            norm_check = True
                            break
                        else:
                            done_plus_one = False
                            for l in range(k - 1, -1, -1):
                                if calc_pattern[l] != norm_patterns[calc[l]] - 1:
                                    calc_pattern[l] += 1
                                    done_plus_one = True
                                    break
                                else:
                                    calc_pattern[l] = 0
                            if not done_plus_one:
                                break
                    if norm_check:
                        break
                    else:
                        done_plus_one = False
                        for l in range(k - 1, -1, -1):
                            if calc[l] != n_use - (k - l):
                                calc[l] += 1
                                done_plus_one = True
                                break
                            else:
                                for m in range(l - 1, -1, -1):
                                    if calc[m] != n_use - (k - m):
                                        calc[l] = calc[m] + 1 + (l - m)
                                        break
                        if not done_plus_one:
                            break
                if norm_check:
                    break
        # """# print
        if not norm_check:
            print("not norm_check found")
            print("len_cache_for_mask_x, len_norm_cache_for_mask_x : ", len_cache_for_mask_x, len_norm_cache_for_mask_x)
            for i in range(n_use):
                print("eq : ", check_exist_eq[indexes[return_mat_use[i, 0]]])
                print(mat_covered_norm_num[return_mat_use[i, 0], return_mat_use[i, 1]])
        # """  # print
        if norm_check:
            for m in range(n_use):
                if m in calc:
                    one_eq_calc_check_change_x = dict_norm_check_change_x[m][calc_pattern[m]]
                    _len = one_eq_calc_check_change_x.shape[0]
                    dict_one_eq_calc_check_change_x[return_mat_use[m, 0], :_len] = one_eq_calc_check_change_x
        return return_mat_use, dict_check_change_x, calc, dict_one_eq_calc_check_change_x

    # round robin
    # """ # print
    print("find : round-robin")
    for j in range(indexes.shape[0]):
        print(j)
        print("   eq : ", check_exist_eq[indexes[j]])
        print("   same_eq_shape L, S : ", same_eq_shape_L[j], same_eq_shape_S[j])
        # print("   n_max,min mat_covered_num : ", n_max_mat_covered_num[j], n_min_mat_covered_num[j])
        for k in range(patterns[j]):
            print("   mat_covered_num[", k, "] : ", np.sum(mat_covered_num[j, k]), mat_covered_num[j, k])
        print()
    print("\nERROR : not seleced")
    for j in range(indexes.shape[0]):
        print("   eq : ", check_exist_eq[indexes[j]])
        n_op1 = check_exist_id[indexes[j], 0]
        id1 = check_exist_id[indexes[j], 1]
        n_op2 = max_op - 1 - n_op1
        id2 = check_exist_id[indexes[j], 2]
        back_change_pattern = arr_back_change_pattern[indexes[j]]
        before_check_change_x1 = before_check_change_x_dict[n_op1][id1]
        len_before_check_change_x1 = count_True(before_check_change_x1[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
        print("   check1 : ", before_check_change_x1[:len_before_check_change_x1])
        before_check_change_x2 = before_check_change_x_dict[n_op2][id2]
        changed_before_check_change_x2 = make_eq(before_check_change_x2, back_change_pattern)
        len_changed_before_check_change_x2 = count_True(
            changed_before_check_change_x2[:, 0], 2, int_nan
        )  # 2 -> lambda x: x != border
        print("   check2 : ", changed_before_check_change_x2[:len_changed_before_check_change_x2])
        print("   mat_check_change_x")
        l_mat_check_change_x = []
        arr_check_change_x = dict_check_change_x[j]
        for k in range(patterns[j]):
            l_one_mat_check_change_x = []
            for l in range(arr_check_change_x.shape[1]):
                if arr_check_change_x[k, l, 0] != int_nan:
                    l_one_mat_check_change_x.append(
                        [int(arr_check_change_x[k, l, 0]), int(arr_check_change_x[k, l, 1])]
                    )
            l_mat_check_change_x.append(l_one_mat_check_change_x)
        print(l_mat_check_change_x)
        print("   same_num_index : ", same_num_index[j])
        print("   mat_covered_num : ", [mat_covered_num[j, k] for k in range(patterns[j])])
    # """  # print
    return return_mat_use, dict_check_change_x, calc, dict_one_eq_calc_check_change_x


@njit(parallel=True, error_model="numpy")
def check_exist_step4(same, check_exist_num_arr, check_exist_need_calc, progress_proxy):
    int_nan = -100
    n_check_exist = check_exist_num_arr.shape[0]
    n_pattern = np.max(same) + 1 if same.shape[0] != 0 else 0
    head_indexes = np.full((n_pattern), int_nan, dtype="int64")
    same_need_calc_num = np.full((n_pattern), int_nan, dtype="int64")
    n_eqs = np.zeros(n_pattern, dtype="int64")

    for i in range(n_pattern):
        first = True
        for j in range(n_check_exist):
            if same[j] == i:
                if first:
                    head_indexes[i] = j
                    first = False
                n_eqs[i] += 1
    c = 0
    for i in range(n_pattern):
        head_index = head_indexes[i]
        is_same = False
        for j in range(i):
            target_index = head_indexes[j]
            if isclose(check_exist_num_arr[head_index, 0, 0], check_exist_num_arr[target_index, 0, 0]):
                if isclose_arr(check_exist_num_arr[head_index, 0], check_exist_num_arr[target_index, 0]):
                    same_need_calc_num[i] = same_need_calc_num[j]
                    is_same = True
                    break
        if not is_same:
            same_need_calc_num[i] = c
            c += 1
        progress_proxy.update(1)
    arange = np.arange(n_pattern)
    for i in prange(c):
        indexes = arange[same_need_calc_num == i]
        start = indexes.shape[0]
        for j in range(indexes.shape[0]):
            if np.any(check_exist_need_calc[same == indexes[j]]):
                min_index = indexes[j]
                min_n_eqs = n_eqs[indexes[j]]
                start = j + 1
                break
        for j in indexes[start:]:
            if np.any(check_exist_need_calc[same == j]):
                if min_n_eqs > n_eqs[j]:
                    check_exist_need_calc[same == min_index] = False
                    min_n_eqs = n_eqs[j]
                    min_index = j
                else:
                    check_exist_need_calc[same == j] = False
    return check_exist_need_calc


# max_op = 7


# @njit(error_model="numpy")
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
    save_back_change_pattern = np.full((num_threads, loop_per_threads, max_op), int_nan, dtype="int8")
    save_check_change_x_tot = np.full(
        (num_threads, loop_per_threads, (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    save_check_change_x_ones = np.full(
        (num_threads, loop_per_threads, (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    if n_op1 >= n_op2:
        ops = np.array([-1, -2, -3, -4])
    else:
        ops = np.array([-4])
    len_base_eq_arr1 = base_eq_arr1.shape[0]
    len_base_eq_arr2 = base_eq_arr2.shape[0]
    loop = len_base_eq_arr1 * len_base_eq_arr2
    split_indexes = np.array_split(np.arange(loop), num_threads)
    for thread_id in prange(num_threads):
        equation = np.full((2 * max_op + 1), int_nan, dtype="int8")
        cache_for_mask_x = np.zeros((tot_mask_x.shape[0], random_x.shape[1]), dtype="float64")
        TF_mask_x = np.ones(tot_mask_x.shape[0], dtype="bool")
        norm_same_for_mask_x = np.zeros((tot_mask_x.shape[0]), dtype="int64")
        counter = 0
        for i in split_indexes[thread_id]:
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
                    is_calc, all_covered = True, True
                    back_change_pattern = dict_change_x_pattern[base_back_x_max][number]
                    equation[len_base_eq1 : len_base_eq1 + len_base_eq2] = make_eq(
                        base_back_equation, back_change_pattern
                    )
                    if find_min_x_max(equation, random_x, random_for_find_min_x_max):
                        is_calc = False
                    if is_calc:
                        before_check_change_x1 = before_check_change_x_list1[id1]
                        len_before_check_change_x1 = count_True(
                            before_check_change_x1[:, 0], 2, int_nan
                        )  # 2 -> lambda x: x != border
                        before_check_change_x2 = before_check_change_x_list2[id2]
                        changed_before_check_change_x2 = make_eq(before_check_change_x2, back_change_pattern)
                        len_before_check_change_x2 = count_True(
                            changed_before_check_change_x2[:, 0], 2, int_nan
                        )  # 2 -> lambda x: x != border

                        save_similar_num = nb_calc_RPN(random_x, equation)[0]
                        if save_similar_num[0] == int_nan:  # any_isinf, all_const
                            is_calc = False
                    if is_calc:
                        eq_x_max = np.max(equation)
                        mask_x = dict_mask_x[eq_x_max]
                        norm_c = 0
                        TF_mask_x[: mask_x.shape[0]] = True
                        for k in range(mask_x.shape[0]):
                            for l in range(len_before_check_change_x1):
                                if mask_x[k, before_check_change_x1[l, 0]] > mask_x[k, before_check_change_x1[l, 1]]:
                                    TF_mask_x[k] = False
                                    break
                            if TF_mask_x[k]:
                                for l in range(len_before_check_change_x2):
                                    if (
                                        mask_x[k, changed_before_check_change_x2[l, 0]]
                                        > mask_x[k, changed_before_check_change_x2[l, 1]]
                                    ):
                                        TF_mask_x[k] = False
                                        break
                            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)[0]
                            norm_same = False
                            for l in range(norm_c):
                                if isclose(cache_for_mask_x[l, 0], similar_num[0]):
                                    if isclose_arr(cache_for_mask_x[l], similar_num):
                                        norm_same_for_mask_x[k] = l
                                        norm_same = True
                            if not norm_same:
                                norm_same_for_mask_x[k] = norm_c
                                cache_for_mask_x[norm_c] = similar_num
                                norm_c += 1
                            if save_similar_num[0] > similar_num[0]:  # min
                                save_similar_num[:] = similar_num
                        if not np.any(TF_mask_x[: mask_x.shape[0]]):
                            is_calc = False
                        if is_calc and max_op + 1 != eq_x_max:
                            # except when using x of (number of operators + 1) type
                            for j in range(head_before_similar_num_list.shape[0]):
                                if isclose(save_similar_num[0], head_before_similar_num_list[j]):
                                    if isclose_arr(
                                        save_similar_num,
                                        before_similar_num_list[j, 0],
                                    ):
                                        is_calc = False
                                        break
                                elif save_similar_num[0] < head_before_similar_num_list[j]:
                                    break
                        if is_calc:
                            for j in range(head_saved_similar_num_list.shape[0]):
                                if isclose(save_similar_num[0], head_saved_similar_num_list[j]):
                                    if isclose_arr(
                                        save_similar_num,
                                        saved_similar_num_list[j],
                                    ):
                                        is_calc = False
                                        break
                                elif save_similar_num[0] < head_saved_similar_num_list[j]:
                                    break
                        if is_calc:
                            for j in range(counter):
                                if head_save_similar_num_list[thread_id, j] != int_nan:
                                    if isclose(save_similar_num[0], head_save_similar_num_list[thread_id, j]):
                                        if isclose_arr(
                                            save_similar_num,
                                            save_similar_num_list[thread_id, j],
                                        ):
                                            is_calc = False
                                            break
                        if is_calc:
                            can_use_norm, all_covered, add_check_change_x = make_check_change_x(
                                mask_x, norm_same_for_mask_x[: mask_x.shape[0]], TF_mask_x[: mask_x.shape[0]]
                            )
                            if not can_use_norm:
                                is_calc = False
                                # print("drop : make_check_change_x")
                                # print(equation)
                                # print(same_for_mask_x[: mask_x.shape[0]])
                                # print("arr_similar_num_head : ", arr_similar_num_head[: mask_x.shape[0]])
                                # print(TF_mask_x[: mask_x.shape[0]])
                                # print(can_use, all_covered, add_check_change_x)
                            if is_calc:
                                if not all_covered:
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
                                    save_back_change_pattern[thread_id, counter, : back_change_pattern.shape[0]] = (
                                        back_change_pattern
                                    )
                                    is_calc = False
                        if is_calc:
                            TF_list[thread_id, counter] = True
                            save_similar_num_list[thread_id, counter] = save_similar_num
                            head_save_similar_num_list[thread_id, counter] = save_similar_num[0]
                            save_equation_list[thread_id, counter] = equation
                            save_base_eq_id_list[thread_id, counter, 0] = n_op1
                            save_base_eq_id_list[thread_id, counter, 1] = id1
                            save_base_eq_id_list[thread_id, counter, 2] = id2
                            save_base_eq_id_list[thread_id, counter, 3] = make_change_x_id(
                                back_change_pattern, max_op + 1
                            )
                            save_base_eq_id_list[thread_id, counter, 4] = merge_op
                            n = 0
                            for k in range(len_before_check_change_x1):
                                save_check_change_x_tot[thread_id, counter, n] = before_check_change_x1[k]
                                n += 1
                            for k in range(len_before_check_change_x2):
                                unique = True
                                for l in range(len_before_check_change_x1):
                                    if changed_before_check_change_x2[k, 0] == before_check_change_x1[l, 0]:
                                        if changed_before_check_change_x2[k, 1] == before_check_change_x1[l, 1]:
                                            unique = False
                                            break
                                if unique:
                                    save_check_change_x_tot[thread_id, counter, n] = changed_before_check_change_x2[k]
                                    n += 1
                            for k in range(add_check_change_x.shape[1]):
                                if add_check_change_x[0, k, 0] != int_nan:
                                    save_check_change_x_tot[thread_id, counter, n] = add_check_change_x[0, k]
                                    save_check_change_x_ones[thread_id, counter, k] = add_check_change_x[0, k]
                                    n += 1
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
        save_back_change_pattern,
    )


@njit(parallel=True, error_model="numpy")
def dim_reduction_7(TF_list, similar_num_list, progress_proxy):
    int_nan = -100
    num_threads = similar_num_list.shape[0]
    return_similar_num_list = np.full((np.sum(TF_list), similar_num_list.shape[2]), int_nan, dtype="float64")
    return_similar_num_list[: np.sum(TF_list[0])] = similar_num_list[0, TF_list[0]]
    head_return_similar_num_list = np.full((np.sum(TF_list)), int_nan, dtype="float64")
    head_similar_num_list = similar_num_list[:, :, 0].copy()
    head_return_similar_num_list[: np.sum(TF_list[0])] = head_similar_num_list[0, TF_list[0]]
    last_index = np.sum(TF_list[0])
    for target_index in range(1, num_threads):
        true_index = np.arange(TF_list.shape[1])[TF_list[target_index]]
        for thread_id in prange(num_threads):
            true_index_thread = true_index[thread_id::num_threads].copy()
            for i in true_index_thread:
                head_target = similar_num_list[target_index, i, 0]
                for j in range(last_index):
                    if isclose(head_target, head_return_similar_num_list[j]):
                        if isclose_arr(
                            similar_num_list[target_index, i],
                            return_similar_num_list[j],
                        ):
                            TF_list[target_index, i] = False
                            break
                progress_proxy.update(1)
        next_index = last_index + np.sum(TF_list[target_index])
        return_similar_num_list[last_index:next_index] = similar_num_list[target_index, TF_list[target_index]]
        head_return_similar_num_list[last_index:next_index] = head_similar_num_list[target_index, TF_list[target_index]]
        last_index = next_index
    # return TF_list


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
    for i in range(num_threads):
        indexes = np.arange(TF_list.shape[1])[TF_list[i]]
        return_similar_num_list[last_index : last_index + indexes.shape[0]] = similar_num_list[i, indexes]
        return_equation_list[last_index : last_index + indexes.shape[0]] = equation_list[i, indexes]
        return_base_eq_id_list[last_index : last_index + indexes.shape[0]] = base_eq_id_list[i, indexes]
        return_check_change_x_tot[last_index : last_index + indexes.shape[0]] = check_change_x_tot[i, indexes]
        return_check_change_x_ones[last_index : last_index + indexes.shape[0]] = check_change_x_ones[i, indexes]
        last_index += indexes.shape[0]
    return (
        return_equation_list,
        return_similar_num_list,
        return_base_eq_id_list,
        return_check_change_x_tot,
        return_check_change_x_ones,
    )


@njit(parallel=True, error_model="numpy")
def make_check_exist_info_7(check_exist_TF, similar_num_list, equation_list, base_eq_id_list, back_change_pattern):
    int_nan = -100
    num_threads = check_exist_TF.shape[0]
    sum_check_exist = np.sum(check_exist_TF)
    return_check_exist_num = np.full((sum_check_exist, similar_num_list.shape[2]), int_nan, dtype="float64")
    return_check_exist_eq = np.full((sum_check_exist, equation_list.shape[2]), int_nan, dtype="int8")
    return_check_exist_id = np.full((sum_check_exist, 5), int_nan, dtype="int64")
    return_back_change_pattern = np.full((sum_check_exist, back_change_pattern.shape[2]), int_nan, dtype="int8")
    last_index = 0
    for i in range(num_threads):
        indexes = np.arange(check_exist_TF.shape[1])[check_exist_TF[i]]
        return_check_exist_num[last_index : last_index + indexes.shape[0]] = similar_num_list[i, indexes]
        return_check_exist_eq[last_index : last_index + indexes.shape[0]] = equation_list[i, indexes]
        return_check_exist_id[last_index : last_index + indexes.shape[0]] = base_eq_id_list[i, indexes]
        return_back_change_pattern[last_index : last_index + indexes.shape[0]] = back_change_pattern[i, indexes]
        last_index += indexes.shape[0]
    return (
        return_check_exist_num[:last_index],
        return_check_exist_eq[:last_index],
        return_check_exist_id[:last_index],
        return_back_change_pattern[:last_index],
    )


@njit(parallel=True, error_model="numpy")
def check_exist_step1_7(similar_num_list, before_similar_num_list, check_exist_num_arr, progress_proxy):
    int_nan = -100
    head_saved_similar_num_list = similar_num_list[:, 0].copy()
    head_before_similar_num_list = before_similar_num_list[:, 0, 0].copy()
    len_return = check_exist_num_arr.shape[0]
    TF = np.zeros((len_return), dtype="bool")
    for i in prange(check_exist_num_arr.shape[0]):
        is_calc = True
        for j in range(head_before_similar_num_list.shape[0]):
            if isclose(check_exist_num_arr[i, 0], head_before_similar_num_list[j]):
                if isclose_arr(check_exist_num_arr[i], before_similar_num_list[j]):
                    is_calc = False
                    break
            elif check_exist_num_arr[i, 0] < head_before_similar_num_list[j]:
                break
        if is_calc:
            for j in range(head_saved_similar_num_list.shape[0]):
                if isclose(check_exist_num_arr[i, 0], head_saved_similar_num_list[j]):
                    if isclose_arr(check_exist_num_arr[i], similar_num_list[j]):
                        is_calc = False
                        break
                elif check_exist_num_arr[i, 0] < head_saved_similar_num_list[j]:
                    break
        TF[i] = is_calc
        progress_proxy.update(1)
    return TF


@njit(parallel=True, error_model="numpy")
def check_exist_step2_7(num_threads, check_exist_num_arr, progress_proxy):
    int_nan = -100
    n_check_exist = check_exist_num_arr.shape[0]
    arg_index = np.argsort(check_exist_num_arr[:, 0])
    same = np.full((n_check_exist), int_nan, dtype="int64")
    count = 0
    filled_count = np.zeros(num_threads, dtype="int64")
    for i in range(n_check_exist):
        if same[arg_index[i]] == int_nan:
            same[arg_index[i]] = count
            filled_count[:] = 0
            for thread_id in prange(num_threads):
                for j in range(i + 1 + thread_id, n_check_exist, num_threads):
                    if same[arg_index[j]] == int_nan:
                        if isclose(check_exist_num_arr[arg_index[i], 0], check_exist_num_arr[arg_index[j], 0]):
                            if isclose_arr(check_exist_num_arr[arg_index[i]], check_exist_num_arr[arg_index[j]]):
                                same[arg_index[j]] = count
                                filled_count[thread_id] += 1
                        else:
                            break
            count += 1
            progress_proxy.update(np.sum(filled_count) + 1)
    change_nums = np.full(count, int_nan, dtype="int64")
    base_same = same.copy()
    n = 0
    for i in range(n_check_exist):
        if change_nums[base_same[i]] == int_nan:
            change_nums[base_same[i]] = n
            same[i] = n
            n += 1
        else:
            same[i] = change_nums[base_same[i]]
    return same


# @njit(error_model="numpy")
@njit(parallel=True, error_model="numpy")
def check_exist_step3_7(
    num_threads,
    random_x,
    arr_same,
    check_exist_eq,
    check_exist_id,
    arr_back_change_pattern,
    progress_proxy,
):
    max_op = 7
    dict_mask_x = make_dict_mask_x(max_op + 1)
    before_check_change_x_dict = cache_check_change_x_load(max_op - 1)
    arange = np.arange(check_exist_eq.shape[0])
    int_nan = -100
    if arr_same.shape[0] == 0:
        loop = 0
    else:
        loop = np.max(arr_same) + 1
    TF = np.zeros(check_exist_eq.shape[0], dtype="bool")
    save_check_change_x_tot = np.full((check_exist_eq.shape[0], (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8")
    save_check_change_x_ones = np.full(
        (check_exist_eq.shape[0], (max_op * (max_op + 1)) // 2, 2), int_nan, dtype="int8"
    )
    for i in prange(loop):
        indexes = arange[arr_same == i]
        mat_use, dict_check_change_x = sub_check_exist_step3_7(
            num_threads,
            random_x,
            check_exist_eq,
            check_exist_id,
            arr_back_change_pattern,
            dict_mask_x,
            before_check_change_x_dict,
            indexes,
        )
        len_mat_use = count_True(mat_use[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
        for m in range(len_mat_use):
            TF[indexes[mat_use[m, 0]]] = True
            add = dict_check_change_x[mat_use[m, 0]][mat_use[m, 1]]
            len_add = count_True(add[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
            n_op1 = check_exist_id[indexes[mat_use[m, 0]], 0]
            id1 = check_exist_id[indexes[mat_use[m, 0]], 1]
            n_op2 = max_op - 1 - n_op1
            id2 = check_exist_id[indexes[mat_use[m, 0]], 2]
            back_change_pattern = arr_back_change_pattern[indexes[mat_use[m, 0]]]
            before_check_change_x1 = before_check_change_x_dict[n_op1][id1]
            len_before1 = count_True(before_check_change_x1[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
            before_check_change_x2 = before_check_change_x_dict[n_op2][id2]
            changed_before_check_change_x2 = make_eq(before_check_change_x2, back_change_pattern)
            len_before2 = count_True(changed_before_check_change_x2[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
            save_check_change_x_ones[indexes[mat_use[m, 0]], :len_add] = add[:len_add]
            save_check_change_x_tot[indexes[mat_use[m, 0]], :len_before1] = before_check_change_x1[:len_before1]
            save_check_change_x_tot[indexes[mat_use[m, 0]], len_before1 : len_before1 + len_before2] = (
                changed_before_check_change_x2[:len_before2]
            )
            save_check_change_x_tot[
                indexes[mat_use[m, 0]],
                len_before1 + len_before2 : len_before1 + len_before2 + len_add,
            ] = add[:len_add]
        progress_proxy.update(1)
    return (
        TF,
        save_check_change_x_ones,
        save_check_change_x_tot,
    )


# @njit(parallel=True, error_model="numpy")
@njit(error_model="numpy")
def sub_check_exist_step3_7(
    num_threads,
    random_x,
    check_exist_eq,
    check_exist_id,
    arr_back_change_pattern,
    dict_mask_x,
    before_check_change_x_dict,
    indexes,
):
    max_op = 7
    print_counter = 0
    lim_print_counter = 100000000
    printed = False

    int_nan = -100
    arange = np.arange(indexes.shape[0])
    equation = check_exist_eq[indexes[0]]
    eq_x_max = np.max(equation)
    mask_x = dict_mask_x[eq_x_max]
    mat_use = np.empty((num_threads, indexes.shape[0], 3), dtype="int64")
    return_mat_use = np.full((indexes.shape[0], 2), int_nan, dtype="int64")
    norm_cache_for_mask_x = np.zeros((mask_x.shape[0], random_x.shape[1]), dtype="float64")
    TF_mask_x = np.ones((indexes.shape[0], mask_x.shape[0]), dtype="bool")
    norm_same_for_mask_x = np.zeros((mask_x.shape[0]), dtype="int64")
    dict_check_change_x = dict()
    norm_n = 0
    for k in range(mask_x.shape[0]):
        similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)[0]
        norm_same = False
        for l in range(norm_n):
            if isclose(norm_cache_for_mask_x[l, 0], similar_num[0]):
                if isclose_arr(norm_cache_for_mask_x[l], similar_num):
                    norm_same = True
                    break
        if not norm_same:
            norm_cache_for_mask_x[norm_n] = similar_num
            norm_n += 1
    norm_cache_for_mask_x = norm_cache_for_mask_x[:norm_n]
    len_norm_cache_for_mask_x = norm_n

    same_norm_num_index = np.empty((indexes.shape[0], mask_x.shape[0]), dtype="int64")
    patterns = np.empty((indexes.shape[0]), dtype="int64")
    use = np.full((indexes.shape[0]), int_nan, dtype="int64")
    n_max_mat_covered_num = np.empty((indexes.shape[0]), dtype="int64")
    n_min_mat_covered_num = np.empty((indexes.shape[0]), dtype="int64")

    for j in range(indexes.shape[0]):
        equation = check_exist_eq[indexes[j]]
        n_op1 = check_exist_id[indexes[j], 0]
        id1 = check_exist_id[indexes[j], 1]
        n_op2 = max_op - 1 - n_op1
        id2 = check_exist_id[indexes[j], 2]
        back_change_pattern = arr_back_change_pattern[indexes[j]]
        before_check_change_x1 = before_check_change_x_dict[n_op1][id1]
        before_check_change_x2 = before_check_change_x_dict[n_op2][id2]
        changed_before_check_change_x2 = make_eq(before_check_change_x2, back_change_pattern)
        len_before_check_change_x1 = count_True(before_check_change_x1[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
        len_before_check_change_x2 = count_True(
            changed_before_check_change_x2[:, 0], 2, int_nan
        )  # 2 -> lambda x: x != border
        TF_mask_x[j] = True
        for k in range(mask_x.shape[0]):
            for l in range(len_before_check_change_x1):
                if mask_x[k, before_check_change_x1[l, 0]] > mask_x[k, before_check_change_x1[l, 1]]:
                    TF_mask_x[j, k] = False
                    break
            if TF_mask_x[j, k]:
                for l in range(len_before_check_change_x2):
                    if (
                        mask_x[k, changed_before_check_change_x2[l, 0]]
                        > mask_x[k, changed_before_check_change_x2[l, 1]]
                    ):
                        TF_mask_x[j, k] = False
                        break
            similar_num = nb_calc_RPN(random_x[mask_x[k]], equation)[0]
            for l in range(len_norm_cache_for_mask_x):
                if isclose(norm_cache_for_mask_x[l, 0], similar_num[0]):
                    if isclose_arr(norm_cache_for_mask_x[l], similar_num):
                        norm_same_for_mask_x[k] = l
                        break
        same_norm_num_index[j] = norm_same_for_mask_x
        arr_check_change_x = make_check_change_x(mask_x, norm_same_for_mask_x, TF_mask_x[j])[2]
        dict_check_change_x[j] = arr_check_change_x
        patterns[j] = arr_check_change_x.shape[0]
    n_all_pattern = np.max(patterns)

    mat_covered_num = np.zeros((indexes.shape[0], n_all_pattern, len_norm_cache_for_mask_x), dtype="bool")
    for j in range(indexes.shape[0]):
        arr_check_change_x = dict_check_change_x[j]
        for k in range(patterns[j]):
            for l in range(mask_x.shape[0]):
                if TF_mask_x[j, l]:
                    check = True
                    for m in range(arr_check_change_x.shape[1]):
                        if int_nan == arr_check_change_x[k, m, 0]:
                            break
                        elif mask_x[l, arr_check_change_x[k, m, 0]] > mask_x[l, arr_check_change_x[k, m, 1]]:
                            check = False
                            break
                    if check:
                        if mat_covered_num[j, k, same_norm_num_index[j, l]]:
                            print("ERROR : mat_covered_num")
                            print(check_exist_eq[indexes[j]])
                            print(same_norm_num_index[j], TF_mask_x[j])
                        else:  # if not mat_covered_num[j, k, same_num_index[j, l]]:
                            mat_covered_num[j, k, same_norm_num_index[j, l]] = True
        _arr = np.array([np.sum(mat_covered_num[j, k]) for k in range(patterns[j])])
        n_max_mat_covered_num[j] = np.max(_arr)
        n_min_mat_covered_num[j] = np.min(_arr)

    # same eqs (1 op is safe)
    same_eq_shape_S = np.full((indexes.shape[0]), int_nan, dtype="int64")
    c = 0
    for j in range(indexes.shape[0]):
        max_c = 0
        for k in range(j):
            if same_eq_shape_S[k] == max_c:
                is_same = True
                for l in range(check_exist_eq.shape[1]):
                    if check_exist_eq[indexes[j], l] <= 0:
                        if check_exist_eq[indexes[j], l] != check_exist_eq[indexes[k], l]:
                            is_same = False
                            break
                    elif check_exist_eq[indexes[k], l] <= 0:
                        is_same = False
                        break
                if is_same:
                    same_eq_shape_S[j] = same_eq_shape_S[k]
                    break
                else:
                    max_c += 1
        if same_eq_shape_S[j] == int_nan:
            same_eq_shape_S[j] = c
            c += 1
    same_eq_shape_L = np.full((indexes.shape[0]), int_nan, dtype="int64")
    c = 0
    for j in range(indexes.shape[0]):
        max_c = 0
        for k in range(j):
            if same_eq_shape_L[k] == max_c:
                is_same = True
                for l in range(check_exist_eq.shape[1]):
                    if check_exist_eq[indexes[j], l] == 0:
                        if check_exist_eq[indexes[k], l] != 0:
                            is_same = False
                            break
                    elif check_exist_eq[indexes[k], l] == 0:
                        is_same = False
                        break
                    elif check_exist_eq[indexes[j], l] in [-1, -2]:
                        if not check_exist_eq[indexes[k], l] in [-1, -2]:
                            is_same = False
                            break
                    elif check_exist_eq[indexes[k], l] in [-1, -2]:
                        is_same = False
                        break
                    elif check_exist_eq[indexes[j], l] in [-3, -4]:
                        if check_exist_eq[indexes[j], l] != check_exist_eq[indexes[k], l]:
                            is_same = False
                            break
                    elif check_exist_eq[indexes[k], l] in [-3, -4]:
                        is_same = False
                        break
                if is_same:
                    same_eq_shape_L[j] = same_eq_shape_L[k]
                    break
                else:
                    max_c += 1
        if same_eq_shape_L[j] == int_nan:
            same_eq_shape_L[j] = c
            c += 1
    len_n_same_eq_shape_L = c

    arr_n_min = np.empty(len_n_same_eq_shape_L, dtype="int64")
    arr_n_max = np.empty(len_n_same_eq_shape_L, dtype="int64")
    for i in range(len_n_same_eq_shape_L):
        selected_indexes = arange[same_eq_shape_L == i]
        n_selected_indexes = selected_indexes.shape[0]
        selected_n_max_mat_covered_num = np.sort(n_max_mat_covered_num[selected_indexes])[::-1]
        selected_n_min_mat_covered_num = np.sort(n_min_mat_covered_num[selected_indexes])
        sum_ = 0
        arr_n_min[i] = n_selected_indexes + 1
        for j in range(n_selected_indexes):
            sum_ += selected_n_max_mat_covered_num[j]
            if sum_ >= len_norm_cache_for_mask_x:
                arr_n_min[i] = j + 1
                break
        sum_ = 0
        arr_n_max[i] = n_selected_indexes
        for j in range(n_selected_indexes):
            sum_ += selected_n_min_mat_covered_num[j]
            if sum_ == len_norm_cache_for_mask_x:
                arr_n_max[i] = j + 1
                break
            elif sum_ > len_norm_cache_for_mask_x:
                arr_n_max[i] = j
                break

    mat_use = np.full((len_n_same_eq_shape_L, indexes.shape[0], 2), int_nan, dtype="int64")
    arr_coverd = np.empty(len_norm_cache_for_mask_x, dtype="bool")
    for i in range(len_n_same_eq_shape_L):
        base_selected_indexes = arange[same_eq_shape_L == i]
        selected_indexes = np.empty_like(base_selected_indexes)
        n_selected_indexes = 0
        for j in range(np.max(same_eq_shape_S[base_selected_indexes]) + 1):
            TF = same_eq_shape_S[base_selected_indexes] == j
            selected_indexes[n_selected_indexes : n_selected_indexes + np.sum(TF)] = base_selected_indexes[TF]
            n_selected_indexes += np.sum(TF)
        len_mat_n_covered_num = 0
        for j in selected_indexes:
            len_mat_n_covered_num += patterns[j]
        base_mat_n_covered_num = np.empty((len_mat_n_covered_num, 3), dtype="int64")
        c_mat_n_covered_num = 0
        for j in selected_indexes:
            for k in range(patterns[j]):
                base_mat_n_covered_num[c_mat_n_covered_num, 0] = j
                base_mat_n_covered_num[c_mat_n_covered_num, 1] = k
                base_mat_n_covered_num[c_mat_n_covered_num, 2] = np.sum(mat_covered_num[j, k])
                c_mat_n_covered_num += 1
        mat_n_covered_num = np.empty((len_mat_n_covered_num, 3), dtype="int64")
        c_mat_n_covered_num = 0
        for j in range(np.max(base_mat_n_covered_num[:, 2]), -1, -1):
            same_n_covered_num_indexes = np.arange(len_mat_n_covered_num)[base_mat_n_covered_num[:, 2] == j]
            mat_n_covered_num[c_mat_n_covered_num : c_mat_n_covered_num + same_n_covered_num_indexes.shape[0]] = (
                base_mat_n_covered_num[same_n_covered_num_indexes]
            )
            c_mat_n_covered_num += same_n_covered_num_indexes.shape[0]
        for start_index in range(len_mat_n_covered_num):
            arr_coverd[:] = mat_covered_num[mat_n_covered_num[start_index, 0], mat_n_covered_num[start_index, 1]]
            mat_use[i, 0, 0] = mat_n_covered_num[start_index, 0]
            mat_use[i, 0, 1] = mat_n_covered_num[start_index, 1]
            c_mat_use = 1
            for j in range(start_index + 1, len_mat_n_covered_num):
                if np.all(arr_coverd):
                    break
                if not mat_n_covered_num[j, 0] in mat_use[i, :c_mat_use, 0]:
                    for k in range(len_norm_cache_for_mask_x):
                        if arr_coverd[k]:
                            if mat_covered_num[mat_n_covered_num[j, 0], mat_n_covered_num[j, 1], k]:
                                break
                    else:
                        for k in range(len_norm_cache_for_mask_x):
                            if not arr_coverd[k]:
                                if mat_covered_num[mat_n_covered_num[j, 0], mat_n_covered_num[j, 1], k]:
                                    arr_coverd[k] = True
                        mat_use[i, c_mat_use, 0] = mat_n_covered_num[j, 0]
                        mat_use[i, c_mat_use, 1] = mat_n_covered_num[j, 1]
                        c_mat_use += 1
            if not np.all(arr_coverd):
                mat_use[i] = int_nan
            else:
                break
    if np.any(mat_use[:, 0, 0] != int_nan):
        found_index = np.arange(len_n_same_eq_shape_L)[mat_use[:, 0, 0] != int_nan]
        use_index = found_index[0]
        n_use = np.sum(mat_use[use_index, :, 0] != int_nan)
        for i in found_index[1:]:
            if n_use > np.sum(mat_use[i, :, 0] != int_nan):
                use_index = i
                n_use = np.sum(mat_use[i, :, 0] != int_nan)
        for i in range(n_use):
            return_mat_use[i, 0] = mat_use[use_index, i, 0]
            return_mat_use[i, 1] = mat_use[use_index, i, 1]
        return return_mat_use, dict_check_change_x

    print("not found aming")
    one_found = False
    for i in range(2, indexes.shape[0] + 1):
        for j in range(len_n_same_eq_shape_L):
            if (arr_n_min[j] <= i) and (i <= arr_n_max[j]):
                if not printed and (print_counter >= lim_print_counter):
                    printed = True
                    # """  # print
                    print("   arr_n_min : ", arr_n_min)
                    print("   arr_n_max : ", arr_n_max)
                    for j in range(indexes.shape[0]):
                        print(j)
                        print("   eq : ", check_exist_eq[indexes[j]])
                        print("   same_eq_shape L, S : ", same_eq_shape_L[j], same_eq_shape_S[j])
                        print("   n_max,min mat_covered_num : ", n_max_mat_covered_num[j], n_min_mat_covered_num[j])
                        print("   mat_covered_num[0] : ", mat_covered_num[j, 0])
                        print()
                    print()
                    # """  # print
                base_selected_indexes = arange[same_eq_shape_L == j]
                selected_indexes = np.empty_like(base_selected_indexes)
                n_selected_indexes = 0
                for k in range(np.max(same_eq_shape_S[base_selected_indexes]) + 1):
                    TF = same_eq_shape_S[base_selected_indexes] == k
                    selected_indexes[n_selected_indexes : n_selected_indexes + np.sum(TF)] = base_selected_indexes[TF]
                    n_selected_indexes += np.sum(TF)
                # if n_selected_indexes != selected_indexes.shape[0]:
                #    print("error : n_selected_indexes")
                use = np.arange(i)
                use_pattern = np.empty(i, dtype="int64")
                tot_done_plus = True
                while True:
                    if np.sum(n_max_mat_covered_num[selected_indexes[use]]) >= len_norm_cache_for_mask_x:
                        if len_norm_cache_for_mask_x >= np.sum(n_min_mat_covered_num[selected_indexes[use]]):
                            use_pattern[:] = 0
                            while True:
                                print_counter += 1
                                coverd = True
                                for k in range(len_norm_cache_for_mask_x):
                                    one_coverd = False
                                    for l in range(i):
                                        if mat_covered_num[selected_indexes[use[l]], use_pattern[l], k]:
                                            if one_coverd:
                                                coverd = False
                                                break
                                            else:
                                                one_coverd = True
                                    if not one_coverd:
                                        coverd = False
                                    if not coverd:
                                        break
                                if coverd:
                                    for k in range(i):
                                        mat_use[0, k, 0] = selected_indexes[use[k]]
                                        mat_use[0, k, 1] = use_pattern[k]
                                        mat_use[0, k, 2] = use[k]
                                    mat_use[0, i:] = int_nan
                                    one_found = True
                                    break
                                else:
                                    done_plus_one = False
                                    for l in range(i - 1, -1, -1):
                                        if use_pattern[l] != patterns[selected_indexes[use[l]]] - 1:
                                            use_pattern[l] += 1
                                            done_plus_one = True
                                            break
                                        else:
                                            use_pattern[l] = 0
                                    if not done_plus_one:
                                        break
                    if one_found:
                        break
                    else:
                        tot_done_plus = True
                        for _ in range(num_threads):
                            done_plus = False
                            for l in range(i - 1, -1, -1):
                                if use[l] != n_selected_indexes - (i - l):
                                    use[l] += 1
                                    done_plus = True
                                    break
                                else:
                                    for m in range(l - 1, -1, -1):
                                        if use[m] != n_selected_indexes - (i - m):
                                            use[l] = use[m] + 1 + (l - m)
                                            break
                            if not done_plus:
                                tot_done_plus = False
                                break
                        if not tot_done_plus:
                            break
            if one_found:
                break
        if one_found:
            break
    if one_found:
        n_use = np.sum(mat_use[0, :, 0] != int_nan)
        for m in range(n_use):
            return_mat_use[m, 0] = mat_use[0, m, 0]
            return_mat_use[m, 1] = mat_use[0, m, 1]
        return return_mat_use, dict_check_change_x

    # round robin
    # """ # print
    print("find : round-robin")
    for j in range(indexes.shape[0]):
        print(j)
        print("   eq : ", check_exist_eq[indexes[j]])
        print("   same_eq_shape L, S : ", same_eq_shape_L[j], same_eq_shape_S[j])
        # print("   n_max,min mat_covered_num : ", n_max_mat_covered_num[j], n_min_mat_covered_num[j])
        for k in range(patterns[j]):
            print("   mat_covered_num[", k, "] : ", np.sum(mat_covered_num[j, k]), mat_covered_num[j, k])
        print()
    print("\nERROR : not seleced")
    for j in range(indexes.shape[0]):
        print("   eq : ", check_exist_eq[indexes[j]])
        n_op1 = check_exist_id[indexes[j], 0]
        id1 = check_exist_id[indexes[j], 1]
        n_op2 = max_op - 1 - n_op1
        id2 = check_exist_id[indexes[j], 2]
        back_change_pattern = arr_back_change_pattern[indexes[j]]
        before_check_change_x1 = before_check_change_x_dict[n_op1][id1]
        len_before_check_change_x1 = count_True(before_check_change_x1[:, 0], 2, int_nan)  # 2 -> lambda x: x != border
        print("   check1 : ", before_check_change_x1[:len_before_check_change_x1])
        before_check_change_x2 = before_check_change_x_dict[n_op2][id2]
        changed_before_check_change_x2 = make_eq(before_check_change_x2, back_change_pattern)
        len_changed_before_check_change_x2 = count_True(
            changed_before_check_change_x2[:, 0], 2, int_nan
        )  # 2 -> lambda x: x != border
        print("   check2 : ", changed_before_check_change_x2[:len_changed_before_check_change_x2])
        print("   mat_check_change_x")
        l_mat_check_change_x = []
        arr_check_change_x = dict_check_change_x[j]
        for k in range(patterns[j]):
            l_one_mat_check_change_x = []
            for l in range(arr_check_change_x.shape[1]):
                if arr_check_change_x[k, l, 0] != int_nan:
                    l_one_mat_check_change_x.append(
                        [int(arr_check_change_x[k, l, 0]), int(arr_check_change_x[k, l, 1])]
                    )
            l_mat_check_change_x.append(l_one_mat_check_change_x)
        print(l_mat_check_change_x)
        print("   same_num_index : ", same_norm_num_index[j])
        print("   mat_covered_num : ", [mat_covered_num[j, k] for k in range(patterns[j])])
    # """  # print
    return return_mat_use, dict_check_change_x
