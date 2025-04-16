import numpy as np
from numba import njit, prange, objmode
import gc, os, psutil


@njit(error_model="numpy")
def cache_load(max_ops):
    unique_eq_dict = dict()
    for i in range(max_ops + 1):
        with objmode(data="int8[:,:]"):
            data = np.load(f"operator_{i}.npy")
        unique_eq_dict[i] = data
    return unique_eq_dict


@njit(error_model="numpy")
def cache_check_change_x_load(max_ops):
    int_nan = -100
    check_change_x_dict = dict()
    for i in range(max_ops + 1):
        with objmode(data="int8[:,:,:]"):
            data = np.load(f"check_change_x_tot_{i}.npz")["arr_0"]
        check_change_x_dict[i] = data
    return check_change_x_dict


@njit(error_model="numpy")
def nb_permutations(pattern, length):
    # pattern :1d ndarray, likely np.arange(1,10,2)
    # length :int, length<=pattern.shape[0]
    loop = 1
    n_pattern = pattern.shape[0]
    for i in range(length):
        loop *= n_pattern - i
    return_arr = np.zeros((loop, length), dtype="int8")
    list_index_pattern = np.arange(n_pattern)
    for i in range(loop):
        mask = np.ones(n_pattern, dtype="bool")
        num = i
        for j in range(length):
            index_pattern = (list_index_pattern[mask])[num % (n_pattern - j)]
            return_arr[i, j] = pattern[index_pattern]
            num //= n_pattern - j
            mask[index_pattern] = False
    return return_arr


@njit(error_model="numpy")
def make_dict_change_x_pattern(max_ops):
    dict_change_x_pattern = dict()
    dict_change_x_pattern[0] = np.zeros((1, 1), dtype="int8")
    dict_max_loop = np.ones((max_ops + 1, max_ops + 1), dtype="int64")
    for i in range(1, max_ops + 1):
        change_x_pattern = nb_permutations(np.arange(1, max_ops + 2), i)  # not use 0,it is const(1)
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
        dict_change_x_pattern[i] = change_x_pattern[np.argsort(max_drop_num)].astype("int8")
        dict_max_loop[i] = np.array([np.sum(max_drop_num <= j) for j in range(np.max(max_drop_num) + 1)])
    return dict_change_x_pattern, dict_max_loop


@njit(error_model="numpy")
def make_eq(base_back_equation, change_x_pattern):
    int_nan = -100
    return_equation = base_back_equation.copy()
    for i in range(return_equation.size):
        if return_equation.flat[i] > 0:
            return_equation.flat[i] = change_x_pattern[return_equation.flat[i] - 1]
    return return_equation


"""
@njit(error_model="numpy")
def make_check_change_x(base_check_change_x, change_x_pattern):
    return_check_change_x = base_check_change_x.copy()
    for i in range(1, np.max(base_check_change_x) + 1):
        for j in range(base_check_change_x.shape[0]):
            if base_check_change_x[j, 0] == i:
                return_check_change_x[j, 0] = change_x_pattern[i - 1]
            if base_check_change_x[j, 1] == i:
                return_check_change_x[j, 1] = change_x_pattern[i - 1]
    return return_check_change_x
"""


@njit(error_model="numpy")
def nb_calc_RPN(x, equation):
    int_nan = -100
    n_sample = x.shape[1]
    stack = np.empty((count_True(equation, 6, 0), n_sample), dtype="float64")  # 6 -> lambda x: x >= border
    next_index = 0
    for i in range(equation.shape[0]):
        last_op = equation[i]
        if last_op >= 0:
            set_x(stack[next_index], x[last_op])
            next_index += 1
        elif last_op == int_nan:
            break
        else:
            calc_arr_binary_op(last_op, stack[next_index - 2], stack[next_index - 1])
            next_index -= 1
    return_stack = np.empty((2, n_sample), dtype="float64")
    if is_inf(stack[0]):
        return_stack[:] = int_nan
        return return_stack
    if is_all_zero(stack[0]):
        return_stack[0] = int_nan
        return_stack[1] = 0
        return return_stack
    if is_all_const(stack[0]):
        return_stack[0] = int_nan
        return_stack[1] = np.sign(stack[0, 0]) * stack[0]
        return return_stack
    use_index = np.argsort(stack[0])[n_sample // 4 : -(n_sample // 4)]
    _mean = np.mean(stack[0, use_index])
    plus_minus_0 = np.sign(stack[0, 0] - _mean)
    plus_minus_1 = np.sign(stack[0, 0])
    _sum = 0.0
    for i in use_index:
        _sum += (stack[0, i] - _mean) ** 2
    std = np.sqrt(_sum / n_sample)
    for i in range(n_sample):
        return_stack[0, i] = plus_minus_0 * (stack[0, i] - _mean) / std
        # return_stack[1, i] = stack[0, i]
        return_stack[1, i] = plus_minus_1 * stack[0, i]
    return return_stack


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
        case -4:  # a / b
            for i in range(arr1.shape[0]):
                arr1[i] /= arr2[i]


@njit(error_model="numpy")
def isclose(a, b, rtol=1e-04, atol=0):
    return np.abs(a - b) <= atol + rtol * min(np.abs(b), np.abs(a))


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


"""
@njit(error_model="numpy")
def isclose_arr(arr1, arr2, rtol=1e-06, atol=0):
    # if np.all(isclose(arr1, arr2, rtol=rtol, atol=atol)):
    #    return True
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
def arg_close_arr(arr, mat, rtol=1e-06, atol=0):
    int_nan = -100
    for i in range(mat.shape[0]):
        if isclose_arr(arr, mat[i], rtol=rtol, atol=atol):
            return i
    return int_nan


@njit(error_model="numpy")
def find_min_x_max(equation, random_x, random_for_find_min_x_max, rtol=1e-06, atol=0):
    # a*(b-b) -> drop
    if count_True(equation, 5, 0) == 0:  # 5 -> lambda x: x > border
        return False
    similar_num = nb_calc_RPN(random_x, equation)[0]
    max_x = np.max(equation)
    counter = np.zeros(max_x + 1, dtype="int8")
    for i in range(1, max_x + 1):
        counter[i] = np.sum(equation == i)
    change_num = np.arange(max_x + 1)[counter >= 2]  # (b-b) -> count = 2, counter >= 2
    changed_random = random_x.copy()
    for i in change_num:
        for j in range(random_for_find_min_x_max.shape[0]):
            changed_random[i, j] = random_for_find_min_x_max[j]
        changed_similar_num = nb_calc_RPN(changed_random, equation)[0]
        if isclose(similar_num[0], changed_similar_num[0], rtol=rtol, atol=atol):
            if isclose_arr(similar_num, changed_similar_num, rtol=rtol, atol=atol):
                return True
        changed_random[i] = random_x[i]
    return False


@njit(error_model="numpy")
def make_dict_mask_x(eq_x_max):
    dict_mask_x = dict()
    for i in range(eq_x_max + 1):
        __mask_x = nb_permutations(np.arange(1, i + 1), i)  # 第一要素が0にする。
        mask_x = np.zeros((__mask_x.shape[0], __mask_x.shape[1] + 1), dtype="int8")
        mask_x[:, 1:] = __mask_x  # 第一要素が0にする。
        change_x_num = np.array([make_change_x_id(mask_x[k, 1:], i) for k in range(mask_x.shape[0])])
        index = np.argsort(change_x_num)
        dict_mask_x[i] = mask_x[index].copy()
    return dict_mask_x


@njit(parallel=True, error_model="numpy")
def make_before_similar_num_list(num_threads, max_op, random_x, before_equations, progress_proxy):
    int_nan = -100
    dict_mask_x = make_dict_mask_x(max_op)  # only ~max_op,
    # Even if operator 3 is to be calculated from this, the use of four kinds of x is not in the before.
    shape = (before_equations.shape[0], 2, random_x.shape[1])
    similar_num_list = np.full(shape, int_nan, dtype="float64")
    for thread_id in prange(num_threads):
        for i in range(thread_id, before_equations.shape[0], num_threads):
            eq_x_max = np.max(before_equations[i])
            mask_x = dict_mask_x[eq_x_max]
            save_similar_num = nb_calc_RPN(random_x, before_equations[i])
            for k in range(1, mask_x.shape[0]):
                similar_num = nb_calc_RPN(random_x[mask_x[k]], before_equations[i])
                if save_similar_num[0, 0] > similar_num[0, 0]:  # min
                    save_similar_num[0] = similar_num[0]
                if save_similar_num[1, 0] > similar_num[1, 0]:  # min
                    save_similar_num[1] = similar_num[1]
            similar_num_list[i] = save_similar_num
            progress_proxy.update(1)
    similar_num_list = similar_num_list[np.argsort(similar_num_list[:, 0, 0])].copy()
    return similar_num_list


@njit(error_model="numpy")
def loop_count(max_op, n_op1, num_threads):
    base_dict = cache_load(max_op - 1)
    _, dict_max_loop = make_dict_change_x_pattern(max_op)
    last_index = np.zeros(num_threads, dtype="int64")
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
                last_index[thread_id] += 4 * dict_max_loop[np.max(base_eq_arr2[id2]), np.max(base_eq_arr1[id1])]
            else:
                last_index[thread_id] += dict_max_loop[np.max(base_eq_arr2[id2]), np.max(base_eq_arr1[id1])]
    tot_loop, loop_per_threads = int(np.sum(last_index)), int(np.max(last_index))
    return tot_loop, loop_per_threads


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
        return_num += count_True(TF[: mask[i] - 1], 0, 0)  # 0 -> lambda x: x
        TF[mask[i] - 1] = False
    return return_num


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
            case -4:  # a / b  (a > b)
                b, a = stack.pop(), stack.pop()
                txt = "(" + a + "/" + b + ")"
                stack.append(txt)
    return stack[0]


@njit(parallel=True)
def thread_check(num_threads):
    checker = np.zeros(num_threads, dtype="bool")
    for thread_id in prange(num_threads):
        checker[thread_id] = True
    return np.all(checker)


@njit(error_model="numpy")
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
def make_check_change_x(mask, same_arr, TF_mask_x):
    int_nan = -100
    max_same_arr = np.max(same_arr)
    unique = np.unique(same_arr[TF_mask_x])
    all_covered = unique.shape[0] == max_same_arr + 1
    if same_arr[TF_mask_x].shape[0] == unique.shape[0]:
        return True, all_covered, np.empty((1, 0, 2), dtype="int8")
    len_arr = mask.shape[1]
    TF = np.empty(((len_arr - 2) * (len_arr - 1), mask.shape[0]), dtype="bool")
    check_pattern = np.empty(((len_arr - 2) * (len_arr - 1), 2), dtype="int8")
    n = 0
    saved_num = np.empty((max_same_arr + 1), dtype="bool")
    for i in range(1, len_arr):
        for j in range(1, len_arr):
            if i != j:
                check_pattern[n, 0] = i
                check_pattern[n, 1] = j
                saved_num[:] = False
                for t in range(mask.shape[0]):
                    TF[n, t] = mask[t, i] < mask[t, j]
                    if TF[n, t]:
                        saved_num[same_arr[t]] = True
                if np.all(saved_num):
                    n += 1
    len_check_pattern = n
    TF = TF[:len_check_pattern]
    check_pattern = check_pattern[:len_check_pattern]

    return_len = min(len_arr - 2, len_check_pattern)
    use = np.empty((return_len), dtype="int64")
    save_use = np.empty((return_len), dtype="bool")
    return_dict = dict()
    dict_saved_num = dict()
    n = 0

    for i in range(1, return_len + 1):
        for j in range(i):
            use[j] = j
        while True:
            saved_num[:] = False
            for j in range(mask.shape[0]):
                one_TF = True
                for k in range(i):
                    if not TF[use[k], j]:
                        one_TF = False
                        break
                if one_TF:
                    if saved_num[same_arr[j]]:
                        saved_num[same_arr[j]] = False
                        break
                    else:
                        saved_num[same_arr[j]] = True
            if np.all(saved_num):
                saved_num[:] = False
                for j in range(mask.shape[0]):
                    if TF_mask_x[j]:
                        one_TF = True
                        for k in range(i):
                            if not TF[use[k], j]:
                                one_TF = False
                                break
                        if one_TF:
                            saved_num[same_arr[j]] = True
                if np.all(saved_num):
                    return True, all_covered, np.expand_dims(check_pattern[use[:i]], axis=0)
                elif np.any(saved_num):
                    is_unique = True
                    for j in range(n):
                        one_dict_saved_num = dict_saved_num[j]
                        sub_is_unique = False
                        for k in range(saved_num.shape[0]):
                            if one_dict_saved_num[k] != saved_num[k]:
                                sub_is_unique = True
                                break
                        if not sub_is_unique:
                            is_unique = False
                            break
                    if is_unique:
                        save_use[:] = False
                        for j in range(i):
                            if not np.all(TF[use[j]][TF_mask_x]):
                                save_use[j] = True
                        return_dict[n] = check_pattern[use[save_use]].copy()
                        dict_saved_num[n] = saved_num.copy()
                        n += 1
            done_plus = update_combination(use[:i], len_check_pattern)
            if not done_plus:
                break
        if n != 0:
            return_arr = np.full((n, return_len, 2), int_nan, dtype="int8")
            for j in range(n):
                arr = return_dict[j]
                return_arr[j, : arr.shape[0]] = arr
            return True, False, return_arr
    return False, False, np.empty((0, return_len, 2), dtype="int8")


@njit(error_model="numpy")  # ,fastmath=True)
def make_check_change_x_norm(base_mask, base_norm_same_arr, base_TF_mask_x, base_check_pattern):
    int_nan = -100
    TF_mask_x = base_TF_mask_x.copy()
    for i in range(base_mask.shape[0]):
        if base_TF_mask_x[i]:
            for j in range(base_check_pattern.shape[0]):
                if base_mask[i, base_check_pattern[j, 0]] > base_mask[i, base_check_pattern[j, 1]]:
                    TF_mask_x[i] = False
                    break
    mask = base_mask[TF_mask_x].copy()
    max_norm_same_arr = np.max(base_norm_same_arr)
    norm_same_arr_set = np.full(max_norm_same_arr + 1, int_nan, dtype="int64")
    norm_same_arr = np.empty_like(base_norm_same_arr[TF_mask_x])
    n = 0
    for i, j in enumerate(np.arange(len(TF_mask_x))[TF_mask_x]):
        num = base_norm_same_arr[j]
        if norm_same_arr_set[num] == int_nan:
            norm_same_arr_set[num] = n
            norm_same_arr[i] = n
            n += 1
        else:
            norm_same_arr[i] = norm_same_arr_set[num]

    max_norm_same_arr = np.max(norm_same_arr)
    n_unique_norm_same_arr = np.unique(norm_same_arr).shape[0]
    if n_unique_norm_same_arr == np.sum(TF_mask_x):
        return True, np.empty((0, 2), dtype="int8")
    len_arr = mask.shape[1]
    TF = np.empty(((len_arr - 2) * (len_arr - 1), mask.shape[0]), dtype="bool")
    check_pattern = np.empty(((len_arr - 2) * (len_arr - 1), 2), dtype="int8")
    n = 0
    norm_saved_num = np.empty((max_norm_same_arr + 1), dtype="bool")
    for i in range(1, len_arr):
        for j in range(1, len_arr):
            if i != j:
                check_pattern[n, 0] = i
                check_pattern[n, 1] = j
                norm_saved_num[:] = False
                for t in range(mask.shape[0]):
                    TF[n, t] = mask[t, i] < mask[t, j]
                    if TF[n, t]:
                        norm_saved_num[norm_same_arr[t]] = True
                if np.all(norm_saved_num):
                    n += 1
    len_check_pattern = n
    TF = TF[:len_check_pattern]
    check_pattern = check_pattern[:len_check_pattern]

    use = np.zeros((len_check_pattern), dtype="bool")
    tot_TF = np.empty((mask.shape[0]), dtype="bool")
    return_len = min(len_arr - 2, len_check_pattern)
    n = 0
    for i in range(1, return_len + 1):
        divide_num = len_check_pattern - (i - 1)
        for j in range(divide_num**i):
            num = j
            max_setted_num = -1
            use[:] = False
            check = True
            for k in range(i):
                set_num = num % divide_num + k
                num //= divide_num
                if max_setted_num < set_num:
                    use[set_num] = True
                    max_setted_num = set_num
                else:
                    check = False
                    break
            if check:
                can_use = True
                tot_TF[:] = True
                norm_saved_num[:] = False
                for k in range(len_check_pattern):
                    if use[k]:
                        for l in range(mask.shape[0]):
                            if not TF[k, l]:
                                tot_TF[l] = False
                for k in range(mask.shape[0]):
                    if tot_TF[k]:
                        if norm_saved_num[norm_same_arr[k]]:
                            can_use = False
                            break
                        else:
                            norm_saved_num[norm_same_arr[k]] = True
                if can_use:
                    if np.all(norm_saved_num):
                        return True, check_pattern[use]
    return False, np.empty((0, 2), dtype="int8")


@njit(error_model="numpy")
def corrcoef(x_1, x_2):
    mean_1, mean_2 = np.mean(x_1), np.mean(x_2)
    x_len = x_1.shape[0]
    sum_1, sum_2, cov = 0.0, 0.0, 0.0
    for i in range(x_len):
        sum_1 += (x_1[i] - mean_1) ** 2
        sum_2 += (x_2[i] - mean_2) ** 2
        cov += (x_1[i] - mean_1) * (x_2[i] - mean_2)
    return np.abs(cov / np.sqrt(sum_1 * sum_2))


@njit(error_model="numpy")
def make_random_x(max_op, len_x, seed=-100, loop=100000, upper=0.3, lower=10):
    if seed != -100:
        np.random.seed(seed)
    save_random_x = np.random.uniform(upper, lower, (max_op + 2, len_x))
    save_corrcoef = 0
    for i in range(1, max_op + 1):
        for j in range(i + 1, max_op + 2):
            new_corrcoef = corrcoef(save_random_x[i], save_random_x[j])
            if new_corrcoef > save_corrcoef:
                save_corrcoef = new_corrcoef
    for _ in range(loop):
        random_x = np.random.uniform(upper, lower, (max_op + 2, len_x))
        now_corrcoef = 0
        for i in range(1, max_op + 1):
            for j in range(i + 1, max_op + 2):
                new_corrcoef = corrcoef(random_x[i], random_x[j])
                if new_corrcoef > now_corrcoef:
                    now_corrcoef = new_corrcoef
        if save_corrcoef > now_corrcoef:
            save_corrcoef = now_corrcoef
            save_random_x = random_x
    save_random_x[0] = 1
    return save_corrcoef, save_random_x


def str_using_mem():
    using_mem = psutil.Process(os.getpid()).memory_info().rss
    return f"{using_mem /(1024**2):.1f} MB"


def del_concatenate(a, b):
    return_arr = np.concatenate((a, b))
    del a, b
    gc.collect()
    return return_arr


@njit(error_model="numpy")
def update_combination(use, n):
    r = use.shape[0]
    if use[r - 1] != n - 1:
        use[r - 1] += 1
        return True
    else:
        for i in range(r - 2, -1, -1):
            if use[i] != n - (r - i):
                use[i] += 1
                if i != r - 2:
                    for j in range(1, r - i):
                        use[i + j] = use[i] + j
                else:
                    use[i + 1] = use[i] + 1
                return True
        return False


@njit(error_model="numpy")
def update_all_pattern(use, patterns):
    r = use.shape[0]
    if use[r - 1] != patterns[r - 1] - 1:
        use[r - 1] += 1
        return True
    else:
        for i in range(r - 2, -1, -1):
            if use[i] != patterns[i] - 1:
                use[i] += 1
                use[i + 1 :] = 0
                return True
        return False
