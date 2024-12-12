import numpy as np
from numba import njit, prange, objmode


@njit(error_model="numpy")
def cache_load(max_ops):
    unique_eq_dict = dict()
    for i in range(max_ops + 1):
        with objmode(data="int8[:,:]"):
            data = np.load(f"operator_{i}.npy")
        unique_eq_dict[i] = data
    return unique_eq_dict


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
    return_equation = base_back_equation.copy()
    for i in range(1, np.max(base_back_equation) + 1):
        return_equation[base_back_equation == i] = change_x_pattern[i - 1]
    return return_equation


@njit(error_model="numpy")
def nb_calc_RPN(x, equation):
    int_nan = -100
    stack = np.full((np.sum(equation >= 0), x.shape[1]), int_nan, dtype="float64")
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
                case int_nan:
                    break
    return_stack = np.empty((2, x.shape[1]), dtype="float64")
    if np.any(np.isinf(stack[0])):
        return_stack[:] = int_nan
        return return_stack
    elif is_all_zero(stack[0]):
        return_stack[0] = int_nan
        return_stack[1] = 0
        return return_stack
    elif is_all_const(stack[0]):
        return_stack[0] = int_nan
        return_stack[1] = np.sign(stack[0, 0]) * stack[0]
        return return_stack
    else:
        _mean = np.mean(stack[0])
        return_stack[0] = np.sign(stack[0, 0] - _mean) * (stack[0] - _mean) / np.sqrt(np.mean((stack[0] - _mean) ** 2))
        return_stack[1] = np.sign(stack[0, 0]) * stack[0]
        return return_stack


@njit(error_model="numpy")
def isclose(a, b, rtol=1e-05, atol=1e-08):
    return np.abs(a - b) <= atol + rtol * np.abs(b)


@njit(error_model="numpy")
def is_all_zero(num, atol=1e-06):
    return np.all(isclose(num, 0, rtol=0, atol=atol))


@njit(error_model="numpy")
def is_all_const(num, rtol=1e-06, atol=0):
    return (np.max(num) - np.min(num)) <= atol + rtol * np.abs(np.mean(num))


@njit(error_model="numpy")
def isclose_arr(arr1, arr2, rtol=1e-06, atol=0):
    # if np.all(isclose(arr1, arr2, rtol=rtol, atol=atol)):
    #    return True
    if isclose(arr1[0], arr2[0], rtol=rtol, atol=atol):
        if isclose(arr1[1], arr2[1], rtol=rtol, atol=atol):
            if isclose(arr1[2], arr2[2], rtol=rtol, atol=atol):
                if np.all(isclose(arr1[3:], arr2[3:], rtol=rtol, atol=atol)):
                    return True
    return False


@njit(error_model="numpy")
def arg_close_arr(arr, mat, rtol=1e-06, atol=0):
    int_nan = -100
    for i in range(mat.shape[0]):
        if isclose_arr(arr, mat[i], rtol=rtol, atol=atol):
            return i
    return int_nan


@njit(error_model="numpy")
def indexes_close_arr(num, arr, rtol=1e-06, atol=0):
    return np.arange(arr.shape[0])[isclose(num, arr, rtol=rtol, atol=atol)]


@njit(error_model="numpy")
def find_min_x_max(equation, random_x, random_for_find_min_x_max, rtol=1e-06, atol=0):
    # a*(b-b) -> drop
    if np.sum(equation > 0) == 0:
        return False
    similar_num = nb_calc_RPN(random_x, equation)[0]
    max_x = np.max(equation)
    counter = np.zeros(max_x + 1, dtype="int8")
    for i in range(1, max_x + 1):
        counter[i] = np.sum(equation == i)
    change_num = np.arange(max_x + 1)[counter >= 2]  # (b-b) -> count = 2, counter >= 2
    changed_random = random_x.copy()
    for i in change_num:
        changed_random[i] = random_for_find_min_x_max
        changed_similar_num = nb_calc_RPN(changed_random, equation)[0]
        if isclose(similar_num[0], changed_similar_num[0], rtol=rtol, atol=atol):
            if np.all(isclose(similar_num, changed_similar_num, rtol=rtol, atol=atol)):
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
    shape = (2, before_equations.shape[0], random_x.shape[1])
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
            similar_num_list[:, i] = save_similar_num
            progress_proxy.update(1)
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


@njit(parallel=True)
def thread_check(num_threads):
    checker = np.zeros(num_threads, dtype="bool")
    for thread_id in prange(num_threads):
        checker[thread_id] = True
    return np.all(checker)
