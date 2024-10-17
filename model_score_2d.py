#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from utils import jit_cov
from numba import njit

# https://qiita.com/m1t0/items/06f2d07e626d1c4733fd

### LDA_2d
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# LDA(solver="lsqr")


@njit(error_model="numpy", fastmath=True)
def LDA_2d(X, y):
    args = sub_LDA_2d_fit(X, y)
    return sub_LDA_2d_score(X, y, *args)


@njit(error_model="numpy", fastmath=True)
def sub_LDA_2d_fit(X, y):
    pi_T = np.sum(y) + 1e-300
    pi_F = np.sum(~y) + 1e-300
    classT_var_0, classT_var_1, classT_cov = jit_cov(X[:, y], ddof=0)
    classF_var_0, classF_var_1, classF_cov = jit_cov(X[:, ~y], ddof=0)
    var_0 = (pi_T * classT_var_0 + pi_F * classF_var_0) / (pi_T + pi_F)
    var_1 = (pi_T * classT_var_1 + pi_F * classF_var_1) / (pi_T + pi_F)
    cov = (pi_T * classT_cov + pi_F * classF_cov) / (pi_T + pi_F)
    mean_T_0, mean_T_1 = np.mean(X[0, y]), np.mean(X[1, y])
    mean_F_0, mean_F_1 = np.mean(X[0, ~y]), np.mean(X[1, ~y])
    dmean_0, dmean_1 = mean_T_0 - mean_F_0, mean_T_1 - mean_F_1
    c = (mean_F_0**2 - mean_T_0**2) * var_1 / 2 + (mean_F_1**2 - mean_T_1**2) * var_0 / 2
    c += (mean_T_0 * mean_T_1 - mean_F_0 * mean_F_1) * cov
    c -= (var_0 * var_1 - cov**2) * (np.log(pi_F / pi_T))
    return var_0, var_1, cov, dmean_0, dmean_1, c


@njit(error_model="numpy", fastmath=True)
def sub_LDA_2d_score(X, y, var_0, var_1, cov, dmean_0, dmean_1, c):
    a = dmean_0 * var_1 - dmean_1 * cov
    b = dmean_1 * var_0 - dmean_0 * cov
    score = np.sum(((a * X[0] + b * X[1] + c) > 0) == y) / X.shape[1]
    # Kullback-Leibler Divergence , https://sucrose.hatenablog.com/entry/2013/07/20/190146
    kl_d = (
        (dmean_0**2 * var_1 + dmean_1**2 * var_0 - 2 * dmean_0 * dmean_1 * cov) / (var_0 * var_1 - cov**2 + 1e-300) / 2
    )
    return score, kl_d


### QDA_2d
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
# QDA()


@njit(error_model="numpy", fastmath=True)
def QDA_2d(X, y):
    args = sub_QDA_2d_fit(X, y)
    return sub_QDA_2d_score(X, y, *args)


@njit(error_model="numpy", fastmath=True)
def sub_QDA_2d_fit(X, y):
    classT_X = X[:, y]
    classF_X = X[:, ~y]
    pi_T, pi_F = classT_X.shape[1] + 1e-300, classF_X.shape[1] + 1e-300
    classT_var_1, classT_var_2, classT_cov = jit_cov(classT_X)
    det_covT = classT_var_1 * classT_var_2 - classT_cov**2 + 1e-300

    classF_var_1, classF_var_2, classF_cov = jit_cov(classF_X)
    det_covF = classF_var_1 * classF_var_2 - classF_cov**2 + 1e-300

    mean_T = np.array([np.mean(classT_X[0, :]), np.mean(classT_X[1, :])])
    mean_F = np.array([np.mean(classF_X[0, :]), np.mean(classF_X[1, :])])
    return (
        pi_T,
        pi_F,
        classT_var_1,
        classT_var_2,
        classT_cov,
        classF_var_1,
        classF_var_2,
        classF_cov,
        mean_T,
        mean_F,
        det_covT,
        det_covF,
    )


@njit(error_model="numpy", fastmath=True)
def sub_QDA_2d_score(
    X,
    y,
    pi_T,
    pi_F,
    classT_var_1,
    classT_var_2,
    classT_cov,
    classF_var_1,
    classF_var_2,
    classF_cov,
    mean_T,
    mean_F,
    det_covT,
    det_covF,
):
    target_x_1 = (X.T - mean_T).T
    target_x_2 = (X.T - mean_F).T
    value1 = (
        classT_var_2 * target_x_1[0, :] ** 2
        + classT_var_1 * target_x_1[1, :] ** 2
        - 2 * classT_cov * target_x_1[0, :] * target_x_1[1, :]
    )
    value1 /= det_covT
    value2 = (
        classF_var_2 * target_x_2[0, :] ** 2
        + classF_var_1 * target_x_2[1, :] ** 2
        - 2 * classF_cov * target_x_2[0, :] * target_x_2[1, :]
    )
    value2 /= det_covF
    value3 = 2 * np.log(pi_T / pi_F) - np.log(np.abs(det_covT / det_covF) + 1e-300)
    value = -value1 + value2 + value3
    score = np.sum((value > 0) == y) / X.shape[1]

    # Kullback-Leibler Divergence , https://sucrose.hatenablog.com/entry/2013/07/20/190146
    kl_d = np.log(np.abs(det_covF / det_covT + 1e-300)) / 2 - 1
    kl_d += (classF_var_2 * classT_var_1 + classF_var_1 * classT_var_2 - 2 * classT_cov * classF_cov) / det_covF / 2
    dmean = mean_T - mean_F
    kl_d += (
        (dmean[0] ** 2 * classF_var_2 + dmean[1] ** 2 * classF_var_1 - 2 * dmean[0] * dmean[1] * classF_cov)
        / det_covF
        / 2
    )
    return score, kl_d


### DT_2d
# from sklearn.tree import DecisionTreeClassifier as DT
# DT(criterion='entropy',max_depth=2)


from model_score_1d import sub_DT_1d_fit


@njit(error_model="numpy")
def DT_2d(X, y):
    args = sub_DT_2d_fit(X, y)
    return sub_DT_2d_score(X, y, *args)


@njit(error_model="numpy")
def sub_DT_2d_fit(x, y):
    # root node
    border_0, entropy_0, area_predict_0, entropy_0_0, entropy_0_1 = sub_DT_1d_fit(x[0], y)
    border_1, entropy_1, area_predict_1, entropy_1_0, entropy_1_1 = sub_DT_1d_fit(x[1], y)
    if entropy_0 < entropy_1:
        root_node = 0
        root_border = border_0
        root_entropy_0 = entropy_0_0
        root_entropy_1 = entropy_0_1
        root_area_predict = area_predict_0
    else:
        root_node = 1
        root_border = border_1
        root_entropy_0 = entropy_1_0
        root_entropy_1 = entropy_1_1
        root_area_predict = area_predict_1

    area_0 = x[root_node] < root_border
    area_1 = x[root_node] >= root_border
    predict = np.zeros((2, 2), dtype="bool")

    # leaf node  (area_0)
    border_0, entropy_0, area_predict_0, _, _ = sub_DT_1d_fit(x[0, area_0], y[area_0])
    border_1, entropy_1, area_predict_1, _, _ = sub_DT_1d_fit(x[1, area_0], y[area_0])

    if min(entropy_0, entropy_1) < root_entropy_0:
        if entropy_0 < entropy_1:
            leaf_0_node = 0
            leaf_0_border = border_0
            predict[0] = area_predict_0
        else:
            leaf_0_node = 1
            leaf_0_border = border_1
            predict[0] = area_predict_1
    else:
        leaf_0_node = root_node
        leaf_0_border = root_border
        predict[0] = root_area_predict[0]

    # leaf node  (area_1)
    border0, entropy_0, area_predict_0, _, _ = sub_DT_1d_fit(x[0, area_1], y[area_1])
    border1, entropy_1, area_predict_1, _, _ = sub_DT_1d_fit(x[1, area_1], y[area_1])

    if min(entropy_0, entropy_1) < root_entropy_1:
        if entropy_0 < entropy_1:
            leaf_1_node = 0
            leaf_1_border = border0
            predict[1] = area_predict_0
        else:
            leaf_1_node = 1
            leaf_1_border = border1
            predict[1] = area_predict_1
    else:
        leaf_1_node = root_node
        leaf_1_border = root_border
        predict[1] = root_area_predict[1]

    return (
        root_node,
        root_border,
        leaf_0_node,
        leaf_0_border,
        leaf_1_node,
        leaf_1_border,
        predict,
    )


@njit(error_model="numpy")
def sub_DT_2d_score(
    x,
    y,
    root_node,
    root_border,
    leaf_0_node,
    leaf_0_border,
    leaf_1_node,
    leaf_1_border,
    predict,
):
    area_0 = x[root_node] < root_border
    area_0_0 = area_0 & (x[leaf_0_node] < leaf_0_border)
    area_0_1 = area_0 & (x[leaf_0_node] >= leaf_0_border)

    area_1 = x[root_node] >= root_border
    area_1_0 = area_1 & (x[leaf_1_node] < leaf_1_border)
    area_1_1 = area_1 & (x[leaf_1_node] >= leaf_1_border)

    score = np.sum(y[area_0_0] == predict[0, 0]) + np.sum(y[area_0_1] == predict[0, 1])
    score += np.sum(y[area_1_0] == predict[1, 0]) + np.sum(y[area_1_1] == predict[1, 1])
    score /= y.shape[0]
    n_area_0_0 = np.sum(area_0_0) + 1e-300
    n_area_0_0_T = np.sum(y[area_0_0]) + 1e-300
    n_area_0_0_F = n_area_0_0 - n_area_0_0_T + 1e-300
    n_area_0_1 = np.sum(area_0_1) + 1e-300
    n_area_0_1_T = np.sum(y[area_0_1]) + 1e-300
    n_area_0_1_F = n_area_0_1 - n_area_0_1_T + 1e-300
    n_area_1_0 = np.sum(area_1_0) + 1e-300
    n_area_1_0_T = np.sum(y[area_1_0]) + 1e-300
    n_area_1_0_F = n_area_1_0 - n_area_1_0_T + 1e-300
    n_area_1_1 = np.sum(area_1_1) + 1e-300
    n_area_1_1_T = np.sum(y[area_1_1]) + 1e-300
    n_area_1_1_F = n_area_1_1 - n_area_1_1_T + 1e-300
    entropy = (
        n_area_0_0_T * np.log2(n_area_0_0_T / n_area_0_0)
        + n_area_0_0_F * np.log2(n_area_0_0_F / n_area_0_0)
        + n_area_0_1_T * np.log2(n_area_0_1_T / n_area_0_1)
        + n_area_0_1_F * np.log2(n_area_0_1_F / n_area_0_1)
        + n_area_1_0_T * np.log2(n_area_1_0_T / n_area_1_0)
        + n_area_1_0_F * np.log2(n_area_1_0_F / n_area_1_0)
        + n_area_1_1_T * np.log2(n_area_1_1_T / n_area_1_1)
        + n_area_1_1_F * np.log2(n_area_1_1_F / n_area_1_1)
    ) / y.shape[0]

    return score, -entropy


### KNN_2d
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=5)
# from sklearn.model_selection import LeaveOneOut
# LeaveOneOutCV


def make_KNN_2d(k=5, name=None):
    @njit(error_model="numpy")
    def KNN_2d(x, y):
        n_samples = y.shape[0]
        entropy, count = 0, 0
        p_arr = np.array([np.log(1 + np.exp(k - 2 * i)) for i in range(k + 1)])
        for i in range(n_samples):
            d = (x[0] - x[0, i]) ** 2 + (x[1] - x[1, i]) ** 2
            index = np.argpartition(d, kth=k + 1)[: k + 1]
            count_T = int(np.sum(y[index]))
            if i in index:
                count_T -= int(y[i])
            else:
                count_T -= int(y[index[np.argmax(d[index])]])
            count_F = k - count_T
            if y[i]:
                entropy += p_arr[count_T]
                if count_T > count_F:
                    count += 1
            else:  # if not y[i]:
                entropy += p_arr[count_F]
                if count_T < count_F:
                    count += 1
        count /= n_samples
        entropy /= n_samples
        return -entropy, count

    model = KNN_2d
    if name is None:
        model.__name__ = f"KNN_k_{k}_2d"
    else:
        if not isinstance(name, str):
            raise TypeError(f"Expected variable 'name' to be of type str, but got {type(name)}.")
        else:
            model.__name__ = name
    return model


@njit(error_model="numpy")
def sub_KNN_2d_fit(x, y):
    return x, y


def make_sub_KNN_2d_score(k=5, name=None):
    @njit(error_model="numpy")
    def sub_KNN_2d_score(x, y, train_x, train_y):
        n_samples = y.shape[0]
        entropy, count = 0, 0
        p_arr = np.array([np.log(1 + np.exp(k - 2 * i)) for i in range(k + 1)])
        for i in range(n_samples):
            d = (train_x[0] - x[0, i]) ** 2 + (train_x[1] - x[1, i]) ** 2
            index = np.argpartition(d, kth=k)[:k]
            count_T = int(np.sum(train_y[index]))
            count_F = k - count_T
            if y[i]:
                entropy += p_arr[count_T]
                if count_T > count_F:
                    count += 1
            else:  # if not y[i]:
                entropy += p_arr[count_F]
                if count_T < count_F:
                    count += 1
        count /= n_samples
        entropy /= n_samples
        return -entropy, count

    model = sub_KNN_2d_score
    if name is None:
        model.__name__ = f"sub_KNN_k_{k}_2d_score"
    else:
        if not isinstance(name, str):
            raise TypeError(f"Expected variable 'name' to be of type str, but got {type(name)}.")
        else:
            model.__name__ = name
    return model


### Weighted KNN_2d
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=t.shape[0]-1,weights=lambda d: 1/d**2)
# from sklearn.model_selection import LeaveOneOut
# LeaveOneOutCV


@njit(error_model="numpy")
def WKNN_2d(x, y):
    n_samples = y.shape[0]
    w = np.empty((n_samples, n_samples), dtype="float64")
    for i in range(n_samples):
        w[i, i] = 0
        w[i, i + 1 :] = 1 / ((x[0, i + 1 :] - x[0, i]) ** 2 + (x[1, i + 1 :] - x[1, i]) ** 2 + 1e-300)
        w[i + 1 :, i] = w[i, i + 1 :]
    p_T = 1 / (1 + np.exp(1 - 2 * np.sum(w[y], axis=0) / np.sum(w, axis=0)))
    entropy = -(np.sum(np.log(p_T[y])) + np.sum(np.log(1 - p_T[~y]))) / n_samples
    count = (np.sum(p_T[y] > 0.5) + np.sum(p_T[~y] < 0.5)) / n_samples
    return -entropy, count


@njit(error_model="numpy")
def sub_WKNN_2d_fit(x, y):
    return x, y


@njit(error_model="numpy")
def sub_WKNN_2d_score(x, y, train_x, train_y):
    n_samples = y.shape[0]
    d = (np.repeat(train_x[0], n_samples).reshape((train_x.shape[1], n_samples)) - x[0]) ** 2
    d += (np.repeat(train_x[1], n_samples).reshape((train_x.shape[1], n_samples)) - x[1]) ** 2
    w = 1 / (d + 1e-300)
    w /= np.sum(w, axis=0)
    p_T = 1 / (1 + np.exp(1 - 2 * np.sum(w[train_y], axis=0)))
    entropy = -(np.sum(np.log(p_T[y])) + np.sum(np.log(1 - p_T[~y]))) / n_samples
    count = (np.sum(p_T[y] > 0.5) + np.sum(p_T[~y] < 0.5)) / n_samples
    return -entropy, count


### Hull_2d
KNN_2d_for_Hull = make_KNN_2d(k=1)


@njit(error_model="numpy")
def Spearman_coefficient(X):
    index0 = np.argsort(np.argsort(X[0]))
    index1 = np.argsort(np.argsort(X[1]))
    n = index0.shape[0]
    r_R = np.abs(1 - 6 * np.sum((index0 - index1) ** 2) / (n * (n**2 - 1)))
    return r_R


@njit(error_model="numpy")
def checker(X, y):
    X_T = X[:, y].copy()
    X_F = X[:, ~y].copy()
    min_d = np.empty(y.shape[0], dtype="float64")
    for i in range(y.shape[0]):
        if y[i]:
            min_d[i] = np.sqrt(np.min((X_F[0] - X[0, i]) ** 2 + (X_F[1] - X[1, i]) ** 2))
        else:
            min_d[i] = np.sqrt(np.min((X_T[0] - X[0, i]) ** 2 + (X_T[1] - X[1, i]) ** 2))
    var_1, var_2, _ = jit_cov(X)
    sum_min_d = np.mean(np.sort(min_d)[: (4 * y.shape[0]) // 5])
    normalized_sum_min_d = sum_min_d / np.sqrt(var_1 + var_2)
    return normalized_sum_min_d


@njit(error_model="numpy")
def Hull_2d(X, y):
    # Spearman rank correlation coefficient
    r_R = Spearman_coefficient(X)
    if r_R > 0.9:
        return 0, 0
    classT_X, classF_X = X[:, y], X[:, ~y]
    index_x_max = np.argmax(classT_X[0])
    index_x_min = np.argmin(classT_X[0])
    arange = np.arange(classT_X.shape[1])
    classT_X_mask = np.ones(classT_X.shape[1], dtype="bool")
    not_is_in = np.ones(classF_X.shape[1], dtype="bool")
    classT_X_mask[index_x_max] = False
    classT_X_mask[index_x_min] = False
    copy_classT_X_mask = classT_X_mask.copy()
    sub_Hull_2d(
        classT_X,
        classF_X,
        not_is_in,
        arange,
        index_x_max,
        index_x_min,
        copy_classT_X_mask,
    )
    sub_Hull_2d(classT_X, classF_X, not_is_in, arange, index_x_min, index_x_max, classT_X_mask)
    ans = np.sum(~not_is_in)
    index_x_max = np.argmax(classF_X[0])
    index_x_min = np.argmin(classF_X[0])
    arange = np.arange(classF_X.shape[1])
    classF_X_mask = np.ones(classF_X.shape[1], dtype="bool")
    not_is_in = np.ones(classT_X.shape[1], dtype="bool")
    classF_X_mask[index_x_max] = False
    classF_X_mask[index_x_min] = False
    copy_classF_X_mask = classF_X_mask.copy()
    sub_Hull_2d(
        classF_X,
        classT_X,
        not_is_in,
        arange,
        index_x_max,
        index_x_min,
        copy_classF_X_mask,
    )
    sub_Hull_2d(classF_X, classT_X, not_is_in, arange, index_x_min, index_x_max, classF_X_mask)
    ans += np.sum(~not_is_in)
    score = 1 - (ans / (y.shape[0]))
    # KNN check
    if score > 0.5:
        KNN_score, _ = KNN_2d_for_Hull(X, y)
        if KNN_score + 0.2 < score:
            return 0, 0
    return score, 0


@njit(error_model="numpy")
def sub_Hull_2d(base_x, other_x, not_is_in, arange, base_index1, base_index2, base_x_mask):
    if np.any(not_is_in):
        copy_base_x_mask = base_x_mask.copy()
        base_vec = base_x[:, base_index1] - base_x[:, base_index2]
        bec_xy = base_x[:, arange[copy_base_x_mask]] - np.expand_dims(base_x[:, base_index2], axis=1)
        cross = base_vec[0] * bec_xy[1] - base_vec[1] * bec_xy[0]
        if np.any(cross > 0):
            next_point = np.argmax(cross)
            next_index = arange[copy_base_x_mask][next_point]
            copy_base_x_mask[arange[copy_base_x_mask][cross <= 0]] = False
            copy_base_x_mask[next_index] = False
            use_other_x = other_x[:, not_is_in] - np.expand_dims(base_x[:, base_index2], axis=1)
            num_is_in = np.empty((2, np.sum(not_is_in)), dtype="float")
            num_is_in[0] = base_vec[1] * use_other_x[0] - base_vec[0] * use_other_x[1]
            num_is_in[1] = -bec_xy[1, next_point] * use_other_x[0] + bec_xy[0, next_point] * use_other_x[1]
            num_is_in /= bec_xy[0, next_point] * base_vec[1] - base_vec[0] * bec_xy[1, next_point]
            not_is_in[not_is_in] = ((num_is_in[0] + num_is_in[1]) > 1) | (0 > num_is_in[0]) | (0 > num_is_in[1])
            sub_Hull_2d(
                base_x,
                other_x,
                not_is_in,
                arange,
                next_index,
                base_index2,
                copy_base_x_mask,
            )
            sub_Hull_2d(
                base_x,
                other_x,
                not_is_in,
                arange,
                base_index1,
                next_index,
                copy_base_x_mask,
            )


### make CV_model
def CV_model_2d(sub_func_fit, sub_func_score, k=5, name=None):
    if k <= 1:
        raise

    @njit(error_model="numpy")
    def CrossValidation_2d(X, y):
        index = np.arange(y.shape[0])
        np.random.shuffle(index)
        TF = np.ones(y.shape[0], dtype="bool")
        sum_score1, sum_score2 = 0, 0
        for i in range(k):
            TF[index[i::k]] = False
            args = sub_func_fit(X[:, TF], y[TF])
            score1, score2 = sub_func_score(X[:, ~TF], y[~TF], *args)
            sum_score1 += score1
            sum_score2 += score2
            TF[index[i::k]] = True
        return sum_score1 / k, sum_score2 / k

    model = CrossValidation_2d
    if name is None:
        base_model_name = sub_func_fit.__name__
        base_model_name = base_model_name.replace("sub_", "").replace("_2d_fit", "")
        model.__name__ = f"CV_k_{k}_{base_model_name}_2d"
    else:
        if not isinstance(name, str):
            raise TypeError(f"Expected variable 'name' to be of type str, but got {type(name)}.")
        else:
            model.__name__ = name
    return model
