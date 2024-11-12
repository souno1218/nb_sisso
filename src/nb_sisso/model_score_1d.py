#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numba import njit

# If error_model is available for jitclass, replace all of them with class


### debug
@njit(error_model="numpy")  # ,fastmath=True)
def debug_1d(x, y):
    args = sub_debug_1d_fit(x, y)
    return sub_debug_1d_score(x, y, *args)


@njit(error_model="numpy")  # ,fastmath=True)
def sub_debug_1d_fit(x, y):
    mu = np.mean(x)
    sigma = np.sqrt(np.mean((x - mu) ** 2)) + 1e-300
    return mu, sigma


@njit(error_model="numpy")  # ,fastmath=True)
def sub_debug_1d_score(x, y, mu, sigma):
    return np.abs((x[0] - mu) / sigma), 0


### LDA  ,  https://qiita.com/m1t0/items/06f2d07e626d1c4733fd
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# LDA(solver="lsqr")


@njit(error_model="numpy")  # ,fastmath=True)
def LDA_1d(x, y):
    args = sub_LDA_1d_fit(x, y)
    return sub_LDA_1d_score(x, y, *args)


@njit(error_model="numpy")
def sub_LDA_1d_fit(x, y):
    pi_T = np.sum(y)
    pi_F = np.sum(~y)
    mu_T = np.mean(x[y])
    mu_F = np.mean(x[~y])
    sigma = (np.sum((x[y] - mu_T) ** 2) + np.sum((x[~y] - mu_F) ** 2)) / y.shape[0] + 1e-300
    return pi_T, pi_F, mu_T, mu_F, sigma


@njit(error_model="numpy")
def sub_LDA_1d_score(x, y, pi_T, pi_F, mu_T, mu_F, sigma):
    f = (mu_T - mu_F) * x - (mu_T - mu_F) * (mu_T + mu_F) / 2 + sigma * np.log(pi_T / pi_F)
    score = np.sum((f >= 0) == y) / y.shape[0]
    # Kullback-Leibler Divergence , https://sucrose.hatenablog.com/entry/2013/07/20/190146
    kl_d = ((mu_T - mu_F) ** 2) / 2 / sigma
    return score, kl_d


### QDA  ,  https://qiita.com/m1t0/items/06f2d07e626d1c4733fd
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
# QDA()


@njit(error_model="numpy")
def QDA_1d(x, y):
    args = sub_QDA_1d_fit(x, y)
    return sub_QDA_1d_score(x, y, *args)


@njit(error_model="numpy")
def sub_QDA_1d_fit(x, y):
    mu_T, mu_F = np.mean(x[y]) + 1e-300, np.mean(x[~y]) + 1e-300
    pi_T, pi_F = np.sum(y), np.sum(~y)
    sigma_T = np.sum((x[y] - mu_T) ** 2) / (pi_T - 1) + 1e-300
    sigma_F = np.sum((x[~y] - mu_F) ** 2) / (pi_F - 1) + 1e-300
    value2 = 2 * np.log(pi_T / pi_F) - np.log(sigma_T / sigma_F)
    return mu_T, mu_F, sigma_T, sigma_F, value2


@njit(error_model="numpy")
def sub_QDA_1d_score(x, y, mu_T, mu_F, sigma_T, sigma_F, value2):
    value = -((x - mu_T) ** 2 / sigma_T) + (((x - mu_F)) ** 2 / sigma_F) + value2
    score = np.sum((value > 0) == y) / y.shape[0]
    # Kullback-Leibler Divergence , https://sucrose.hatenablog.com/entry/2013/07/20/190146
    kl_d = (sigma_T + (mu_T - mu_F) ** 2) / 2 / sigma_F - 0.5
    kl_d += np.log(sigma_F / sigma_T) / 2
    return score, kl_d


### DT
# from sklearn.tree import DecisionTreeClassifier as DT
# DT(criterion='entropy',max_depth=2)


@njit(error_model="numpy")
def DT_1d(x, y):
    border, tot_entropy, area_predict, _, _ = sub_DT_1d_fit(x, y)
    n_tot = x.shape[0] + 1e-300
    score = (np.sum(y[x < border] == area_predict[0]) + np.sum(y[x >= border] == area_predict[1])) / n_tot
    return score, -tot_entropy


@njit(error_model="numpy")
def sub_DT_1d_fit(x, y):
    n_T = np.sum(y)
    n_F = np.sum(~y)
    n_samples = y.shape[0]
    sort_index = np.argsort(x)
    TF = np.empty(n_samples, dtype="bool")
    TF[0] = True
    TF[1:] = x[sort_index[1:]] != x[sort_index[:-1]]
    sorted_y_T = y[sort_index].astype("int64")
    sorted_y_F = (~y)[sort_index].astype("int64")
    for i in np.arange(n_samples)[~TF][::-1]:
        sorted_y_T[i - 1] += sorted_y_T[i]
        sorted_y_F[i - 1] += sorted_y_F[i]
    cumsum_T = np.cumsum(sorted_y_T[TF]) + 1e-300
    cumsum_F = np.cumsum(sorted_y_F[TF]) + 1e-300
    reverse_T = n_T - cumsum_T + 2e-300
    reverse_F = n_F - cumsum_F + 2e-300
    sum_cumsum = cumsum_T + cumsum_F - 1e-300
    reverse_sum_cumsum = reverse_T + reverse_F - 1e-300
    entropy0 = -(
        cumsum_T * (np.log2(cumsum_T / sum_cumsum)) / sum_cumsum
        + cumsum_F * (np.log2(cumsum_F / sum_cumsum)) / sum_cumsum
    )
    entropy1 = -(
        reverse_T * (np.log2(reverse_T / reverse_sum_cumsum)) / reverse_sum_cumsum
        + reverse_F * (np.log2(reverse_F / reverse_sum_cumsum)) / reverse_sum_cumsum
    )
    tot_entropy = (entropy0 * sum_cumsum + entropy1 * reverse_sum_cumsum) / n_samples

    index = np.argmin(tot_entropy)
    border = (x[sort_index[index]] + x[sort_index[min(np.sum(TF) - 1, index + 1)]]) / 2
    first_area = x < border
    area_predict = np.array(
        [
            np.sum(y[first_area]) > np.sum(~y[first_area]),
            np.sum(y[~first_area]) > np.sum(~y[~first_area]),
        ]
    )
    return border, tot_entropy[index], area_predict, entropy0[index], entropy1[index]


@njit(error_model="numpy")
def sub_DT_1d_score(x, y, border, tot_entropy, area_predict, entropy0, entropy1):
    n_tot = x.shape[0] + 1e-300
    first_area = x < border
    n_first_area = np.sum(first_area) + 1e-300
    second_area = ~first_area
    n_second_area = np.sum(second_area) + 1e-300
    n_first_area_T = np.sum(y[first_area]) + 1e-300
    n_second_area_T = np.sum(y[second_area]) + 1e-300
    score = (np.sum(y[first_area] == area_predict[0]) + np.sum(y[~first_area] == area_predict[1])) / n_tot

    entropy = (
        -n_first_area_T * np.log2(n_first_area_T / n_first_area)
        - (n_first_area - n_first_area_T) * np.log2((n_first_area - n_first_area_T + 1e-300) / n_first_area)
        - n_second_area_T * np.log2(n_second_area_T / n_second_area)
        - (n_second_area - n_second_area_T) * np.log2((n_second_area - n_second_area_T + 1e-300) / n_second_area)
    ) / n_tot
    return score, -entropy


### KNN
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=5)
# from sklearn.model_selection import LeaveOneOut
# LeaveOneOutCV


def make_KNN_1d(k=5, name=None):
    @njit(error_model="numpy")
    def KNN_1d(x, y):
        n_samples = y.shape[0]
        sort_index = np.argsort(x)
        sorted_x = x[sort_index].copy()
        sorted_y = y[sort_index].copy()
        entropy, count = 0, 0
        p_arr = np.array([np.log(1 + np.exp(k - 2 * i)) for i in range(k + 1)])
        for i in range(n_samples):
            d = np.abs(sorted_x[max(i - k, 0) : min(i + k + 1, n_samples)] - sorted_x[i])
            index = max(i - k, 0) + np.argsort(d)[: k + 1]
            index = index[index != i][:k]
            count_T = int(np.sum(sorted_y[index]))
            count_F = k - count_T
            if sorted_y[i]:
                entropy += p_arr[count_T]
                if count_T > count_F:
                    count += 1
            else:  # if not sorted_y[i]:
                entropy += p_arr[count_F]
                if count_T < count_F:
                    count += 1
        count /= n_samples
        entropy /= n_samples
        return count, -entropy

    model = KNN_1d
    if name is None:
        model.__name__ = f"KNN_k_{k}_1d"
    else:
        if not isinstance(name, str):
            raise TypeError(f"Expected variable 'name' to be of type str, but got {type(name)}.")
        else:
            model.__name__ = name
    return model


@njit(error_model="numpy")
def sub_KNN_1d_fit(x, y):
    return x, y


def make_sub_KNN_1d_score(k=5, name=None):
    @njit(error_model="numpy")
    def sub_KNN_1d_score(x, y, train_x, train_y):
        n_samples = y.shape[0]
        entropy, count = 0, 0
        p_arr = np.array([np.log(1 + np.exp(k - 2 * i)) for i in range(k + 1)])
        for i in range(n_samples):
            d = np.abs(train_x - x[i])
            index = np.argpartition(d, kth=k)[:k]  # ??
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
        return count, -entropy

    model = sub_KNN_1d_score
    if name is None:
        model.__name__ = f"sub_KNN_k_{k}_1d_score"
    else:
        if not isinstance(name, str):
            raise TypeError(f"Expected variable 'name' to be of type str, but got {type(name)}.")
        else:
            model.__name__ = name
    return model


### Weighted KNN_1d
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=t.shape[0]-1,weights=lambda d: 1/d**2)
# from sklearn.model_selection import LeaveOneOut
# LeaveOneOutCV


def make_WNN_1d(p=2, name=None):
    @njit(error_model="numpy")
    def WNN_1d_even(x, y):
        n_samples = y.shape[0]
        w = np.empty((n_samples, n_samples), dtype="float64")
        for i in range(n_samples):
            w[i, i] = 0
            w[i, i + 1 :] = 1 / ((x[i + 1 :] - x[i]) ** p + 1e-300)
            w[i + 1 :, i] = w[i, i + 1 :]
        p_T = 1 / (1 + np.exp(1 - 2 * np.sum(w[y], axis=0) / np.sum(w, axis=0)))
        entropy = -(np.sum(np.log(p_T[y])) + np.sum(np.log(1 - p_T[~y]))) / n_samples
        count = (np.sum(p_T[y] > 0.5) + np.sum(p_T[~y] < 0.5)) / n_samples
        return count, -entropy

    @njit(error_model="numpy")
    def WNN_1d_odd(x, y):
        n_samples = y.shape[0]
        w = np.empty((n_samples, n_samples), dtype="float64")
        for i in range(n_samples):
            w[i, i] = 0
            w[i, i + 1 :] = 1 / (np.abs(x[i + 1 :] - x[i]) ** p + 1e-300)
            w[i + 1 :, i] = w[i, i + 1 :]
        p_T = 1 / (1 + np.exp(1 - 2 * np.sum(w[y], axis=0) / np.sum(w, axis=0)))
        entropy = -(np.sum(np.log(p_T[y])) + np.sum(np.log(1 - p_T[~y]))) / n_samples
        count = (np.sum(p_T[y] > 0.5) + np.sum(p_T[~y] < 0.5)) / n_samples
        return count, -entropy

    if p % 2 == 0:
        model = WNN_1d_even
    else:
        model = WNN_1d_odd
    if name is None:
        model.__name__ = f"WNN_p_{p}_1d"
    else:
        if not isinstance(name, str):
            raise TypeError(f"Expected variable 'name' to be of type str, but got {type(name)}.")
        else:
            model.__name__ = name
    return model


@njit(error_model="numpy")
def sub_WNN_1d_fit(x, y):
    return x, y


def make_sub_WNN_1d_score(p=2, name=None):
    @njit(error_model="numpy")
    def sub_WNN_1d_score_even(x, y, train_x, train_y):
        n_samples = y.shape[0]
        w = 1 / ((np.repeat(train_x, n_samples).reshape((train_x.shape[0], n_samples)) - x) ** p + 1e-300)
        w /= np.sum(w, axis=0)
        p_T = 1 / (1 + np.exp(1 - 2 * np.sum(w[train_y], axis=0)))
        entropy = -(np.sum(np.log(p_T[y])) + np.sum(np.log(1 - p_T[~y]))) / n_samples
        count = (np.sum(p_T[y] > 0.5) + np.sum(p_T[~y] < 0.5)) / n_samples
        return count, -entropy

    @njit(error_model="numpy")
    def sub_WNN_1d_score_odd(x, y, train_x, train_y):
        n_samples = y.shape[0]
        w = 1 / (np.abs(np.repeat(train_x, n_samples).reshape((train_x.shape[0], n_samples)) - x) ** p + 1e-300)
        w /= np.sum(w, axis=0)
        p_T = 1 / (1 + np.exp(1 - 2 * np.sum(w[train_y], axis=0)))
        entropy = -(np.sum(np.log(p_T[y])) + np.sum(np.log(1 - p_T[~y]))) / n_samples
        count = (np.sum(p_T[y] > 0.5) + np.sum(p_T[~y] < 0.5)) / n_samples
        return count, -entropy

    if p % 2 == 0:
        model = sub_WNN_1d_score_even
    else:
        model = sub_WNN_1d_score_odd

    if name is None:
        model.__name__ = f"sub_WNN_p_{p}_1d_score"
    else:
        if not isinstance(name, str):
            raise TypeError(f"Expected variable 'name' to be of type str, but got {type(name)}.")
        else:
            model.__name__ = name
    return model


### Weighted gauss KNN_1d
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier(n_neighbors=t.shape[0]-1,weights=lambda d: 1/d**2)
# from sklearn.model_selection import LeaveOneOut
# LeaveOneOutCV


@njit(error_model="numpy")  # ,fastmath=True)
def WGNN_1d(x, y):
    n_samples = y.shape[0]
    d_2 = np.empty((n_samples, n_samples), dtype="float64")
    for i in range(n_samples):
        d_2[i, i] = 0
        d_2[i, i + 1 :] = (x[i + 1 :] - x[i]) ** 2
        d_2[i + 1 :, i] = d_2[i, i + 1 :]
    w = np.exp(-(n_samples / (2 * np.sum(d_2, axis=0))) * d_2)
    p_T = np.sum(w[y], axis=0) / np.sum(w, axis=0)
    entropy = -(np.sum(np.log(p_T[y])) + np.sum(np.log(1 - p_T[~y]))) / n_samples
    count = (np.sum(p_T[y] > 0.5) + np.sum(p_T[~y] < 0.5)) / n_samples
    return count, -entropy


### Hull
@njit(error_model="numpy")  # ,fastmath=True)
def Hull_1d(x, y):
    args = sub_Hull_1d_fit(x, y)
    return sub_Hull_1d_score(x, y, *args)


@njit(error_model="numpy")  # ,fastmath=True)
def sub_Hull_1d_fit(x, y):
    min_T, max_T = np.min(x[y]), np.max(x[y])
    min_F, max_F = np.min(x[~y]), np.max(x[~y])
    return min_T, max_T, min_F, max_F


@njit(error_model="numpy")  # ,fastmath=True)
def sub_Hull_1d_score(x, y, min_T, max_T, min_F, max_F):
    TF = np.empty(x.shape[0], dtype="bool")
    TF[y] = min_F > x[y]
    TF[y] |= x[y] > max_F
    TF[~y] = min_T > x[~y]
    TF[~y] |= x[~y] > max_T
    S_overlap = min(max_T, max_F) - max(min_T, min_F)
    S = S_overlap / min((max_T - min_T), (max_F - min_F))
    return np.mean(TF), -S


### make CV_model
def CV_model_1d(sub_func_fit, sub_func_score, k=5, name=None):
    if k <= 1:
        raise

    @njit(error_model="numpy")
    def CrossValidation_1d(x, y):
        index = np.arange(y.shape[0])
        np.random.shuffle(index)
        TF = np.ones(y.shape[0], dtype="bool")
        sum_score1, sum_score2 = 0, 0
        for i in range(k):
            TF[index[i::k]] = False
            args = sub_func_fit(x[TF], y[TF])
            score1, score2 = sub_func_score(x[~TF], y[~TF], *args)
            sum_score1 += score1
            sum_score2 += score2
            TF[index[i::k]] = True
        return sum_score1 / k, sum_score2 / k

    model = CrossValidation_1d
    if name is None:
        base_model_name = sub_func_fit.__name__
        base_model_name = base_model_name.replace("sub_", "").replace("_1d_fit", "")
        model.__name__ = f"CV_k_{k}_{base_model_name}_1d"
    else:
        if not isinstance(name, str):
            raise TypeError(f"Expected variable 'name' to be of type str, but got {type(name)}.")
        else:
            model.__name__ = name
    return model
