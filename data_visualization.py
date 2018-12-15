import numpy as np
from sklearn import datasets
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt


def dist(o1, o2):
    return abs(o1[0] - o2[0])


def static(data_set):
    D = np.zeros(shape=(1, 1))
    L = [data_set[0]]

    for ok in data_set[1:]:
        d = [dist(ok, op) for op in L]

        lamb = [dist(ok, L[0])]
        for p in range(1, len(L) - 1):
            lamb.append(dist(ok, L[p - 1]) + dist(ok, L[p]) - dist(L[p - 1], L[p]))
        if len(L) > 1:
            lamb.append(dist(ok, L[-1]))

        p_opt = np.argmin(lamb)
        L.insert(p_opt, ok)

        D = np.insert(D, p_opt, d, 0)
        d.insert(p_opt, 0)
        D = np.insert(D, p_opt, d, 1)

    return D


def dynamic(data_set):
    D_s_max = None
    s_max = 50
    D = np.zeros(shape=(1, 1))
    L = [(data_set[0], 0)]

    for k, ok in enumerate(data_set[1:], 1):

        if k >= s_max:
            o_to_delete, _ = min(enumerate(L), key=lambda x: x[1][1])
            del L[o_to_delete]

            D = np.delete(D, o_to_delete, axis=0)
            D = np.delete(D, o_to_delete, axis=1)

            if k == s_max:
                D_s_max = D.copy()

        d = [dist(ok, op[0]) for op in L]

        lamb = [dist(ok, L[0][0])]
        for p in range(1, len(L) - 1):
            lamb.append(dist(ok, L[p - 1][0]) + dist(ok, L[p][0]) - dist(L[p - 1][0], L[p][0]))
        if len(L) > 1:
            lamb.append(dist(ok, L[-1][0]))

        p_opt = np.argmin(lamb)
        L.insert(p_opt, (ok, k))

        D = np.insert(D, p_opt, d, 0)
        d.insert(p_opt, 0)
        D = np.insert(D, p_opt, d, 1)

    return D_s_max, D


def prepare_origin_order():
    result = np.zeros(shape=(150, 150))

    for i1, i2 in combinations_with_replacement(range(150), 2):
        distance = dist(iris['data'][i1], iris['data'][i2])
        result[i1][i2] = distance
        result[i2][i1] = distance

    f = plt.figure(1, figsize=(8, 6))
    plt.pcolor(result, figure=f)


def prepare_static_reordering():
    result = static(iris['data'])

    f = plt.figure(2, figsize=(8, 6))
    plt.pcolor(result, figure=f)


def prepare_dynamic_reordering():
    result_50, result_150 = dynamic(iris['data'])

    f = plt.figure(3, figsize=(8, 6))
    plt.pcolor(result_50, figure=f)

    f = plt.figure(4, figsize=(8, 6))
    plt.pcolor(result_150, figure=f)


if __name__ == '__main__':
    iris = datasets.load_iris()

    prepare_origin_order()
    prepare_static_reordering()
    prepare_dynamic_reordering()

    plt.show()
