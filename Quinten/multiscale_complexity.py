import numpy as np

def split_single(a, n_subs):
    l = n_subs
    # means = []
    m, n = a.shape
    b = np.zeros_like(a)
    for i in range(l):
        for j in range(l):
            i0 = i * int(m/l)
            i1 = (i + 1) * int(m/l)
            j0 = j * int(n/l)
            j1 = (j + 1) * int(n/l)
            sub_a = a[i0:i1, j0:j1]
            b[i0:i1, j0:j1] = np.mean(sub_a)

    return b

def create_T(a, n_subs):
    x = np.zeros((len(n_subs), a.shape[0], a.shape[1]))
    for i, n_sub in enumerate(n_subs[::-1]):
        a = split_single(a, n_sub)
        x[i] = a.copy()
    return x

def calc_C_O(T):
    C, O = [], []
    for k in range(1, len(T)):
        O_k = 1/2 * (T[k] - T[k-1]) * (T[k] - T[k-1])
        O.append(O_k)

        C_k = np.abs(np.sum(O_k[k-1]))
        C.append(C_k)

    return np.sum(C), O


def calc_complexity(array):
    n_subs = [2, 4, 8, 16, 32, 64, 128, 256]
    T = create_T(array, n_subs)
    C, O = calc_C_O(T)

    # Florian plotted C
    return C
