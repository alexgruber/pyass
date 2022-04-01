
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def plot_KF_behaviour():

    fname = '/Users/u0116961/Documents/work/DA/KF_behaviour_H_diag_nonzero2.png'

    # 5 layers
    n = 5

    # range of observation operator values
    Hs = np.linspace(0.000, 1, 1000)

    res11 = np.zeros((len(Hs), 4))
    res12 = np.zeros((len(Hs), 4))
    res21 = np.zeros((len(Hs), 4))
    res22 = np.zeros((len(Hs), 4))

    for i, Hik in enumerate(Hs):
        x = np.ones(n).reshape(-1, 1)

        y = np.ones(n).reshape(-1, 1) * 2

        # H diagonals either zero or identical to diagnoals
        # H = np.diag(np.ones(n),0) * Hik
        H = np.ones((n, n)) * Hik

        R1 = np.diag(np.ones(n), 0)
        P1 = np.diag(np.ones(n), 0)

        R2 = np.ones((n, n)) * (1 - 1e-2)
        R2[np.arange(n).astype('int'), np.arange(n).astype('int')] = 1

        P2 = np.ones((n, n)) * (1 - 1e-2)
        P2[np.arange(n).astype('int'), np.arange(n).astype('int')] = 1

        K11 = np.dot(np.dot(P1, H), np.linalg.inv(np.dot(np.dot(H, P1), H) + R1))
        K12 = np.dot(np.dot(P1, H), np.linalg.inv(np.dot(np.dot(H, P1), H) + R2))
        K21 = np.dot(np.dot(P2, H), np.linalg.inv(np.dot(np.dot(H, P2), H) + R1))
        K22 = np.dot(np.dot(P2, H), np.linalg.inv(np.dot(np.dot(H, P2), H) + R2))

        P11p = np.dot((np.diag(np.ones(n), 0) - np.dot(K11, H)), P1)
        P12p = np.dot((np.diag(np.ones(n), 0) - np.dot(K12, H)), P1)
        P21p = np.dot((np.diag(np.ones(n), 0) - np.dot(K21, H)), P2)
        P22p = np.dot((np.diag(np.ones(n), 0) - np.dot(K22, H)), P2)

        xa11 = x + np.dot(K11, (y - np.dot(H, x)))
        xa12 = x + np.dot(K12, (y - np.dot(H, x)))
        xa21 = x + np.dot(K21, (y - np.dot(H, x)))
        xa22 = x + np.dot(K22, (y - np.dot(H, x)))

        res11[i, 0] = K11.sum(axis=0)[0]
        res11[i, 1] = xa11[0]
        res11[i, 2] = P11p[0, 0]
        res11[i, 3] = P11p[0, 1]

        res12[i, 0] = K12.sum(axis=0)[0]
        res12[i, 1] = xa12[0]
        res12[i, 2] = P12p[0, 0]
        res12[i, 3] = P12p[0, 1]

        res21[i, 0] = K21.sum(axis=0)[0]
        res21[i, 1] = xa21[0]
        res21[i, 2] = P21p[0, 0]
        res21[i, 3] = P21p[0, 1]

        res22[i, 0] = K22.sum(axis=0)[0]
        res22[i, 1] = xa22[0]
        res22[i, 2] = P22p[0, 0]
        res22[i, 3] = P22p[0, 1]

    ylim1 = (-0.5, 3.0)
    ylim2 = (-0.5, 3.0)
    cols = ['sum(K$_{1i}$)', 'x$^+$', 'P$^+_{ii}$', 'P$^+_{ik}$']

    f = plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.title('P$_{ik}$ = 0, R$_{ik}$ = 0')
    pd.DataFrame(res11, index=Hs, columns=cols).plot(ax=plt.gca())
    plt.axvline(1 / n, linestyle='--', color='k', linewidth=0.5)
    plt.axhline(0, linestyle='--', color='k', linewidth=0.5)
    plt.axhline(1, linestyle='--', color='k', linewidth=0.5)
    plt.xlabel('$H_{ik}$')
    plt.ylim(ylim1)

    plt.subplot(2, 2, 2)
    plt.title('P$_{ik}$ = 0, R$_{ik}$ = 1')
    pd.DataFrame(res12, index=Hs, columns=cols).plot(ax=plt.gca())
    plt.axvline(1 / n, linestyle='--', color='k', linewidth=0.5)
    plt.axhline(0, linestyle='--', color='k', linewidth=0.5)
    plt.axhline(1, linestyle='--', color='k', linewidth=0.5)
    plt.xlabel('$H_{ik}$')
    plt.ylim(ylim1)

    plt.subplot(2, 2, 3)
    plt.title('P$_{ik}$ = 1, R$_{ik}$ = 0')
    pd.DataFrame(res21, index=Hs, columns=cols).plot(ax=plt.gca())
    plt.axvline(1 / n, linestyle='--', color='k', linewidth=0.5)
    plt.axhline(0, linestyle='--', color='k', linewidth=0.5)
    plt.axhline(1, linestyle='--', color='k', linewidth=0.5)
    plt.xlabel('$H_{ik}$')
    plt.ylim(ylim2)

    plt.subplot(2, 2, 4)
    plt.title('P$_{ik}$ = 1, R$_{ik}$ = 1')
    pd.DataFrame(res22, index=Hs, columns=cols).plot(ax=plt.gca())
    plt.axvline(1 / n, linestyle='--', color='k', linewidth=0.5)
    plt.axhline(0, linestyle='--', color='k', linewidth=0.5)
    plt.axhline(1, linestyle='--', color='k', linewidth=0.5)
    plt.xlabel('$H_{ik}$')
    plt.ylim(ylim2)

    f.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    # plt.tight_layout()
    # plt.show()

if __name__=='__main__':
    plot_KF_behaviour()

    # print(K11)
    # print(K12)
    # print(K21)
    # print(K22)
    # print(np.dot(P1,H))
    # print(np.dot(np.dot(H,P1),H) + R1)
    # print(np.linalg.inv(np.dot(np.dot(H,P1),H) + R1))

    # print(K)
    #
    # print(K.sum(axis=0))
    # print(xa)