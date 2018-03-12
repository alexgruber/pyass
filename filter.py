
import timeit

import numpy as np
import pandas as pd

from numpy.linalg import inv
from numpy import transpose, dot

from copy import deepcopy


def analyse(x, y, P, R, H=None):

    x, y = np.atleast_1d(x), np.atleast_1d(y)
    P, R = np.atleast_2d(P), np.atleast_2d(R)

    if H is None:
        H = np.atleast_2d(np.identity(len(x)))

    K = dot(P, dot(transpose(H), inv(R + dot(H, dot(P, transpose(H))))))
    x_upd = x + dot(K, y - dot(H,x))
    P_upd = dot(np.identity(P.shape[0]) - dot(K, H), P)

    return x_upd, P_upd


def KF(model, forcing, obs, R, H=None):

    x_ana, P_ana = np.full(len(forcing), np.nan), np.full(len(forcing), np.nan)

    for t, f in enumerate(forcing):
        x, P = model.step(f)

        y = obs[t]
        x_upd, P_upd = analyse(x, y, P, R, H=H)

        x_ana[t], P_ana[t] = x_upd, P_upd
        model.x, model.P = x_upd, P_upd

    return x_ana, P_ana


def generate_ensemble(data, n_ens, params):

    n_dates = len(data)
    ens = data.repeat(n_ens).reshape(n_dates, n_ens)
    pert = getattr(np.random, params[0])
    for n in np.arange(n_ens):
        if params[1] == 'additive':
            ens[:, n] = ens[:, n] + pert(0, np.sqrt(params[2]), n_dates)
        else:
            ens[:, n] = ens[:, n] * pert(0, np.sqrt(params[2]), n_dates)
    return ens


def EnKF(model, forcing, obs, force_pert=None, obs_pert=None, H=None, n_ens=24):

    mod_ens = [deepcopy(model) for n in np.arange(n_ens)]

    frc_ens = generate_ensemble(forcing, n_ens, force_pert)
    obs_ens = generate_ensemble(obs, n_ens, obs_pert)

    n_dates = len(forcing)

    x_ana = np.full(n_dates, np.nan)
    P_ana = np.full(n_dates, np.nan)

    for t in np.arange(len(forcing)):

        # model step for each ensemble member
        x_ens = np.full(n_ens, np.nan)
        y_ens = np.full(n_ens, np.nan)
        for n in np.arange(n_ens):
            x_ens[n], P = mod_ens[n].step(frc_ens[t, n])
            y_ens[n] = obs_ens[t, n]

        # diagnose model and observation error from the ensemble
        P = x_ens.var()
        R = y_ens.var()

        # update state of each ensemble member
        x_ens_upd = np.full(n_ens, np.nan)
        for n in np.arange(n_ens):
            x_ens_upd[n], P_ens_upd = analyse(x_ens[n], y_ens[n], P, R, H=H)
            mod_ens[n].x = x_ens_upd[n]

        # diagnose analysis mean and -error
        x_ana[t] = x_ens_upd.mean()
        P_ana[t] = x_ens_upd.var()

    return x_ana, P_ana

import matplotlib.pyplot as plt

def AdaptEnKF(model, forcing, obs, H=None, n_ens=1, n_iter=1):

    n_dates = len(forcing)

    # Get initial values for R and Q
    ol = np.array([deepcopy(model).step(f)[0] for f in forcing])
    R = np.mean((obs-ol)**2)
    Q = R * (1 - model.gamma**2)

    c_obs_ol_ts = np.full(n_iter, np.nan)
    c_obs_ana_ts = np.full(n_iter, np.nan)
    c_ol_ana_ts = np.full(n_iter, np.nan)

    R_ts = np.full(n_iter, np.nan)
    Q_ts = np.full(n_iter, np.nan)

    for k in np.arange(n_iter):

        ol_ens = [deepcopy(model) for n in np.arange(n_ens)]
        mod_ens = [deepcopy(model) for n in np.arange(n_ens)]

        x_ol = np.full(n_dates, np.nan)
        x_ana = np.full(n_dates, np.nan)
        P_ana = np.full(n_dates, np.nan)

        c_obs_ol = np.full(n_dates, np.nan)
        c_obs_ana = np.full(n_dates, np.nan)
        c_ol_ana = np.full(n_dates, np.nan)

        frc_ens = generate_ensemble(forcing, n_ens, ['normal', 'additive', Q])
        obs_ens = generate_ensemble(obs, n_ens, ['normal', 'additive', R])

        for t in np.arange(n_dates):

            # model step for each ensemble members
            x_ol_ens = np.full(n_ens, np.nan)
            x_ens = np.full(n_ens, np.nan)
            y_ens = np.full(n_ens, np.nan)
            for n in np.arange(n_ens):
                x_ol_ens[n] = ol_ens[n].step(frc_ens[t, n])[0]
                x_ens[n] = mod_ens[n].step(frc_ens[t, n])[0]
                y_ens[n] = obs_ens[t, n]

            # calculate OL mean
            x_ol[t] = x_ol_ens.mean()

            # diagnose model and observation error from the ensemble
            P_est = x_ens.var()
            R_est = y_ens.var()

            # update state of each ensemble member
            x_ens_upd = np.full(n_ens, np.nan)
            for n in np.arange(n_ens):
                x_ens_upd[n] = analyse(x_ens[n], y_ens[n], P_est, R_est, H=H)[0]
                mod_ens[n].x = x_ens_upd[n]

            # diagnose analysis mean and -error
            x_ana[t] = x_ens_upd.mean()
            P_ana[t] = x_ens_upd.var()

            # diagnose error covariances
            c_obs_ol[t] = np.cov(y_ens,x_ol_ens)[0,1]
            c_obs_ana[t] = np.cov(y_ens,x_ens_upd)[0,1]
            c_ol_ana[t] = np.cov(x_ol_ens,x_ens_upd)[0,1]

        c_obs_ol = c_obs_ol.mean()
        c_obs_ana = c_obs_ana.mean()
        c_ol_ana = c_ol_ana.mean()

        R = np.mean((obs - x_ol) * (obs - x_ana)) + c_obs_ana + c_obs_ol - c_ol_ana
        Q = (np.mean((x_ol - obs) * (x_ol - x_ana)) + c_ol_ana + c_obs_ol - c_obs_ana) * (1 - model.gamma**2)

        c_obs_ol_ts[k] = c_obs_ol
        c_obs_ana_ts[k] = c_obs_ana
        c_ol_ana_ts[k] = c_ol_ana

        R_ts[k] = R
        Q_ts[k] = Q

    R0 = np.array(60).repeat(n_iter)
    Q0 = np.array(40).repeat(n_iter)

    # pd.DataFrame({'obs_ol': c_obs_ol_ts, 'obs_ana': c_obs_ana_ts, 'ol_ana':c_ol_ana_ts, 'R' : R_ts, 'Q': Q_ts, 'R0':R0, 'Q0':Q0}).plot()
    pd.DataFrame({'R' : R_ts, 'Q': Q_ts, 'R_true':R0, 'Q_true':Q0}).plot()
    plt.show()

    return x_ana, P_ana



