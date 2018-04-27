
import numpy as np
import pandas as pd

from numpy.linalg import inv
from numpy import transpose, dot

from copy import deepcopy


def analyse(x, y, P, R, H=None):


    # x, y = np.atleast_1d(x), np.atleast_1d(y)
    if H is None:
        H = 1
        # H = np.identity(len(x))
    # P, R, H = np.atleast_2d(P), np.atleast_2d(R), np.atleast_2d(H)

    # K = dot(P, dot(transpose(H), inv(R + dot(H, dot(P, transpose(H))))))
    # x_upd = x + dot(K, y - dot(H,x))
    # P_upd = dot(np.identity(P.shape[0]) - dot(K, H), P)

    # K = P * (H*R*H.T + P)**-1
    K = P * (H**2*R + P)**-1
    x_upd = x + K * (H*y - x)
    P_upd = (1 - K) * P

    # return np.array(x_upd).flatten(), np.array(P_upd).flatten()
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


def EnKF(model, forcing, obs, force_pert, obs_pert, H=None, n_ens=24):

    mod_ens = [deepcopy(model) for n in np.arange(n_ens)]

    frc_ens = generate_ensemble(forcing, n_ens, force_pert)
    obs_ens = generate_ensemble(obs, n_ens, obs_pert)

    n_dates = len(forcing)

    x_ana = np.full(n_dates, np.nan)
    P_ana = np.full(n_dates, np.nan)

    norm_innov = np.full(n_dates, np.nan)

    for t in np.arange(len(forcing)):

        # model step for each ensemble member
        x_ens = np.full(n_ens, np.nan)
        y_ens = np.full(n_ens, np.nan)
        for n in np.arange(n_ens):
            x_ens[n] = mod_ens[n].step(frc_ens[t, n])
            y_ens[n] = obs_ens[t, n]

        # check if there is an observation to assimilate
        if ~np.isnan(obs[t]):

            # diagnose model and observation error from the ensemble
            P = x_ens.var()
            R = y_ens.var()

            norm_innov[t] = (y_ens.mean() - x_ens.mean()) / np.sqrt(P + R)

            # update state of each ensemble member
            x_ens_upd = np.full(n_ens, np.nan)
            for n in np.arange(n_ens):
                x_ens_upd[n], P_ens_upd = analyse(x_ens[n], y_ens[n], P, R, H=H)
                mod_ens[n].x = x_ens_upd[n]

            # diagnose analysis mean and -error
            x_ana[t] = x_ens_upd.mean()
            P_ana[t] = x_ens_upd.var()
        else:
            x_ana[t] = x_ens.mean()
            P_ana[t] = x_ens.var()


    check_var = np.nanvar(norm_innov)

    return x_ana, P_ana, check_var

def TCA(obs, ol, ana, c_obs_ol, c_obs_ana, c_ol_ana, gamma):

    mask = ~np.isnan(obs)

    C = np.cov(np.vstack((obs[mask],ol[mask],ana[mask])))
    C[0,1] -= abs(c_obs_ol[mask].mean())
    C[0,2] -= abs(c_obs_ana[mask].mean())
    C[1,2] -= abs(c_ol_ana[mask].mean())

    R = abs(C[0,0] - abs(C[0,1] * C[0,2] / C[1,2]))
    P = abs(C[1,1] - abs(C[0,1] * C[1,2] / C[0,2]))

    H = C[1,2] / C[0,2]

    Q = P * (1 - gamma**2)

    return R, Q, H


def MadEnKF(model, forcing, obs, n_ens=40, n_iter=10):

    n_dates = len(forcing)

    # Get initial values for R and Q
    ol = np.array([deepcopy(model).step(f) for f in forcing])
    R = np.nanmean((obs-ol)**2)
    Q = R * (1 - model.gamma**2)
    H = 1

    for k in np.arange(n_iter):

        # iterative update of R and Q
        if k > 0:
            R, Q, H =   TCA(y, x_ol, x_ana, c_obs_ol, c_obs_ana, c_ol_ana, model.gamma)

        # initialize variables
        dummy = np.full(n_dates, np.nan)
        x_ol, x_ana, P_ana, y = dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy()
        c_obs_ol, c_obs_ana, c_ol_ana = dummy.copy(), dummy.copy(), dummy.copy()
        norm_innov = dummy.copy()

        # create model instance ensemble for OL run and filter run
        ol_ens = [deepcopy(model) for n in np.arange(n_ens)]
        kf_ens = [deepcopy(model) for n in np.arange(n_ens)]

        # create forcing and observation ensemble
        frc_ens = generate_ensemble(forcing, n_ens, ['normal', 'additive', Q])
        obs_ens = generate_ensemble(obs, n_ens, ['normal', 'additive', R])

        # EnKF run
        for t in np.arange(n_dates):

            dummy = np.full(n_ens, np.nan)
            x_ens_ol, x_ens, x_ens_upd, y_ens = dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy()

            # Ensemble forecast
            for n in np.arange(n_ens):
                x_ens_ol[n] = ol_ens[n].step(frc_ens[t, n])
                x_ens[n] = kf_ens[n].step(frc_ens[t, n])
                y_ens[n] = obs_ens[t, n]
            x_ol[t] = x_ens_ol.mean()
            y[t] = y_ens.mean()

            # check if there is an observation to assimilate
            if ~np.isnan(y[t]):

                # Diagnose model and observation error variance
                P_est = x_ens.var()
                R_est = y_ens.var()

                # Store normalized innovations for self-consistency check
                norm_innov[t] = (H*y[t] - x_ens.mean()) / np.sqrt(P_est + R_est * H**2)

                # Ensemble update
                for n in np.arange(n_ens):
                    x_ens_upd[n], P = analyse(x_ens[n], y_ens[n], P_est, R_est, H=H)
                    kf_ens[n].x = x_ens_upd[n]

                # Diagnose analysis mean and uncertainty
                x_ana[t] = x_ens_upd.mean()
                P_ana[t] = x_ens_upd.var()

                # Diagnose error covariances for adaptive updating
                c_obs_ol[t] = np.cov(y_ens,x_ens_ol)[0,1]
                c_obs_ana[t] = np.cov(y_ens,x_ens_upd)[0,1]
                c_ol_ana[t] = np.cov(x_ens_ol,x_ens_upd)[0,1]

            else:
                x_ana[t] = x_ens.mean()
                P_ana[t] = x_ens.var()

        check_var = np.nanvar(norm_innov)

    return x_ana, P_ana, R, Q, H, check_var



