
import numpy as np
import pandas as pd

from numpy.linalg import inv
from numpy import dot

from copy import deepcopy

from scipy.stats import pearsonr

def analyse(x, y, P, R, H=None):

    if H is None:
        H = 1.
    K = P * (H**2*R + P)**-1

    x_upd = x + K * (H*y - x)
    P_upd = (1 - K) * P

    return x_upd, P_upd, K


def analyse_2d(x, y, P, R, H=None):

    x, y = np.atleast_2d(x), np.atleast_2d(y)
    P, R = np.atleast_2d(P), np.atleast_2d(R)

    if H is None:
        H = np.atleast_2d(np.ones(len(x))).T

    H = H.reshape((H.size,1))
    x = x.reshape((x.size,1))
    y = y.reshape((y.size,1))

    R = np.identity(R.size) * R

    K = dot(P, dot(H.T, inv(R + dot(H, dot(P, H.T)))))
    x_upd = x + dot(K, y - dot(H, x))
    P_upd = dot(np.identity(P.shape[0]) - dot(K, H), P)

    return x_upd, P_upd, K.flatten()

def KF(model, forcing, obs, R, H=None):

    dummy = np.full(len(forcing), np.nan)
    x_ana, P_ana, innov, norm_innov, K_arr = dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy()

    for t, f in enumerate(forcing):
        x, P = model.step(f)
        y = obs[t]

        if not np.isnan(y):
            innov[t] = (H*y - x)
            norm_innov[t] = (H*y - x) / np.sqrt(P + H**2 * R)

            x_ana[t], P_ana[t], K_arr[t] = analyse(x, y, P, R, H=H)
            model.x, model.P = x_ana[t], P_ana[t]
        else:
            x_ana[t], P_ana[t] = x, P

    innov_l1 = innov[1::]
    innov = innov[0:-1]
    ind = np.where(~np.isnan(innov) & ~np.isnan(innov_l1))
    R_innov = pearsonr(innov[ind], innov_l1[ind])[0]

    K = np.nanmean(K_arr)
    check_var = np.nanvar(norm_innov,ddof=1)

    return x_ana, P_ana, R_innov, check_var, K

def KF_2D(model, forcing, obs1, obs2, R, H=None):

    dummy = np.full(len(forcing), np.nan)
    x_ana, P_ana, K1, K2 = dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy()
    norm_innov1, norm_innov2, norm_innov3 = dummy.copy(), dummy.copy(), dummy.copy()

    for t, f in enumerate(forcing):
        x, P = model.step(f)
        y = np.array([obs1[t], obs2[t]])

        if not np.isnan(y[0]):
            norm_innov1[t] = (y[0]/H[0] - x) / np.sqrt(P + R[0]/H[0]**2)
        if not np.isnan(y[1]):
            norm_innov2[t] = (y[1]/H[1] - x) / np.sqrt(P + R[1]/H[1]**2)

        if len(np.where(np.isnan(y))[0]) == 0:
            norm_innov3[t] = (2*x - y[0]/H[0] - y[1]/H[1]) / np.sqrt(4*P + R[0]/H[0]**2 + R[1]/H[1]**2)

        tmp_R = R.copy()
        ind_nan = np.where(np.isnan(y))
        y[ind_nan] = 0.
        tmp_R[ind_nan] = 999999.
        x_ana[t], P_ana[t], (K1[t], K2[t]) = analyse_2d(x, y, P, tmp_R, H=H)
        model.x, model.P = x_ana[t], P_ana[t]

    norm_innov1 = np.nanvar(norm_innov1,ddof=1)
    norm_innov2 = np.nanvar(norm_innov2,ddof=1)
    norm_innov3 = np.nanvar(norm_innov3,ddof=1)

    K1 = np.nanmean(K1)
    K2 = np.nanmean(K2)

    return x_ana, P_ana, norm_innov1, norm_innov2, norm_innov3, K1, K2

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

def generate_perturbations(n_dates, n_ens, params):

    pdf = getattr(np.random, params[0])
    pert = pdf(0, np.sqrt(params[1]), n_dates*n_ens).reshape((n_dates, n_ens))
    return pert


def EnKF(model, forcing, obs, obs_pert, force_pert=None, mod_pert=None, H=None, n_ens=24):

    if H is None:
        H = 1.

    mod_ens = [deepcopy(model) for n in np.arange(n_ens)]

    if force_pert is not None:
        frc_ens = generate_ensemble(forcing, n_ens, force_pert)
    else:
        frc_ens = generate_ensemble(forcing, n_ens, ['normal', 'additive', 0])

    if mod_pert is not None:
        mod_err = generate_perturbations(len(forcing), n_ens, mod_pert)
    else:
        mod_err = generate_perturbations(len(forcing), n_ens, ['normal', 0])


    obs_ens = generate_ensemble(obs, n_ens, obs_pert)

    n_dates = len(forcing)

    x_ana = np.full(n_dates, np.nan)
    P_ana = np.full(n_dates, np.nan)
    K_arr = np.full(n_dates, np.nan)

    innov = np.full(n_dates, np.nan)
    norm_innov = np.full(n_dates, np.nan)

    for t in np.arange(n_dates):

        # model step for each ensemble member
        x_ens = np.full(n_ens, np.nan)
        y_ens = np.full(n_ens, np.nan)
        K_vec = np.full(n_ens, np.nan)
        for n in np.arange(n_ens):
            x_ens[n] = mod_ens[n].step(frc_ens[t, n], err=mod_err[t, n])
            y_ens[n] = obs_ens[t, n]

        # check if there is an observation to assimilate
        if ~np.isnan(obs[t]):

            # diagnose model and observation error from the ensemble
            P = x_ens.var(ddof=1)
            R = y_ens.var(ddof=1)

            innov[t] = H*y_ens.mean() - x_ens.mean()
            norm_innov[t] = (H*(y_ens.mean()) - x_ens.mean()) / np.sqrt(P + H**2*R)

            # update state of each ensemble member
            x_ens_upd = np.full(n_ens, np.nan)
            for n in np.arange(n_ens):
                x_ens_upd[n], P_ens_upd, K_vec[n] = analyse(x_ens[n], y_ens[n], P, R, H=H)
                mod_ens[n].x = x_ens_upd[n]

            # Store Kalman gain
            K_arr[t] = K_vec.mean()

            # diagnose analysis mean and -error
            x_ana[t] = x_ens_upd.mean()
            P_ana[t] = x_ens_upd.var(ddof=1)
        else:
            x_ana[t] = x_ens.mean()
            P_ana[t] = x_ens.var(ddof=1)

    innov_l1 = innov[1::]
    innov = innov[0:-1]
    ind = np.where(~np.isnan(innov) & ~np.isnan(innov_l1))
    R_innov = pearsonr(innov[ind], innov_l1[ind])[0]

    check_var = np.nanvar(norm_innov, ddof=1)
    K = np.nanmean(K_arr)

    return x_ana, P_ana, R_innov, check_var, K

def TCA(obs, ol, ana, c_obs_ol, c_obs_ana, c_ol_ana):

    mask = ~np.isnan(obs)

    C = np.cov(np.vstack((obs[mask],ol[mask],ana[mask])))
    C[0,1] -= abs(c_obs_ol[mask].mean())
    C[0,2] -= abs(c_obs_ana[mask].mean())
    C[1,2] -= abs(c_ol_ana[mask].mean())

    R = abs(C[0,0] - abs(C[0,1] * C[0,2] / C[1,2]))
    P = abs(C[1,1] - abs(C[0,1] * C[1,2] / C[0,2]))

    H = C[1,2] / C[0,2]

    return R, P, H


def MadKF(model, forcing, obs, n_ens=100, n_iter=20):

    n_dates = len(forcing)

    # Get initial values for P and Q
    ol = np.array([deepcopy(model).step(f) for f in forcing])
    R = np.nanmean((obs-ol)**2)
    P_TC = R
    Q = R * (1 - model.gamma ** 2)
    H = 1

    for k in np.arange(n_iter):

        # iterative update of R and Q
        if k > 0:
            R, P_TC, H = TCA(y, x_ol, x_ana, c_obs_ol, c_obs_ana, c_ol_ana)
            Q = P_TC * (1 - model.gamma ** 2)

        # initialize variables
        dummy = np.full(n_dates, np.nan)
        x_ol, x_ana, P_ana, y = dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy()
        c_obs_ol, c_obs_ana, c_ol_ana = dummy.copy(), dummy.copy(), dummy.copy()
        innov = dummy.copy()
        norm_innov = dummy.copy()
        K_arr = dummy.copy()

        # create model instance ensemble for OL run and filter run
        ol_ens = [deepcopy(model) for n in np.arange(n_ens)]
        kf_ens = [deepcopy(model) for n in np.arange(n_ens)]

        # create forcing and observation ensemble
        obs_ens = generate_ensemble(obs, n_ens, ['normal', 'additive', R])

        frc_ens = generate_ensemble(forcing, n_ens, ['normal', 'additive', 0])

        mod_pert = ['normal', Q]
        mod_err = generate_perturbations(len(forcing), n_ens, mod_pert)

        # EnKF run
        for t in np.arange(n_dates):

            dummy = np.full(n_ens, np.nan)
            x_ens_ol, x_ens, x_ens_upd, y_ens = dummy.copy(), dummy.copy(), dummy.copy(), dummy.copy()
            K_vec = dummy.copy()

            # Ensemble forecast + perturbation
            for n in np.arange(n_ens):
                x_ens_ol[n] = ol_ens[n].step(frc_ens[t, n], err=mod_err[t, n])
                x_ens[n] = kf_ens[n].step(frc_ens[t, n], err=mod_err[t, n])
                y_ens[n] = obs_ens[t, n]

            x_ol[t] = x_ens_ol.mean()
            y[t] = y_ens.mean()

            # check if there is an observation to assimilate
            if ~np.isnan(y[t]):

                # Diagnose model and observation error variance
                P_est = x_ens.var()
                R_est = R

                # Store normalized innovations for self-consistency check
                innov[t] = H*y[t] - x_ens.mean()
                norm_innov[t] = (H*y[t] - x_ens.mean()) / np.sqrt(P_est + R_est * H**2)

                # Ensemble update
                for n in np.arange(n_ens):
                    x_ens_upd[n], P, K_vec[n] = analyse(x_ens[n], y_ens[n], P_est, R_est, H=H)
                    kf_ens[n].x = x_ens_upd[n]

                # Store Kalman gain
                K_arr[t] = K_vec.mean()

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

    innov_l1 = innov[1::]
    innov = innov[0:-1]
    ind = np.where(~np.isnan(innov)&~np.isnan(innov_l1))
    R_innov = pearsonr(innov[ind], innov_l1[ind])[0]

    check_var = np.nanvar(norm_innov)
    K = np.nanmean(K_arr)

    return x_ana, P_ana, R, P_TC, H, R_innov, check_var, K



