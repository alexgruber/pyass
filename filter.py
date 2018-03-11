
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

    for t, f in forcing.iterrows():
        x, P = model.step(f)

        y = obs.loc[t,:]
        x_upd, P_upd = analyse(x.values, y.values, P, R, H=H)

        x_ana[t], P_ana[t] = x_upd, P_upd
        model.x[:], model.P[:] = x_upd, P_upd

    return x_ana, P_ana


def generate_ensemble(data, n_ens, perturbation=None):

    ens = [data.copy() for n in np.arange(n_ens)]
    if perturbation is not None:
        for var, param in perturbation.iteritems():
            pert = getattr(np.random, param[0])
            for n in np.arange(n_ens):
                if param[1] == 'additive':
                    ens[n][var] = ens[n][var] + pert(0, np.sqrt(param[2]), len(data))
                else:
                    ens[n][var] = ens[n][var] * pert(0, np.sqrt(param[2]), len(data))
    return ens

from timeit import default_timer

def EnKF(model, forcing, obs, force_pert=None, obs_pert=None, H=None, n_ens=24):

    mod_ens = [deepcopy(model) for n in np.arange(n_ens)]

    frc_ens = generate_ensemble(forcing, n_ens, force_pert)
    obs_ens = generate_ensemble(obs, n_ens, obs_pert)

    x_ana = pd.DataFrame(index=forcing.index, columns=model.x.index, dtype='float')
    P_ana = pd.DataFrame(index=forcing.index, columns=model.x.index, dtype='float')

    for t in np.arange(len(forcing)):

        # model step for each ensemble member
        x_ens = np.full((n_ens, len(model.x)), np.nan)
        y_ens = np.full((n_ens, len(obs.columns)), np.nan)
        for n in np.arange(n_ens):
            x, P = mod_ens[n].step(frc_ens[n].loc[t,:])
            x_ens[n][:] = x
            y_ens[n][:] = obs_ens[n].loc[t,:]

        # diagnose model and observation error from the ensemble
        P = x_ens.var(axis=0)
        R = y_ens.var(axis=0)

        # update state of each ensemble member
        x_ens_upd = np.full((n_ens, len(model.x)), np.nan)
        for n in np.arange(n_ens):
            x_ens_upd[n], P_ens_upd = analyse(x_ens[n], y_ens[n], P, R, H=H)
            mod_ens[n].x[:] = x_ens_upd[n]

        # diagnose analysis mean and -error
        x_ana.loc[t,:] = x_ens_upd.mean(axis=0)
        P_ana.loc[t,:] = x_ens_upd.var(axis=0)

    return x_ana, P_ana



