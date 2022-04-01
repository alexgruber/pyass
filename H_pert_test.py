
import seaborn as sns
sns.set_context('talk', font_scale=0.8)
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from pathlib import Path
from itertools import repeat, combinations
from myprojects.functions import merge_files
from pathos.multiprocessing import ProcessPool
import numpy as np
import pandas as pd

from numpy.linalg import inv
from numpy import dot

from copy import deepcopy

from scipy.stats import pearsonr
import scipy.ndimage as ndimage

from myprojects.readers.gen_syn_data import generate_soil_moisture

from pyapi.api import API
from pyass.filter import EnKF, generate_ensemble, generate_perturbations, analyse

def EnKF(model, forcing, obs, obs_pert, force_pert=None, mod_pert=None, H=None, n_ens=24):

    if H is None:
        H = np.ones(len(forcing))

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

            # update state of each ensemble member
            x_ens_upd = np.full(n_ens, np.nan)
            for n in np.arange(n_ens):
                x_ens_upd[n], P_ens_upd, K_vec[n] = analyse(x_ens[n], y_ens[n], P, R, H=H[t])
                mod_ens[n].x = x_ens_upd[n]

            # Store Kalman gain
            K_arr[t] = K_vec.mean()

            # diagnose analysis mean and -error
            x_ana[t] = x_ens_upd.mean()
            P_ana[t] = x_ens_upd.var(ddof=1)
        else:
            x_ana[t] = x_ens.mean()
            P_ana[t] = x_ens.var(ddof=1)

    K = np.nanmean(K_arr)

    return x_ana, P_ana, K


def calc_analysis_error(part, parts):

    res_path = Path('/Users/u0116961/Documents/work/MadKF/synthetic_experiment/H_pert_test').expanduser()
    if not res_path.exists():
        Path.mkdir(res_path, parents=True)

    result_file = res_path / ('syn_exp_part%i.csv' % part)

    n = 1000
    n_ens = 30
    gamma = 0.85
    SNR_obs = 3
    SNR_mod = 3
    SNR_H = 100

    H_true = 10

    sm_true, precip_true = generate_soil_moisture(n, gamma=0.85, scale=7, anomaly=True)

    api = API(gamma=gamma)

    R = sm_true.var() / SNR_obs
    Q = sm_true.var() / SNR_mod * (1 - gamma ** 2)
    H_var = sm_true.var() / SNR_H

    obs_err = np.random.normal(0, np.sqrt(R), n)
    forc_err = np.random.normal(0, np.sqrt(Q), n)

    H_err = np.random.normal(0, np.sqrt(H_var), n)
    H = H_true + H_err

    obs = (sm_true + obs_err) * H_true
    forcing = precip_true + forc_err

    # --- get OL ts & error ---
    OL = np.full(n, np.nan)
    model = deepcopy(api)
    for t, f in enumerate(forcing):
        x = model.step(f)
        OL[t] = x

    n_res = 50

    Rs = np.linspace(1,6,n_res) * R
    Qs = np.linspace(1,6,n_res) * Q

    Rs, Qs = np.meshgrid(Rs, Qs)
    Rs = Rs.flatten()
    Qs = Qs.flatten()

    idx = np.arange(len(Rs))

    subs = (np.arange(parts + 1) * len(idx) / parts).astype('int')
    subs[-1] = len(idx)
    start = subs[part - 1]
    end = subs[part]

    res = pd.Series(index=idx[start:end])

    for cnt, i in enumerate(idx[start:end]):

        R = Rs[i]
        Q = Qs[i]

        print(f'{cnt} / {len(res)}')

        obs_pert = ['normal', 'additive', R * H_true ** 2]
        force_pert = ['normal', 'additive', Q]

        x_ana, P_ana, K = EnKF(api, forcing, obs, obs_pert, force_pert=force_pert, H=H, n_ens=n_ens)

        res.loc[i] = np.mean((sm_true - x_ana) ** 2)

    if not result_file.exists():
        res.to_csv(result_file, float_format='%0.4f')
    else:
        res.to_csv(result_file, float_format='%0.4f', mode='a', header=False)

    # print(res.mean())
    # np.save('/Users/u0116961/Documents/work/MadKF/synthetic_experiment/H_pert_test/result', res)

    res = np.full(1, np.nan)
    obs_pert = ['normal', 'additive', Rs[0] * H_true ** 2]
    force_pert = ['normal', 'additive', Qs[0]]
    H = np.full(n, H_true)
    x_ana, P_ana, K = EnKF(api, forcing, obs, obs_pert, force_pert=force_pert, H=H, n_ens=n_ens)

    res[0] = np.mean((sm_true - x_ana) ** 2)

    np.save('/Users/u0116961/Documents/work/MadKF/synthetic_experiment/H_pert_test/ref', res)


def run(n_procs=1):

    res_path = Path('/Users/u0116961/Documents/work/MadKF/synthetic_experiment/H_pert_test').expanduser()
    if not res_path.exists():
        Path.mkdir(res_path, parents=True)

    part = np.arange(n_procs) + 1
    parts = repeat(n_procs, n_procs)

    if n_procs > 1:
        with ProcessPool(n_procs) as p:
            p.map(calc_analysis_error, part, parts)
    else:
        calc_analysis_error(1, 1)

    merge_files(res_path, pattern='syn_exp_part*.csv', fname='syn_exp.csv', delete=True)


def plot_error():

    arr = pd.read_csv('/Users/u0116961/Documents/work/MadKF/synthetic_experiment/H_pert_test1/syn_exp.csv', index_col=0)
    arr_ref = np.load('/Users/u0116961/Documents/work/MadKF/synthetic_experiment/H_pert_test1/ref.npy')

    n_res = int(np.sqrt(len(arr)))

    arr = arr.values.reshape((n_res,n_res))

    Rs = np.linspace(1,6,n_res)
    Qs = np.linspace(1,6,n_res)

    f = plt.figure(figsize=(7,6))

    # ax = plt.subplot(1,2,1)
    #
    # im = plt.pcolormesh(Rs, Qs, arr, cmap='viridis')
    # ax = plt.gca()
    #
    # plt.xlabel('R inflation factor')
    # plt.ylabel('P inflation factor')
    # plt.title('Absolute DA skill')
    #
    # f.colorbar(im, ax=ax)

    # ax = plt.subplot(1, 2, 2)

    arr = ndimage.gaussian_filter(arr, sigma=(5, 5), order=0)

    im = plt.pcolormesh(Rs, Qs, arr/arr_ref, cmap='viridis')
    ax = plt.gca()

    plt.xlabel('Increasing observation error')
    plt.title('Skill deterioration (w.r.t. no H error)')
    plt.ylabel('Increasing model error')

    f.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

if __name__=='__main__':

    # run(35)
    plot_error()



    # R_true = obs_err.var()
    # Q_true = forc_err.var()
    # P_true = Q_true / (1 - gamma ** 2)
    #
    # MSE_OL = np.mean((sm_true - OL) ** 2)
    # MSE_obs = np.mean((sm_true - obs / H_true) ** 2)
    #
    # MSE_ana = np.mean((sm_true - x_ana) ** 2)
    # MSE_ana_diag = P_ana.mean()
    #
    # print(f'R (true): {R_true}')
    # print(f'R (est): {MSE_obs}')
    # print('')
    # print(f'P (true): {P_true}')
    # print(f'P (est): {MSE_OL}')
    # print('')
    # print(f'P+ (diag): {MSE_ana_diag}')
    # print(f'P+ (est): {MSE_ana}')

















