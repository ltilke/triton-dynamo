from PlanetaryBody import Triton
from ThermalModel import ThermalModel
from HelperFunctions import plot

from astropy import units as u
import pandas as pd
import pickle

import os
import time
import warnings
# import numpy as np
# import multiprocessing as mp

warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in power')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in multiply')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in sqrt')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')


def run_models(df, out_folder: str = ''):
    for i, r in df.iterrows():
        model_TC, model_TM, model_rad, model_wtS, model_Lam, model_kc, core_type = r['T_Ci'], r['T_Mi'], r['rad_coeff'], r['wtS_C'], r['Lambda'], r['k_C'], r['core']
        model_name = f'{core_type}_RemC{Rem_C}_TC{model_TC:.0f}_TM{model_TM:.0f}_rad{model_rad:.1f}_wtS{model_wtS:.3f}_Lam{model_Lam:+.0e}_kc{model_kc:.0f}'
        print(f'i: {i}\tcore: {core_type}\tmodel: {model_name}')

        start_time = time.time()

        model_path = f'Triton_models/{model_name}_{model_version}.pkl'  # version tag avoids reusing stale cached models
        if os.path.isfile(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            triton = Triton(T_Ci=model_TC * u.K, T_Mi=model_TM * u.K, rad_coeff=model_rad, Lambda=model_Lam * u.K / u.Pa, wtS_C=model_wtS, k_C=model_kc * u.W / u.m / u.K)
            model = ThermalModel(pb=triton, t_end=14 * u.Gyr, t_clip=5, out_folder=out_folder, model_name=model_name, Rem_C=Rem_C)

            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60

        df.loc[i, 'dynamo_cessation_time'] = model.pb.dynamo_cessation_time
        print(f'elapsed time: {elapsed_time:.2f} minutes')
        print('**************************************************')

    return df


Rem_C = 40
# Rem_C = 40, 100, 150, 200
model_version = 'CN-v2'  # bump whenever the model physics or numerics change
os.makedirs('Triton_models', exist_ok=True)
os.makedirs('Triton_figs', exist_ok=True)
run_models(df=pd.read_csv('Triton_models.csv'), out_folder='Triton_figs').to_csv(f'Triton_results_RemC{Rem_C}.csv')
