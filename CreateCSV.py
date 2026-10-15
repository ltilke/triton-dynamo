import numpy as np
import pandas as pd
from itertools import product


T_Ci_arr = np.arange(1800, 2601, 200)
T_Mi_arr = np.arange(1600, 2001, 100)
rad_coeff_arr = np.arange(0, 2.1, 0.5)

FeSLayer_wtS_C_arr = np.arange(30, 36.1, 1, dtype=float)
FeInner_wtS_C_arr = np.array([0, 5, 10, 12.506, 15, 20, 25])
wtS_C_arr = np.concatenate((FeInner_wtS_C_arr, FeSLayer_wtS_C_arr))
kc_arr = np.array([5, 15, 30])
# wtS_C_arr = np.array([12.506, 33])

#                      Fe Snow, Fe Inner, FeS Layer
Lambda_arr = np.array([ -37e-9,    37e-9,   -138e-9])

df = pd.DataFrame(list(product(T_Ci_arr, T_Mi_arr, rad_coeff_arr, wtS_C_arr, Lambda_arr, kc_arr)), columns=['T_Ci', 'T_Mi', 'rad_coeff', 'wtS_C', 'Lambda', 'k_C'])

df_core_conditions = [
    (df['Lambda'] == -37e-9) & (df['wtS_C'] > 0) & (df['wtS_C'] < 28.5),  # iron snow needs S in the core
    (df['Lambda'] == 37e-9) & (df['wtS_C'] < 28.5),
    (df['Lambda'] == -138e-9) & (df['wtS_C'] > 28.5)
]
df_core_options = ['FeSnow', 'FeInner', 'FeSLayer']
df['core'] = np.select(df_core_conditions, df_core_options, default=None)
df.dropna(inplace=True)

TCi_nom = 2200
TMi_nom = 1800
rad_nom = 1
kc_nom = 5

# IRON SNOW: Lambda = -37e-9, wtS_C = 12.506
# INNER CORE: Lambda = 37e-9, wtS_C = 12.506
# FES LAYER: Lambda = -138e-9, wtS_C > 28.5

wtS_C_nom = {'FeSnow': 12.506, 'FeInner': 12.506, 'FeSLayer': 33}
Lambda_nom = {'FeSnow': -37e-9, 'FeInner': 37e-9, 'FeSLayer': -138e-9}

df_var_conditions = [
    (df['T_Ci'] == TCi_nom) & (df['T_Mi'] == TMi_nom) & (df['rad_coeff'] == rad_nom) & (df['wtS_C'] == [wtS_C_nom[core_type] for core_type in df['core']]) & (df['k_C'] == kc_nom),
    (df['T_Ci'] != TCi_nom) & (df['T_Mi'] == TMi_nom) & (df['rad_coeff'] == rad_nom) & (df['wtS_C'] == [wtS_C_nom[core_type] for core_type in df['core']]) & (df['k_C'] == kc_nom),
    (df['T_Ci'] == TCi_nom) & (df['T_Mi'] != TMi_nom) & (df['rad_coeff'] == rad_nom) & (df['wtS_C'] == [wtS_C_nom[core_type] for core_type in df['core']]) & (df['k_C'] == kc_nom),
    (df['T_Ci'] == TCi_nom) & (df['T_Mi'] == TMi_nom) & (df['rad_coeff'] != rad_nom) & (df['wtS_C'] == [wtS_C_nom[core_type] for core_type in df['core']]) & (df['k_C'] == kc_nom),
    (df['T_Ci'] == TCi_nom) & (df['T_Mi'] == TMi_nom) & (df['rad_coeff'] == rad_nom) & (df['wtS_C'] != [wtS_C_nom[core_type] for core_type in df['core']]) & (df['k_C'] == kc_nom),
    (df['T_Ci'] == TCi_nom) & (df['T_Mi'] == TMi_nom) & (df['rad_coeff'] == rad_nom) & (df['wtS_C'] == [wtS_C_nom[core_type] for core_type in df['core']]) & (df['k_C'] != kc_nom)

]
df_var_options = ['nominal', 'T_Ci', 'T_Mi', 'rad_coeff', 'wtS_C', 'k_C']
df['var'] = np.select(df_var_conditions, df_var_options, default=None)
df.dropna(inplace=True)
df.sort_values(inplace=True, by=['core', 'var'])
df.reset_index(inplace=True, drop=True)

print(df.value_counts('core'))


df.to_csv('Triton_models.csv', index=False)