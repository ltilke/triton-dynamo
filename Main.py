from PlanetaryBody import Triton
from ThermalModel import ThermalModel
from HelperFunctions import plot

from astropy import units as u
import pickle


###
# inner core: Lambda = 1e-9 * u.K / u.Pa
# no inner core: Lambda = -1e-9 * u.K / u.Pa
# for sulfur-poor: do not specify wtS_C

# IRON SNOW: Lambda = -37e-9, wtS_C = 12.506
# INNER CORE: Lambda = 37e-9, wtS_C = 12.506
# FES LAYER: Lambda = -138e-9, wtS_C > 28.5

triton_nominal_FeSLayer = Triton(T_Ci=2200 * u.K, T_Mi=1800 * u.K, rad_coeff=1.0, Lambda=-138e-9 * u.K / u.Pa, wtS_C=33)
model_nominal_FeSLayer = ThermalModel(pb=triton_nominal_FeSLayer, t_end=14 * u.Gyr, t_clip=5, show_plots=True)
with open(f'Triton_models/model_nominal_FeSLayer.pkl', 'wb') as f:
    pickle.dump(model_nominal_FeSLayer, f)

triton_nominal_FeSnow = Triton(T_Ci=2200 * u.K, T_Mi=1800 * u.K, rad_coeff=1.0, Lambda=-37e-9 * u.K / u.Pa)
model_nominal_FeSnow = ThermalModel(pb=triton_nominal_FeSnow, t_end=14 * u.Gyr, t_clip=5, show_plots=True)
with open(f'Triton_models/model_nominal_FeSnow.pkl', 'wb') as f:
    pickle.dump(model_nominal_FeSnow, f)

triton_nominal_FeInner = Triton(T_Ci=2200 * u.K, T_Mi=1800 * u.K, rad_coeff=1.0, Lambda=37e-9 * u.K / u.Pa)
model_nominal_FeInner = ThermalModel(pb=triton_nominal_FeInner, t_end=14 * u.Gyr, t_clip=5, show_plots=True)
with open(f'Triton_models/model_nominal_FeInner.pkl', 'wb') as f:
    pickle.dump(model_nominal_FeInner, f)


# with open(f'Triton_models/model_test.pkl', 'rb') as f:
#     model_test = pickle.load(f)
# plot(model_test, show_plots=True)
