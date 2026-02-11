from HelperFunctions import EOS_Morard, liquidus_BW2011

import numpy as np
from typing import Tuple
from astropy import units as u


class PlanetaryBody:
    def __init__(self,
                 Omega: u.Quantity,
                 P_CMB: u.Quantity, P_CEN: u.Quantity, T_OMB: u.Quantity,
                 R_C: u.Quantity, R_M: u.Quantity, R_O: u.Quantity, R_I: u.Quantity,
                 rho_C: u.Quantity, rho_M: u.Quantity, rho_O: u.Quantity, rho_I: u.Quantity,

                 k_C: u.Quantity, etaM_C: u.Quantity, gamma_C: float, mu0_C: u.Quantity, wtS_C: float,
                 k_M: u.Quantity, cp_M: u.Quantity,
                 U_ppm_M: float, Th_ppm_M: float, K_ppm_M: float,

                 U238_hl: u.Quantity, U238_qR: u.Quantity, U238_ft: float,
                 U235_hl: u.Quantity, U235_qR: u.Quantity, U235_ft: float,
                 Th232_hl: u.Quantity, Th232_qR: u.Quantity, Th232_ft: float,
                 K40_hl: u.Quantity, K40_qR: u.Quantity, K40_ft: float,

                 T_Ci: u.Quantity, T_Mi: u.Quantity, rad_coeff: float, Lambda: u.Quantity, S_rich: bool):

        self.Omega = Omega
        self.P_CMB, self.P_CEN, self.T_OMB = P_CMB, P_CEN, T_OMB
        self.R_C, self.R_M, self.R_O, self.R_I = R_C, R_M, R_O, R_I
        self.rho_C, self.rho_M, self.rho_O, self.rho_I = rho_C, rho_M, rho_O, rho_I

        self.k_C, self.etaM_C, self.gamma_C, self.mu0_C, self.wtS_C = k_C, etaM_C, gamma_C, mu0_C, wtS_C
        self.k_M, self.cp_M = k_M, cp_M
        self.U_ppm_M, self.Th_ppm_M, self.K_ppm_M = U_ppm_M, Th_ppm_M, K_ppm_M

        self.T_Ci, self.T_Mi, self.rad_coeff, self.Lambda, self.S_rich = T_Ci, T_Mi, rad_coeff, Lambda, S_rich

        # CALCULATED VALUES
        self.cp_C = EOS_Morard(wt_S=self.wtS_C, P=self.P_CMB.to(u.Pa).value, T=self.T_Ci.value)[2]
        self.T_liq_CMB = liquidus_BW2011(wt_S=wtS_C, P_GPa=P_CMB.to(u.GPa).value) * u.K
        self.T_liq_CEN = liquidus_BW2011(wt_S=wtS_C, P_GPa=P_CEN.to(u.GPa).value) * u.K

        # CALCULATED MASSES
        self.M_C = self._calculate_mass(self.R_C, 0 * u.km, self.rho_C)
        self.M_M = self._calculate_mass(self.R_M, self.R_C, self.rho_M)
        self.M_O = self._calculate_mass(self.R_O, self.R_M, self.rho_O)
        self.M_I = self._calculate_mass(self.R_I, self.R_O, self.rho_I)
        self.M_TOTAL = self.M_C + self.M_M + self.M_O + self.M_I

        # MAKE ISOTOPE DICT
        self.iso_dict = self._make_iso_dict(U238_hl, U238_qR, U238_ft,
                                            U235_hl, U235_qR, U235_ft,
                                            Th232_hl, Th232_qR, Th232_ft,
                                            K40_hl, K40_qR, K40_ft)

        # INITIALIZE DATA STRUCTURES
        self.R_arr, self.T_arr, self.cp_arr, self.rho_arr, self.k_arr = self._initialize_arrays()
        self.t_arr, self.T_mat, self.F_C_arr = None, None, None  # from thermal_evolution
        self.F_x_arr, self.F_th_arr, self.F_T_arr, self.T_CEN_arr = None, None, None, None  # from core_heat
        self.Q_ad_arr, self.Q_C_arr, self.Re_m_arr, self.B_S_arr = None, None, None, None  # from core_heat
        self.inner_core, self.R_IC_arr, self.xS_arr, self.xEUT_arr, self.X_arr = False, None, None, None, None  # from inner_core
        self.iron_snow, self.T_liq_mask, self.HSt_arr = False, None, None  # from iron_snow
        self.FeS_layer = False
        self.dynamo_cessation_time = None
        self.crystal_start_time = None

    @staticmethod
    def _calculate_mass(R_outer: u.Quantity, R_inner: u.Quantity, rho: u.Quantity) -> u.Quantity:
        return (4 / 3 * np.pi * (R_outer ** 3 - R_inner ** 3) * rho).to(u.kg)

    def _initialize_arrays(self,
                           n: int = 100) -> Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
        R_arr = np.linspace(self.R_C, self.R_M, n)  # radial distance array

        T_arr = np.full(n, self.T_Mi) * self.T_Mi.unit  # thermal profile
        cp_arr = np.full(n, self.cp_M) * self.cp_M.unit  # specific heat capacity profile
        rho_arr = np.full(n, self.rho_M) * self.rho_M.unit  # density profile
        k_arr = np.full(n, self.k_M) * self.k_M.unit  # thermal conductivity profile

        T_arr[0], cp_arr[0], rho_arr[0], k_arr[0] = self.T_Ci, self.cp_C, self.rho_C, self.k_C  # core values
        T_arr[-1] = self.T_OMB  # set boundary condition

        return u.Quantity(R_arr), u.Quantity(T_arr), u.Quantity(cp_arr), u.Quantity(rho_arr), u.Quantity(k_arr)

    def _make_iso_dict(self,
                     U238_hl: u.Quantity, U238_qR: u.Quantity, U238_ft: float,
                     U235_hl: u.Quantity, U235_qR: u.Quantity, U235_ft: float,
                     Th232_hl: u.Quantity, Th232_qR: u.Quantity, Th232_ft: float,
                     K40_hl: u.Quantity, K40_qR: u.Quantity, K40_ft: float) -> dict:

        def N_t(N_0: float, hl: u.Quantity, t: u.Quantity) -> float:
            lambda_ = np.log(2) / hl
            return (N_0 * np.exp(-lambda_ * t)).value

        iso_props = {'U238': [U238_hl, U238_qR, U238_ft, self.U_ppm_M], 'U235': [U235_hl, U235_qR, U235_ft, self.U_ppm_M], 'Th232': [Th232_hl, Th232_qR, Th232_ft, self.Th_ppm_M], 'K40': [K40_hl, K40_qR, K40_ft, self.K_ppm_M]}
        iso_dict = {}

        for iso, props in iso_props.items():
            hl, qR, ft, Z_ppm = props
            ft /= 100
            lamb = np.log(2) / hl
            ppmt = Z_ppm * ft
            ppm0 = N_t(N_0=ppmt, hl=hl, t=-4.5 * u.Gyr)
            qR0 = (qR * ppm0 * 1e-6).to(u.W / u.kg)
            qR0_sc = qR0 * self.rad_coeff

            iso_dict[iso] = {'hl': hl, 'qR': qR, 'ft': ft, 'lamb': lamb, 'ppmt': ppmt, 'ppm0': ppm0, 'qR0': qR0, 'qR0_sc': qR0_sc}

        return iso_dict

    def summarize_results(self):
        t_vals = self.t_arr.to(u.Gyr).value
        B_S_vals = self.B_S_arr.to(u.uT).value

        present_index = np.abs(t_vals - 4.5).argmin()
        present_dynamo = B_S_vals[present_index] > 0

        if B_S_vals[-1] == 0:
            dynamo_cessation_index = len(B_S_vals) - np.argmax(B_S_vals[::-1] != 0) - 1
            self.dynamo_cessation_time = t_vals[dynamo_cessation_index]
        else:
            self.dynamo_cessation_time = 1000  # beyond model time

        print(f'Dynamo Cessation Time: {self.dynamo_cessation_time:.3f} Gyr')


class Triton(PlanetaryBody):
    def __init__(self, T_Ci: u.Quantity, T_Mi: u.Quantity, rad_coeff: float, Lambda: u.Quantity, wtS_C: float = 12.506, k_C: u.Quantity = 5.0 * u.W / u.m / u.K):
        super().__init__(
            Omega=1.237432e-5 * u.rad / u.s,
            P_CMB=1.9 * u.GPa, P_CEN=2.8 * u.GPa, T_OMB=250 * u.K,
            R_C=425 * u.km, R_M=1028 * u.km, R_O=1111 * u.km, R_I=1353 * u.km,
            rho_C=6 * u.g / u.cm ** 3, rho_M=3.3 * u.g / u.cm ** 3, rho_O=1 * u.g / u.cm ** 3, rho_I=0.95 * u.g / u.cm ** 3,

            k_C=k_C, etaM_C=2.0 * u.m**2 / u.s, gamma_C=0.15, mu0_C=4e-07 * np.pi * u.H / u.m, wtS_C=wtS_C,
            k_M=4.2 * u.W / u.m / u.K, cp_M=900 * u.J / u.kg / u.K,
            U_ppm_M=0.008, Th_ppm_M=0.03, K_ppm_M=544,

            U238_hl=4.46e9 * u.yr, U238_qR=95.13 * u.uW / u.kg, U238_ft=99.27,
            U235_hl=7.04e8 * u.yr, U235_qR=568.47 * u.uW / u.kg, U235_ft=0.72,
            Th232_hl=1.40e10 * u.yr, Th232_qR=26.3 * u.uW / u.kg, Th232_ft=100,
            K40_hl=1.26e9 * u.yr, K40_qR=28.47 * u.uW / u.kg, K40_ft=0.0117,

            T_Ci=T_Ci, T_Mi=T_Mi, rad_coeff=rad_coeff, Lambda=Lambda, S_rich=False if wtS_C < 28.5 else True
        )

        if wtS_C > 28.5:
            self._calculate_core()
            # print(f'CORE SIZE: {self.R_C}')
            # print(f'CORE Cp: {self.cp_C}')
            self.R_arr, self.T_arr, self.cp_arr, self.rho_arr, self.k_arr = self._initialize_arrays()

    def _calculate_core(self):
        M_RM = self.M_TOTAL - self.M_O - self.M_I
        rho_C = EOS_Morard(wt_S=self.wtS_C, P=self.P_CEN.to(u.Pa).value, T=self.T_Ci.value)[0]
        R_C = ((3 * M_RM / 4 / np.pi - self.rho_M * self.R_M ** 3) / (rho_C - self.rho_M)) ** (1 / 3)
        self.rho_C = rho_C
        self.R_C = R_C
        self.P_CMB = 1.5 * u.GPa
        self.P_CEN = 2.8 * u.GPa
        print('wtS_C = ', self.wtS_C)
        print('NEW rho_C = ', self.rho_C)
        print('NEW R_C = ', self.R_C.to(u.km))
