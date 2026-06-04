from PlanetaryBody import PlanetaryBody
from HelperFunctions import gravity, pressure, x_eutectic, latent, EOS_Morard, plot

import numpy as np
from typing import Tuple
from astropy import units as u
from astropy.constants import G
import matplotlib as mpl
import sys

if 'ipykernel' in sys.modules:
    mpl.use('Agg')
else:
    for _backend in ('macosx', 'TkAgg', 'Qt5Agg', 'Agg'):
        try:
            mpl.use(_backend)
            break
        except (ImportError, ValueError):
            continue


def _thomas(a, b, c, d):
    """Solve a tridiagonal system (sub a, diag b, super c, rhs d) in O(n).
 
    Arrays a, b, c, d are length n; a[0] and c[-1] are unused. Returns the
    solution vector. Non-destructive (works on local copies)."""
    n = len(d)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class ThermalModel:
    def __init__(self,
                 pb: PlanetaryBody, t_end: u.Quantity, t_start: u.Quantity = 0 * u.Gyr, t_clip: int = 10,
                 show_plots: bool = False, out_folder: str = '', model_name: str = '', Rem_C: int = 40):
        self.pb = pb
        self.t_end = t_end
        self.t_start = t_start

        self.Rem_C = Rem_C

        self.pb.t_arr, self.pb.T_mat, self.pb.F_C_arr = self._thermal_evolution(t_clip=t_clip)
        # print('THERMAL EVOLUTION MODELLED')

        self.pb.Q_ad_arr, self.pb.Q_C_arr, self.pb.Re_m_arr, self.pb.B_S_arr, self.pb.R_IC_arr, self.pb.xS_arr, self.pb.xEUT_arr, self.pb.X_arr, self.pb.F_x_arr, self.pb.F_th_arr, self.pb.F_T_arr, self.pb.T_CEN_arr = self._core_heat()
        # print('CORE HEAT MODELLED')

        if self.pb.Lambda.value < 0 and not self.pb.S_rich:
            self._iron_snow()

        if self.pb.inner_core:
            print('INNER CORE MODELLED')

        elif self.pb.iron_snow:
            print('IRON SNOW MODELLED')

        elif self.pb.FeS_layer:
            print('FES LAYER MODELLED')

        self.pb.summarize_results()
        self.summary = {'inner_core': self.pb.inner_core, 'iron_snow': self.pb.iron_snow,
                        'FeS_layer': self.pb.FeS_layer, 'crystal_start_time': self.pb.crystal_start_time,
                        'dynamo_cessation_time': self.pb.dynamo_cessation_time}

        plot(model=self, show_plots=show_plots, out_folder=out_folder, model_name=model_name)
        # print('DONE')

    def _thermal_evolution(self,
                           t_clip: int) -> Tuple[u.Quantity, u.Quantity, u.Quantity]:
        # Crank-Nicolson solver for the 1D spherically-symmetric heat diffusion
        # equation. Replaces the original explicit forward-Euler scheme (adapted
        # from Kevin Trinh). Uses the same flux-conservative, harmonic-mean face
        # discretization, so local flux conservation is preserved exactly; only
        # the time integration changes.
        R_arr = self.pb.R_arr.to(u.m).value
        T_arr = self.pb.T_arr.to(u.K).value
        cp_arr = self.pb.cp_arr.to(u.J / u.kg / u.K).value
        rho_arr = self.pb.rho_arr.to(u.kg / u.m ** 3).value
        k_arr = self.pb.k_arr.to(u.W / u.m / u.K).value

        qR0_M_arr = u.Quantity([sub_dict['qR0_sc'] for sub_dict in self.pb.iso_dict.values()]).to(u.W / u.kg).value
        lamb_arr = u.Quantity([sub_dict['lamb'] for sub_dict in self.pb.iso_dict.values()]).to(1 / u.s).value

        k_C_val = self.pb.k_C.to(u.W / u.m / u.K).value
        R_C_val = self.pb.R_C.to(u.m).value
        M_C_val = self.pb.M_C.to(u.kg).value
        cp_C_val = self.pb.cp_C.to(u.J / u.kg / u.K).value
        cp_M_val = self.pb.cp_M.to(u.J / u.kg / u.K).value
        T_OMB_val = self.pb.T_OMB.to(u.K).value

        t_start = self.t_start.to(u.s).value
        t_end = self.t_end.to(u.s).value

        dR = R_arr[1] - R_arr[0]
        n = len(R_arr)

        kappa_local = k_arr / (rho_arr * cp_arr)
        Kappa_C = (self.pb.k_C / self.pb.rho_C / self.pb.cp_C).to(u.m ** 2 / u.s).value
        kappa_max = max(kappa_local.max(), Kappa_C)
        dt = 3.0 * dR ** 2 / kappa_max # CN is unconditionally stable, no stability safety factor needed.

        # Precompute the spatial-operator coefficients.
        a = np.zeros(n)
        b = np.zeros(n)
        c = np.zeros(n)
        pref = 2.0 / (rho_arr * cp_arr * dR * R_arr ** 2)
        for j in range(1, n - 1):
            a[j] = pref[j] * (R_arr[j] - dR / 2) ** 2 / (dR / k_arr[j] + dR / k_arr[j - 1])
            c[j] = pref[j] * (R_arr[j] + dR / 2) ** 2 / (dR / k_arr[j + 1] + dR / k_arr[j])
            b[j] = -(a[j] + c[j])

        alpha_C = 4 * np.pi * R_C_val ** 2 * k_C_val / (M_C_val * cp_C_val * dR)
        b[0] = -alpha_C
        c[0] = alpha_C

        # Source vector defined as a function of time for CN evaluation.
        def H_source(time):
            H_term = np.sum(qR0_M_arr * np.exp(-lamb_arr * time))
            s = np.zeros(n)
            s[1:] = H_term / cp_M_val
            return s

        t = t_start
        t_arr = np.arange(t_start, t_end, dt)[0::t_clip]  # saved time stamps
        T_mat = np.zeros((n, len(t_arr))) * np.nan  # saved T-profiles
        F_C_arr = np.zeros(len(t_arr)) * np.nan  # CMB heat flux [W / m^2]
        i, k = 0, 0  # timestep index, save index

        # LHS tridiagonal (I - dt/2 L) is constant for pure CN, so build it once.
        lhs_a = -0.5 * dt * a
        lhs_b = 1.0 - 0.5 * dt * b
        lhs_c = -0.5 * dt * c
        # Outer node: Dirichlet. Pin row n-1 to T_OMB (constant row).
        lhs_a[-1] = 0.0
        lhs_b[-1] = 1.0
        lhs_c[-1] = 0.0

        while t < t_end:
            # Midpoint source for second-order accuracy.
            s = H_source(t + 0.5 * dt)

            # Build the explicit (RHS) action: (I + dt/2 L) T^n + dt*s.
            rhs = T_arr.copy()
            LT = np.zeros(n)
            LT[0] = b[0] * T_arr[0] + c[0] * T_arr[1]
            LT[1:-1] = a[1:-1] * T_arr[:-2] + b[1:-1] * T_arr[1:-1] + c[1:-1] * T_arr[2:]
            rhs += 0.5 * dt * LT + dt * s
            rhs[-1] = T_OMB_val  # Dirichlet outer node

            T_arr = _thomas(lhs_a, lhs_b, lhs_c, rhs)

            # Solve for CMB conduction.
            F_CMB = - k_C_val * (T_arr[0] - T_arr[1]) / dR

            # Append results occasionally.
            if i % t_clip == 0 and k < len(t_arr):
                F_C_arr[k] = F_CMB
                T_mat[:, k] = T_arr.copy()
                k += 1

            # Update time and counter
            t += dt
            i += 1

        print()
        return t_arr * u.s, T_mat * u.K, F_C_arr * u.W / u.m ** 2

    def _core_heat(self) -> Tuple[
        u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
        def core_mag(mag_rho, mag_cp, mag_alpha, mag_Q_C, mag_Q_ad):
            F_T = (mag_alpha * gravity(pb=self.pb, R=self.pb.R_C) / (mag_rho * mag_cp) * (mag_Q_C - mag_Q_ad) / (
                        4 * np.pi * self.pb.R_C ** 2)).to(u.m ** 2 / u.s ** 3)  # thermal buoyancy flux
            U = (1.3 * (self.pb.R_C / (self.pb.Omega / u.rad)) ** (1 / 5) * F_T ** (2 / 5)).to(
                u.m / u.s)  # typical flow velocity  # gives a runtime warning when it goes to nan
            mag_Re_m = (U * self.pb.R_C / self.pb.etaM_C).si  # magnetic Reynolds number (dimensionless)

            M = (4 * np.pi * self.pb.R_C ** 3 * self.pb.gamma_C * (mag_rho / self.pb.mu0_C) ** (1 / 2) * (
                        F_T * self.pb.R_C) ** (1 / 3)).to(u.A * u.m ** 2)  # dipole moment
            mag_B_S = (self.pb.mu0_C / (4 * np.pi) * (M / self.pb.R_I ** 3)).to(u.uT)  # surface magnetic field strength

            return mag_Re_m, mag_B_S, F_T

        def inner_core(R_IC: u.Quantity, xS_0: float, Q_C: u.Quantity, Q_ad: u.Quantity, X_OC: float):
            g_IC = gravity(pb=self.pb, R=R_IC)
            g_CMB = gravity(pb=self.pb, R=self.pb.R_C)
            P_IC = pressure(R_IC, self.pb.R_C, self.pb.P_CEN, self.pb.P_CMB)

            xS = xS_0 * self.pb.R_C ** 3 / (self.pb.R_C ** 3 - R_IC ** 3)
            xEUT = x_eutectic(P_IC)

            if xS.value > xEUT.value:
                xS = xEUT
            elif xS.value < 0:
                xS = u.Quantity(0)

            L = latent(xS)  # latent heat of freezing of core

            rho_C_new, _, _, _ = EOS_Morard(wt_S=xS.value * 100, P=self.pb.P_CMB.to(u.Pa).value, T=T_C.to(u.K).value)

            if xS < xEUT:
                dTC_dt = (- Q_C / (
                            self.pb.M_C * self.pb.cp_C + ((4 * np.pi * R_IC ** 2 * L) / (g_IC * self.pb.Lambda)))).to(
                    u.K / u.Gyr)
                dRIC_dt = (- 1 / (rho_C_new * g_IC * self.pb.Lambda) * dTC_dt).to(u.km / u.Gyr)
            else:
                dTC_dt = 0 * u.K / u.Gyr
                dRIC_dt = 0 * u.km / u.Gyr

            rho_OC, gamma_OC, cp_OC, alpha_OC = EOS_Morard(wt_S=xS.value * 100, P=self.pb.P_CMB.to(u.Pa).value,
                                                           T=T_C.to(u.K).value)
            rho_Fe, _, _, _ = EOS_Morard(wt_S=0, P=self.pb.P_CMB.to(u.Pa).value, T=T_C.to(u.K).value)
            rho_S = 2200 * u.kg / u.m ** 3

            M_OC = (4 / 3 * np.pi * (self.pb.R_C ** 3 - R_IC ** 3) * rho_OC).to(u.kg)
            dX_dt = (Q_C / (L * M_OC)).to(1 / u.s)

            # compositional buoyancy flux
            F_x = (g_IC * ((rho_C_new - rho_S) / rho_C_new) * (R_IC / self.pb.R_C) ** 2 * dRIC_dt).to(
                u.m ** 2 / u.s ** 3)

            # thermal buoyancy flux
            F_th = ((alpha_OC * g_CMB) / (rho_OC * cp_OC) * (Q_C - Q_ad) / (4 * np.pi * self.pb.R_C ** 2)).to(
                u.m ** 2 / u.s ** 3)

            F_T = (F_x + F_th).to(u.m ** 2 / u.s ** 3)  # Total buoyancy flux
            OC_thickness = self.pb.R_C - R_IC
            U = (1.3 * (OC_thickness / (self.pb.Omega / u.rad)) ** (1 / 5) * F_T ** (2 / 5)).to(
                u.m / u.s)  # typical flow velocity (m/s)

            Re_m = ((U * OC_thickness) / self.pb.etaM_C).si

            # Magnetic dipole moment
            M_dip = (4 * np.pi * self.pb.R_C ** 3 * self.pb.gamma_C * (rho_OC / self.pb.mu0_C) ** (1 / 2) * (
                        F_T * OC_thickness) ** (1 / 3)).to(u.A * u.m ** 2)  # dipole moment (A m2)
            B_S = ((self.pb.mu0_C * M_dip) / (4 * np.pi * self.pb.R_I ** 3)).to(
                u.uT)  # magnetic field on the surface at the equator (uT)

            if B_S.value < 0:
                B_S = 0 * u.uT

            if X_OC >= 0.6:
                Re_m = u.Quantity(0)
                B_S = 0 * u.uT
                if X_OC >= 1.0:
                    dX_dt = 0 / u.s

            if Re_m.value < self.Rem_C:
                B_S = 0 * u.uT

            return Re_m, B_S, xS, xEUT, dRIC_dt, dTC_dt, dX_dt, F_x, F_th, F_T

        def FeS_layer(R_IC: u.Quantity, xS_0: float, Q_C: u.Quantity, Q_ad: u.Quantity, X_IC: float):
            g_IC = gravity(pb=self.pb, R=R_IC)
            P_IC = pressure(R_IC, self.pb.R_C, self.pb.P_CEN, self.pb.P_CMB)

            xS_FeS = 0.3647
            xS = (xS_0 - xS_FeS) * (self.pb.R_C / R_IC) ** 3 + xS_FeS
            xEUT = x_eutectic(P_IC)

            if xS.value < xEUT.value:
                xS = xEUT

            L = latent(xS)  # latent heat of freezing of core

            if xS > xEUT:
                dTC_dt = (- Q_C / (
                            self.pb.M_C * self.pb.cp_C - ((4 * np.pi * R_IC ** 2 * L) / (g_IC * self.pb.Lambda)))).to(
                    u.K / u.Gyr)
                dRIC_dt = (- 1 / (self.pb.rho_C * g_IC * self.pb.Lambda) * dTC_dt).to(u.km / u.Gyr)
            else:
                dTC_dt = 0 * u.K / u.Gyr
                dRIC_dt = 0 * u.km / u.Gyr

            rho_IC, gamma_IC, cp_IC, alpha_IC = EOS_Morard(wt_S=xS.value * 100, P=self.pb.P_CMB.to(u.Pa).value,
                                                           T=T_C.to(u.K).value)
            rho_FeS, _, _, _ = EOS_Morard(wt_S=xS_FeS * 100, P=self.pb.P_CMB.to(u.Pa).value, T=T_C.to(u.K).value)
            rho_C_new, _, _, _ = EOS_Morard(wt_S=xS.value * 100, P=self.pb.P_CMB.to(u.Pa).value, T=T_C.to(u.K).value)

            M_OC = (4 / 3 * np.pi * (self.pb.R_C ** 3 - R_IC ** 3) * rho_FeS).to(u.kg)
            dX_dt = (Q_C / (L * M_OC)).to(1 / u.s)

            # compositional buoyancy flux
            F_x = (g_IC * ((rho_FeS - rho_C_new) / rho_C_new) * (R_IC / self.pb.R_C) ** 2 * dRIC_dt).to(
                u.m ** 2 / u.s ** 3)

            # thermal buoyancy flux
            F_th = ((alpha_IC * gravity(pb=self.pb, R=self.pb.R_C)) / (rho_IC * cp_IC) * (Q_C - Q_ad) / (
                        4 * np.pi * self.pb.R_C ** 2)).to(u.m ** 2 / u.s ** 3)

            F_T = (F_x + F_th).to(u.m ** 2 / u.s ** 3)  # total buoyancy flux
            U = (1.3 * (R_IC / (self.pb.Omega / u.rad)) ** (1 / 5) * F_T ** (2 / 5)).to(
                u.m / u.s)  # Typical flow velocity (m/s)

            Re_m = ((U * R_IC) / self.pb.etaM_C).si

            # Magnetic dipole moment
            M_dip = (4 * np.pi * R_IC ** 3 * self.pb.gamma_C * (rho_IC / self.pb.mu0_C) ** (1 / 2) * (F_T * R_IC) ** (
                        1 / 3)).to(u.A * u.m ** 2)  # dipole moment (A m2)
            B_S = ((self.pb.mu0_C * M_dip) / (4 * np.pi * self.pb.R_I ** 3)).to(
                u.uT)  # magnetic field on the surface at the equator (uT)

            if B_S.value < 0:
                B_S = 0 * u.uT

            if X_IC >= 0.6:
                Re_m = u.Quantity(0)
                B_S = 0 * u.uT
                if X_IC >= 1.0:
                    dX_dt = 0 / u.s

            if Re_m.value < self.Rem_C:
                B_S = 0 * u.uT

            return Re_m, B_S, xS, xEUT, dRIC_dt, dTC_dt, dX_dt, F_x, F_th, F_T

        T_C_arr = self.pb.T_mat[0]
        T_CEN_arr = []

        Q_ad_arr = []
        Q_C_arr = - (4 * np.pi * self.pb.R_C ** 2 * self.pb.F_C_arr).to(u.TW)
        Re_m_arr = []
        B_S_arr = []

        F_x_arr = []
        F_th_arr = []
        F_T_arr = []

        # INNER CORE/FES LAYER STUFF
        R_IC_IC = 15 * u.km
        R_IC_FES = self.pb.R_C.copy()
        T_C_IC, T_C_FES = np.nan, np.nan
        xS_0 = self.pb.wtS_C / 100
        xEUT_0 = x_eutectic(self.pb.P_CEN)
        X_OC, X_IC = 0, 0
        R_IC_arr = []
        xS_arr = []
        xEUT_arr = []
        X_arr = []
        j = 0

        dt = np.diff(self.pb.t_arr)[0]  # constant save-cadence timestep

        for i in range(len(T_C_arr)):
            T_C = T_C_arr[i]
            Q_C = Q_C_arr[i]

            rho, gamma, cp, alpha = EOS_Morard(wt_S=self.pb.wtS_C, P=self.pb.P_CMB.to(u.Pa).value, T=T_C.value)
            T_CEN = T_C + self.pb.R_C * alpha * gravity(self.pb, self.pb.R_C) * T_C / cp

            Q_ad = (4 * np.pi * self.pb.R_C ** 2 * (
                        self.pb.k_C * alpha * T_C * gravity(pb=self.pb, R=self.pb.R_C) / cp)).to(u.GW)
            Re_m, B_S, F_T = core_mag(rho, cp, alpha, Q_C, Q_ad)

            # INNER CORE
            if T_CEN <= self.pb.T_liq_CEN and not self.pb.S_rich and self.pb.Lambda.value > 0:
                if j == 0:
                    self.pb.inner_core = True
                    self.pb.crystal_start_time = self.pb.t_arr.to(u.Gyr).value[i]
                    T_C_IC = T_CEN.copy()
                Re_m, B_S, xS, xEUT, dRIC_dt, dTC_dt, dX_dt, F_x, F_th, F_T = inner_core(R_IC_IC, xS_0, Q_C, Q_ad, X_OC)

                R_IC_IC += dRIC_dt * dt
                T_C_IC += dTC_dt * dt

                if xS == xEUT:
                    X_OC += (dX_dt * dt).to_value()

                R_IC_arr.append(R_IC_IC.copy())
                xS_arr.append(xS)
                xEUT_arr.append(xEUT)
                X_arr.append(X_OC)
                T_CEN_arr.append(self.pb.T_liq_CEN)

                F_x_arr.append(F_x)
                F_th_arr.append(F_th)
                F_T_arr.append(F_T)

                j += 1

            # FES LAYER
            elif T_C <= self.pb.T_liq_CMB and self.pb.S_rich and self.pb.Lambda.value < 0:
                if j == 0:
                    self.pb.FeS_layer = True
                    self.pb.crystal_start_time = self.pb.t_arr.to(u.Gyr).value[i]
                    T_C_IC = T_C.copy()
                Re_m, B_S, xS, xEUT, dRIC_dt, dTC_dt, dX_dt, F_x, F_th, F_T = FeS_layer(R_IC_FES, xS_0, Q_C, Q_ad, X_IC)

                R_IC_FES += dRIC_dt * dt
                T_C_IC += dTC_dt * dt

                if xS == xEUT:
                    X_IC += (dX_dt * dt).to_value()

                R_IC_arr.append(R_IC_FES.copy())
                xS_arr.append(xS)
                xEUT_arr.append(xEUT)
                X_arr.append(X_IC)
                self.pb.T_mat[0][i] = T_C_IC.copy()
                T_CEN_arr.append(T_CEN)

                F_x_arr.append(F_x)
                F_th_arr.append(F_th)
                F_T_arr.append(F_T)

                j += 1

            else:
                R_IC_arr.append(self.pb.R_C) if self.pb.S_rich and self.pb.Lambda.value < 0 else R_IC_arr.append(
                    0 * u.km)
                xS_arr.append(xS_0)
                xEUT_arr.append(xEUT_0)
                X_arr.append(0)

                F_x_arr.append(0 * u.m ** 2 / u.s ** 3)
                # No compositional buoyancy in the pure-thermal regime, so the
                # thermal flux equals the total flux returned by core_mag.
                F_th_arr.append(F_T)
                F_T_arr.append(F_T)
                T_CEN_arr.append(T_CEN)

            if np.isnan(Re_m):
                Re_m = u.Quantity(0)

            if np.isnan(B_S):
                B_S = 0 * u.uT

            Q_ad_arr.append(Q_ad.copy())
            Re_m_arr.append(Re_m.copy())
            B_S_arr.append(B_S.copy())

        return u.Quantity(Q_ad_arr), u.Quantity(Q_C_arr), u.Quantity(Re_m_arr), u.Quantity(B_S_arr), u.Quantity(
            R_IC_arr), u.Quantity(xS_arr), u.Quantity(xEUT_arr), u.Quantity(X_arr), u.Quantity(F_x_arr), u.Quantity(
            F_th_arr), u.Quantity(F_T_arr), u.Quantity(T_CEN_arr)

    def _iron_snow(self) -> Tuple[
        np.array, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
        def iron_snow(Q_C, x_S, H_St, T_St, stop_growth):
            R_WM = self.pb.R_C - H_St  # radius of the well-mixed region (m)
            P_WM = pressure(R_WM, self.pb.R_C, self.pb.P_CEN, self.pb.P_CMB)

            rho_WM, gamma_WM, cp_WM, alpha_WM = EOS_Morard(wt_S=x_S * 100, P=P_WM.to(u.Pa).value,
                                                           T=T_St.value)  # Density, Gruneisen parameter, specific heat, and thermal expansion for the bulk fluid
            rho_Fe, _, _, _ = EOS_Morard(wt_S=0, P=P_WM.to(u.Pa).value, T=T_St.to(u.K).value)

            M_St = (4 / 3 * np.pi * (self.pb.R_C ** 3 - R_WM ** 3) * rho_WM).to(
                u.kg)  # Total mass of the stable region (kg)
            M_WM = self.pb.M_C - M_St
            g_St = (G * M_WM / R_WM ** 2).to(
                u.m / u.s ** 2)  # Gravitational acceleration at the base of the stable region

            # Key derivatives based on the Fe-FeS liquidus and EOS
            dxl_dTl = - 1 / (3650 * u.K)  # Change in sulfur content at the liquidus with temperature
            # dTl_dP = 3.5e-8 * u.K / u.Pa  # Increase in liquidus temperature with pressure (K/Pa)
            dTl_dxl = 1 / dxl_dTl  # Change in liquidus temperature with sulfur content (K)
            # dTad_dP = (alpha_WM * T_St / (rho_WM * cp_WM)).to(u.K / u.Pa)  # Increase in adiabatic temperature with pressure (K/Pa)

            dTC_dt = (- Q_C / (self.pb.M_C * self.pb.cp_C)).to(u.K / u.s)  # Cooling rate of the core (K/s)
            dm_dt = M_St / x_S * dxl_dTl * dTC_dt  # Mass flux of solidified iron (kg/s)
            dxS_dt = - x_S / M_WM * dm_dt  # Change in the sulfur content of the well-mixed region

            if not stop_growth:
                dHSt_dt = - (dxS_dt * dTl_dxl * (1 / (rho_WM * g_St * self.pb.Lambda))).to(
                    u.km / u.s)  # Growth rate of the stable layer
            else:
                dHSt_dt = 0 * u.km / u.s

            # Compositional buoyancy flux
            F_x = (g_St / (4 * np.pi * R_WM ** 2) * (rho_Fe - rho_WM) / (rho_WM * rho_Fe) * dm_dt).to(
                u.m ** 2 / u.s ** 3)  # Compositional buoyancy flux

            # Thermal buoyancy flux
            Q_ad = (4 * np.pi * R_WM ** 2 * self.pb.k_C * alpha_WM * T_St * g_St / cp_WM).to(
                u.GW)  # Adiabatic heat flow
            F_th = ((alpha_WM * g_St) / (rho_WM * cp_WM) * (Q_C - Q_ad) / (4 * np.pi * R_WM ** 2)).to(
                u.m ** 2 / u.s ** 3)  # Thermal buoyancy flux

            F_T = F_x + F_th  # Total buoyancy flux
            U = (1.3 * (R_WM / (self.pb.Omega / u.rad)) ** (1 / 5) * F_T ** (2 / 5)).to(
                u.m / u.s)  # Typical flow velocity (m/s)

            Re_m = (U * R_WM / self.pb.etaM_C).si

            # Magnetic dipole moment
            M_dip = (4 * np.pi * R_WM ** 3 * self.pb.gamma_C * (rho_WM / self.pb.mu0_C) ** (1 / 2) * (F_T * R_WM) ** (
                        1 / 3)).to(u.A * u.m ** 2)  # dipole moment (A m2)
            B_S = ((self.pb.mu0_C * M_dip) / (4 * np.pi * self.pb.R_I ** 3)).to(
                u.uT)  # magnetic field on the surface at the equator (uT)

            if B_S.value < 0:
                B_S = 0 * u.uT

            if Re_m.value < self.Rem_C:  # ***
                B_S = 0 * u.uT

            return Q_ad, Re_m, B_S, dxS_dt, dHSt_dt, dm_dt, F_x, F_th, F_T, M_WM

        T_liq_mask = np.where(self.pb.T_mat[0, :] < self.pb.T_liq_CMB)[0]

        Q_ad_snow_arr = []
        Re_m_snow_arr = []
        B_S_snow_arr = []
        HSt_arr = []
        xS_arr = []
        dm_dt_arr = []

        F_x_arr = []
        F_th_arr = []
        F_T_arr = []

        Q_ad_arr = self.pb.Q_ad_arr.copy()
        Re_m_arr = self.pb.Re_m_arr.copy()
        B_S_arr = self.pb.B_S_arr.copy()

        if T_liq_mask.size != 0:
            self.pb.iron_snow = True
            self.pb.crystal_start_time = self.pb.t_arr.to(u.Gyr).value[T_liq_mask[0]]
            H_St = 1 * u.km
            x_S = self.pb.wtS_C / 100
            stop_growth = False

            for i in T_liq_mask:
                Q_ad_snow, Re_m_snow, B_S_snow, dxS_dt_snow, dHSt_dt_snow, dm_dt_snow, F_x, F_th, F_T, M_WM = iron_snow(
                    self.pb.Q_C_arr[i], x_S, H_St, self.pb.T_liq_CMB, stop_growth)

                Q_ad_snow_arr.append(Q_ad_snow)
                Re_m_snow_arr.append(Re_m_snow)
                B_S_snow_arr.append(B_S_snow)
                HSt_arr.append(H_St.copy())
                xS_arr.append(x_S)
                dm_dt_arr.append(dm_dt_snow)

                F_x_arr.append(F_x)
                F_th_arr.append(F_th)
                F_T_arr.append(F_T)

                delta_t = self.pb.t_arr[i] - self.pb.t_arr[i - 1]
                x_S = x_S + dxS_dt_snow * delta_t
                H_St = H_St + dHSt_dt_snow * delta_t
                if H_St > self.pb.R_C or M_WM.value < 0:
                    H_St = self.pb.R_C.copy()
                    stop_growth = True


            Q_ad_snow_arr = u.Quantity(Q_ad_snow_arr)
            Re_m_snow_arr = u.Quantity(Re_m_snow_arr)
            B_S_snow_arr = u.Quantity(B_S_snow_arr)
            HSt_arr = u.Quantity(HSt_arr)

            Q_ad_arr[T_liq_mask] = Q_ad_snow_arr
            Re_m_arr[T_liq_mask] = Re_m_snow_arr
            B_S_arr[T_liq_mask] = B_S_snow_arr

            self.pb.F_x_arr[T_liq_mask] = F_x_arr
            self.pb.F_th_arr[T_liq_mask] = F_th_arr
            self.pb.F_T_arr[T_liq_mask] = F_T_arr

            Re_m_arr = np.nan_to_num(Re_m_arr)
            B_S_arr = np.nan_to_num(B_S_arr)

        self.pb.T_liq_mask, self.pb.Q_ad_arr, self.pb.Re_m_arr, self.pb.B_S_arr, self.pb.HSt_arr, self.pb.xS_arr = T_liq_mask, u.Quantity(
            Q_ad_arr), u.Quantity(Re_m_arr), u.Quantity(B_S_arr), u.Quantity(HSt_arr), u.Quantity(xS_arr)
