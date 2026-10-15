import numpy as np
from typing import Tuple
from astropy import units as u
from scipy.interpolate import interp1d
from astropy.constants import G
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy


def EOS_Morard(wt_S: float, P: float, T: float) -> Tuple[
    u.Quantity, float, u.Quantity, u.Quantity]:  # From Joe O'Rourke
    # Constants
    mm_Fe = 55.845  # molar mass of iron
    mm_S = 32.065  # molar mass of sulfur

    # Calculate atomic fraction of S in the alloy
    X_S = (wt_S / mm_S) / ((100 - wt_S) / mm_Fe + wt_S / mm_S)

    # Density at ambient pressure and 1900 K
    rho0 = -3108 * X_S ** 2 - 5176 * X_S + 6950

    # Ratio of rho to rho0
    rhorat = np.linspace(0.9, 1.35, 1001)
    rho_all = rho0 * rhorat

    # Bulk modulus
    K_TFe = 76e9
    K_TS = 1.6e9
    K_T0 = K_TFe ** (1 - X_S) * K_TS ** X_S

    # Bulk modulus derivative
    K1_Fe = 6.5
    K1 = K1_Fe + 3 * X_S

    # Change of density with temperature
    drhodT_Fe0 = -0.926
    drhodT_FeS0 = -0.533
    drhodT_0 = drhodT_Fe0 - (drhodT_Fe0 - drhodT_FeS0) * X_S

    # Coefficient of thermal expansion at ambient pressure
    alpha0 = -1 / rho0 * drhodT_0

    # Bulk modulus at a given pressure
    K_T = 0.5 * K_T0 * (7 * rhorat ** (7 / 3) - 5 * rhorat ** (5 / 3)) * (
                1 + 0.75 * (K1 - 4) * (rhorat ** (2 / 3) - 1)) + 0.75 * K_T0 * (K1 - 4) * (
                      rhorat ** 3 - rhorat ** (7 / 3))

    # Thermal expansivity
    alpha = alpha0 * K_T0 / K_T

    # Pressure
    P_1900 = 1.5 * K_T0 * (rhorat ** (7 / 3) - rhorat ** (5 / 3)) * (1 + 0.75 * (K1 - 4) * (rhorat ** (2 / 3) - 1))
    P_th = (T - 1900) * alpha * K_T
    P_all = P_th + P_1900

    # Interpolation to find density associated with the input pressure
    rho_interp = interp1d(P_all, rho_all, fill_value="extrapolate")
    rho = rho_interp(P)

    # Gruneisen parameter
    gamm = 1.52 + 2.5 * X_S

    # Sound speed
    Vp_all = np.sqrt(K_T * (1 + alpha * gamm * T) / rho_all)

    # Specific heat
    Cp_all = alpha * Vp_all ** 2 / gamm
    Cp_interp = interp1d(P_all, Cp_all, fill_value="extrapolate")
    Cp = Cp_interp(P)

    # Thermal expansion
    alpha_interp = interp1d(P_all, alpha, fill_value="extrapolate")
    alpha_out = alpha_interp(P)

    return rho * u.kg / u.m ** 3, gamm, Cp * u.J / u.kg / u.K, alpha_out / u.K


def liquidus_BW2011(wt_S: float, P_GPa: float) -> float:  # From Joe O'Rourke
    # Constants
    mm_Fe = 55.845  # molar mass of iron
    mm_S = 32.065  # molar mass of sulfur

    # Calculate atomic fraction of S in the alloy
    X_S = (wt_S / mm_S) / ((100 - wt_S) / mm_Fe + wt_S / mm_S)

    # Calcualte atomic fraction of FeS in the alloy
    X_FeS = X_S / (1 - X_S)

    A = -2.4724 * P_GPa ** 4 + 28.025 * P_GPa ** 3 + 9.1404 * P_GPa ** 2 + 581.71 * P_GPa + 3394.8
    B = 1.7978 * P_GPa ** 4 - 6.7881 * P_GPa ** 3 - 197.69 * P_GPa ** 2 - 271.69 * P_GPa - 8219.5
    C = -0.1702 * P_GPa ** 4 - 9.3959 * P_GPa ** 3 + 163.53 * P_GPa ** 2 - 319.35 * P_GPa + 5698.6
    D = -0.2308 * P_GPa ** 4 + 7.1 * P_GPa ** 3 - 64.118 * P_GPa ** 2 + 105.98 * P_GPa - 1621.9
    E = 0.2302 * P_GPa ** 4 - 5.3688 * P_GPa ** 3 + 38.124 * P_GPa ** 2 - 46.681 * P_GPa + 1813.8

    T = A * X_FeS ** 4 + B * X_FeS ** 3 + C * X_FeS ** 2 + D * X_FeS + E

    return T


def gravity(pb, R: u.Quantity) -> u.Quantity:  # acceleration due to gravity at given R (m/s^2)
    def mass(mass_R):  # mass within triton at given radius R (kg)
        m = 0 * u.kg

        if mass_R >= pb.R_C:
            m += pb.M_C
            if mass_R >= pb.R_M:
                m += pb.M_M
                if mass_R >= pb.R_O:
                    m += pb.M_O
                    if mass_R >= pb.R_I:
                        m += pb.M_I
                    else:  # if R_O < R < R_I
                        m += 4 / 3 * np.pi * (mass_R ** 3 - pb.R_O ** 3) * pb.rho_I
                else:  # if R_M < R < R_O
                    m += 4 / 3 * np.pi * (mass_R ** 3 - pb.R_M ** 3) * pb.rho_O
            else:  # if R_C < R < R_M
                m += 4 / 3 * np.pi * (mass_R ** 3 - pb.R_C ** 3) * pb.rho_M
        else:  # if R < R_C
            m += 4 / 3 * np.pi * mass_R ** 3 * pb.rho_C

        return m

    return (G * mass(R) / R ** 2).si


def latent(xS_unscaled):
    xS_FeS = 32.06 / (55.845 + 32.06)  # m(S) / (m(Fe) + m(S))
    xS_scaled = xS_unscaled / xS_FeS
    L_Fe = 261.20 * u.kJ / u.kg  # LL chondrite, table 4 of https://onlinelibrary.wiley.com/doi/10.1111/maps.12280
    L_FeS = 357.91 * u.kJ / u.kg  # LL chondrite, table 4 of https://onlinelibrary.wiley.com/doi/10.1111/maps.12280
    return L_Fe + xS_scaled * (L_FeS - L_Fe)


def pressure(r_unscaled, r_C, p_CEN, p_CMB):  # hydrostatic pressure inside the core
    # For uniform density, g = g_CMB r / R_C, so dP/dr = -rho g gives P quadratic in r.
    r_scaled = r_unscaled / r_C
    return p_CEN + r_scaled ** 2 * (p_CMB - p_CEN)


def x_eutectic(p):
    return u.Quantity(0.11 + 0.187 * np.exp(-0.065 * 10 ** -9 * p.to(u.Pa).value))


def plot(model, show_plots: bool = False, out_folder: str = '', model_name: str = ''):  # Adapted from Kevin Trinh
    plt.close('all')

    color_ice = '#D3D3D3'
    color_ocean = '#20B2AA'
    color_mantle = '#FFA07A'
    color_core = '#F08080'
    color_core2 = '#EC5F5F'
    color_core3 = '#E84141'

    # FIGURE 1
    fig_R, ax_R = plt.subplots(figsize=(12, 6))

    ax_R.axhspan(model.pb.R_O.to(u.km).value, model.pb.R_I.to(u.km).value, color=color_ice, zorder=1)
    ax_R.axhspan(model.pb.R_M.to(u.km).value, model.pb.R_O.to(u.km).value, color=color_ocean, zorder=1)
    ax_R.axhspan(model.pb.R_C.to(u.km).value, model.pb.R_M.to(u.km).value, color=color_mantle, zorder=1)
    ax_R.axhspan(0, model.pb.R_C.to(u.km).value, color=color_core, zorder=1)

    t_arr = model.pb.t_arr.to(u.Gyr)

    conv_C_mask = model.pb.F_T_arr.to(u.m ** 2 / u.s ** 3).value < 0
    true_indices = np.where(conv_C_mask)[0]
    if len(true_indices) > 0:
        end_index = np.where(np.diff(true_indices) > 1)[0]
        first_conv_C_mask = true_indices[:end_index[0] + 1] if len(end_index) > 0 else true_indices

        ax_R.fill_between(t_arr.value[first_conv_C_mask], 0, model.pb.R_C.to(u.km).value, facecolor='none',
                          edgecolor=color_core3, linewidth=0, hatch='////', zorder=2)
        conv_x1 = (min(t_arr.value[first_conv_C_mask]) + max(t_arr.value[first_conv_C_mask])) / 2
        ax_R.text(conv_x1, model.pb.R_C.to(u.km).value * 1 / 2, 'No Convection', color='black', ha='center',
                  va='center', rotation='vertical', zorder=4)

        if len(end_index) > 0 and end_index[0] + 1 < len(true_indices):
            second_start = end_index[0] + 1
            second_conv_C_mask = true_indices[second_start:end_index[1] + 1] if len(end_index) > 1 else true_indices[
                                                                                                        second_start:]

            ax_R.fill_between(t_arr.value[second_conv_C_mask], 0, model.pb.R_C.to(u.km).value, facecolor='none',
                              edgecolor=color_core3, linewidth=0, hatch='////', zorder=2)
            conv_x2 = (min(t_arr.value[second_conv_C_mask]) + max(t_arr.value[second_conv_C_mask])) / 2
            ax_R.text(conv_x2, model.pb.R_C.to(u.km).value * 1 / 2, 'No Convection', color='black', ha='center',
                      va='center', rotation='vertical', zorder=4)

    if model.pb.inner_core:
        ax_R.fill_between(t_arr.value, 0, model.pb.R_IC_arr.to(u.km).value, facecolor=color_core2, zorder=3)
        inner_core_x_index = np.argmax(model.pb.R_IC_arr.to(u.km).value > 0)
        inner_core_x = (t_arr.value[inner_core_x_index] + t_arr.value[-1]) / 2
        inner_core_y_index = int((inner_core_x_index + len(model.pb.R_IC_arr)) / 2)
        inner_core_y = model.pb.R_IC_arr.to(u.km).value[inner_core_y_index] / 2
        ax_R.text(inner_core_x, inner_core_y, 'Inner Core', color='black', ha='center', va='center_baseline', zorder=4)

    elif model.pb.FeS_layer:
        ax_R.fill_between(t_arr.value, model.pb.R_IC_arr.to(u.km).value, model.pb.R_C.to(u.km).value,
                          facecolor=color_core2, zorder=3)
        FeS_layer_x_index = np.argmax(model.pb.R_IC_arr.to(u.km).value < model.pb.R_C.to(u.km).value)
        FeS_layer_x = (t_arr.value[FeS_layer_x_index] + t_arr.value[-1]) / 2
        FeS_layer_y_index = int((FeS_layer_x_index + len(model.pb.R_IC_arr)) / 2)
        FeS_layer_y = (model.pb.R_IC_arr.to(u.km).value[FeS_layer_y_index] + model.pb.R_C.to(u.km).value) / 2
        ax_R.text(FeS_layer_x, FeS_layer_y, 'FeS Layer', color='black', ha='center', va='center_baseline', zorder=4)

    elif model.pb.iron_snow:
        ax_R.fill_between(t_arr.value[model.pb.T_liq_mask],
                          model.pb.R_C.to(u.km).value - model.pb.HSt_arr.to(u.km).value, model.pb.R_C.to(u.km).value,
                          facecolor=color_core2, zorder=3)
        stable_x = (min(t_arr.value[model.pb.T_liq_mask]) + max(t_arr.value[model.pb.T_liq_mask])) / 2
        ax_R.text(stable_x, model.pb.R_C.to(u.km).value / 2, 'Iron Snow', color='black', ha='center',
                  va='center_baseline', zorder=4)

    X, Y = np.meshgrid(t_arr, model.pb.R_arr.to(u.km)[1:])
    Z = np.array(np.array(model.pb.T_mat[1:, :]).T) * u.K
    T_contours = np.arange(500, 2500, 250) * u.K

    CS = ax_R.contour(X.value, Y.value, Z.T.value, T_contours.value, colors='black', linewidths=1, linestyles='dashed',
                      zorder=4)
    ax_R.clabel(CS, inline=True, fontsize=8, fmt=lambda val: f'{int(val)} K')
    ax_R.set_xlabel('Time (Gyr)')
    ax_R.set_ylabel('Radial Distance (km)')
    ax_R.set_xlim([model.t_start.to(u.Gyr).value, model.t_end.to(u.Gyr).value])
    ax_R.set_ylim([0, model.pb.R_I.to(u.km).value])

    if show_plots:
        plt.show()
        fig_R.savefig(f'fig1_rename.png', dpi=300, bbox_inches='tight')

    if out_folder != '':
        fig_R.savefig(f'{out_folder}/{model_name}_fig1.png', dpi=300, bbox_inches='tight')

    # FIGURE 2
    lw = 3
    fig_combined, ((ax_Q, ax_T), (ax_Rem, ax_B)) = plt.subplots(2, 2, figsize=(12, 8))

    ax_Q.plot(t_arr.value, model.pb.Q_C_arr.to(u.GW).value, color=color_core, label=r'$\mathregular{Q_C}$',
              linewidth=lw)
    ax_Q.plot(t_arr.value, model.pb.Q_ad_arr.to(u.GW).value, color='black', label=r'$\mathregular{Q_{ad}}$',
              linewidth=lw)
    ax_Q.axhline(0, color='black', linewidth=1, linestyle='dashed')
    ax_Q.set_ylabel('Heat Flow (GW)')
    ax_Q.set_xlim([model.t_start.to(u.Gyr).value, model.t_end.to(u.Gyr).value])
    ax_Q.set_ylim(-10, 15)
    ax_Q.set_xticks(np.arange(model.t_start.to(u.Gyr).value, model.t_end.to(u.Gyr).value + 1, 1))
    ax_Q.set_xlabel('Time (Gyr)')

    T_cmb = model.pb.T_mat[0, :]
    T_mid = model.pb.T_mat[round(len(model.pb.R_arr) / 2), :]
    ax_T.plot(t_arr.value, T_cmb.value, label='CMB', color=color_core, linewidth=lw)
    ax_T.plot(t_arr.value, T_mid.value, label='Mid-mantle', color=color_mantle, linewidth=lw)
    ax_T.set_ylabel('Temperature (K)')
    ax_T.set_xlim([model.t_start.to(u.Gyr).value, model.t_end.to(u.Gyr).value])
    ax_T.set_ylim([300, 2200])
    ax_T.set_xticks(np.arange(model.t_start.to(u.Gyr).value, model.t_end.to(u.Gyr).value + 1, 1))
    ax_T.set_xlabel('Time (Gyr)')

    ax_Rem.plot(t_arr.value, model.pb.Re_m_arr.value, color=color_ocean, linewidth=lw)
    ax_Rem.axhline(model.Rem_C, color='black', linewidth=1, linestyle='dashed')
    ax_Rem.set_ylabel('Magnetic Reynolds Number')
    ax_Rem.set_xlim([model.t_start.to(u.Gyr).value, model.t_end.to(u.Gyr).value])
    ax_Rem.set_ylim(0, 350)
    ax_Rem.set_yticks(sorted(set(range(0, 301, 100)).union([model.Rem_C])),
                      labels=sorted(set(range(0, 301, 100)).union([model.Rem_C])))
    ax_Rem.set_xticks(np.arange(model.t_start.to(u.Gyr).value, model.t_end.to(u.Gyr).value + 1, 1))
    ax_Rem.set_xlabel('Time (Gyr)')

    ax_B.plot(t_arr.value, model.pb.B_S_arr.to(u.uT).value, color=color_ocean, linewidth=lw)
    ax_B.set_ylabel('Surface Magnetic Field ($\mathregular{\mu}$T)')
    ax_B.set_xlim([model.t_start.to(u.Gyr).value, model.t_end.to(u.Gyr).value])
    ax_B.set_ylim(0, 3.75)
    ax_B.set_yticks(sorted(set(range(0, 4, 1))), labels=sorted(set(range(0, 4, 1))))
    ax_B.set_xticks(np.arange(model.t_start.to(u.Gyr).value, model.t_end.to(u.Gyr).value + 1, 1))
    ax_B.set_xlabel('Time (Gyr)')

    if show_plots:
        plt.show()
        fig_combined.savefig(f'fig2_rename.png', dpi=300, bbox_inches='tight')

    if out_folder != '':
        fig_combined.savefig(f'{out_folder}/{model_name}_fig2.png', dpi=300, bbox_inches='tight')

