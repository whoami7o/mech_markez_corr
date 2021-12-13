import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from shapely.geometry import LineString
from unifloc.pvt.fluid_flow import FluidFlow


class MechMarquezCorr:

    @staticmethod
    def v_S(qi: float, r_c: float, r_p: float) -> float:
        return qi / (math.pi * (r_c ** 2 - r_p ** 2))

    @staticmethod
    def v_s(dens_l: float, dens_g: float, sigma_l: float, g=9.81) -> float:
        # slip or bubble rise velocity
        return 1.53 * (g * sigma_l * (dens_l - dens_g) / (dens_l ** 2)) ** (1 / 4)

    @staticmethod
    def v_s_inf(dens_l: float, dens_g: float, sigma_l: float, flow_type: str, g=9.81) -> float or None:
        # slip or bubble rise velocity
        if flow_type in ('bubble', 'disp_bubble'):
            return 1.53 * (g * sigma_l * (dens_l - dens_g) / (dens_l ** 2)) ** (1 / 4)
        elif flow_type == 'slug':
            return 2 ** (1 / 2) * (g * sigma_l * (dens_l - dens_g) / (dens_l ** 2)) ** (1 / 4)
        else:
            return None

    @staticmethod
    def calc_r_d(vi_zsg: float, v_zinf: float, unit='si') -> float or None:
        if unit == 'si':
            return 3 / (71.617 * (1 - math.exp(-2.368 * vi_zsg))) * vi_zsg / (vi_zsg + v_zinf)  # 3.28084
        elif unit == 'field':
            return 3 / (5660.705 * (1 - math.exp(-2.5483248 * vi_zsg))) * vi_zsg / (vi_zsg + v_zinf)  # 3.28084
        else:
            return None

    @staticmethod
    def trans_A(v_Sl: float, v_s: float, ann_type: str) -> float or None:
        # v_Sg = f(v_Sl)
        if ann_type == 'con':
            return v_Sl / 4 + 0.20 * v_s
        elif ann_type == 'ecc':
            return v_Sl / 5.67 + 0.230 * v_s
        else:
            return None

    @staticmethod
    def re_number(dens_mix: float, v_M: float, d_h: float, mu_mix: float) -> float:
        return dens_mix * v_M * d_h / mu_mix

    @staticmethod
    def find_f_ea(k: float, eccentricity: float) -> float:
        cosh_eta_i = (k * (1 + eccentricity ** 2) + (1 - eccentricity ** 2)) / (2 * k * eccentricity)
        cosh_eta_0 = (k * (1 - eccentricity ** 2) + (1 + eccentricity ** 2)) / (2 * eccentricity)

        eta_i = math.acosh(cosh_eta_i)
        eta_0 = math.acosh(cosh_eta_0)

        inf_sum = 0
        for n in range(1, 10 ** (5)):
            try:
                sum_n = 2 * n / (math.exp(2 * n * eta_i) - math.exp(2 * n * eta_0))
            except (ZeroDivisionError, OverflowError):
                break
            else:
                inf_sum += sum_n

        A = 1 / (eta_0 - eta_i) - 2 * inf_sum

        B = 1 / 4 * (1 / (math.sinh(eta_0) ** 4) - 1 / (math.sinh(eta_i) ** 4))

        phi = (math.cosh(eta_i) / math.sinh(eta_i) - math.cosh(eta_0) / math.sinh(eta_0)) ** 2 * A + B

        f_ea = (4 * (1 - k ** 2) * (1 - k) ** 2) / (phi * (math.sinh(eta_0)) ** 4)

        return f_ea

    @staticmethod
    def find_f(r_e: float, k: float, eccentricity: float, ann_type: str) -> float:
        f_p = 16  # const. value

        if ann_type == 'con':
            f_ca = 16 * (1 - k) ** 2 / ((1 - k ** 4) / (1 - k ** 2) - (1 - k ** 2) / math.log(1 / k))

            if r_e <= 2300:  # laminar flow
                return f_ca / r_e

            else:  # turbulent flow
                func_for_f = lambda f: \
                    1 / (f * (f_p / f_ca) ** (0.45 * math.exp(-(r_e - 3000) / 10 ** (6)))) ** (1 / 2) \
                    - (4 * math.log(r_e * 1 / (f * (f_p / f_ca) ** (0.45 * math.exp(-(r_e - 3000) / 10 ** (6))))
                                    ** (1 / 2), 10) - 0.4)

                return fsolve(func_for_f, 10 ** (-4))[0]

        elif ann_type == 'ecc':
            f_ea = MechMarquezCorr.find_f_ea(k, eccentricity)

            if r_e <= 2300:  # laminar flow
                return f_ea / r_e

            else:  # turbulent flow
                func_for_f = lambda f: \
                    1 / (f * (f_p / f_ea) ** (0.45 * math.exp(-(r_e - 3000) / 10 ** (6)))) ** (1 / 2) \
                    - (4 * math.log(r_e * 1 / (f * (f_p / f_ea) ** (0.45 * math.exp(-(r_e - 3000) / 10 ** (6))))
                                    ** (1 / 2), 10) - 0.4)

                return fsolve(func_for_f, 10 ** (-4))[0]
        else:
            return 'Err'

    @staticmethod
    def trans_B(dens_l: float, dens_g: float, sigma_l: float, v_M: float, f: float, d_h: float, g=9.81) -> float:
        # v_Sg = f(v_M)
        # d_h = d_c - d_t
        A = 2 * (0.4 * sigma_l / ((dens_l - dens_g) * g)) ** (1 / 2) * (dens_l / sigma_l) ** (3 / 5) * (2 / d_h) ** (
                    2 / 5) * f ** (2 / 5)
        B = ((A) * v_M ** (6 / 5) - 0.725) / 4.15
        return v_M * (B) ** 2

    @staticmethod
    def trans_C(v_Sl: float, v_s: float) -> float:
        # v_Sg = f(v_Sl)
        return 1.083 * v_Sl + 0.52 * v_s

    @staticmethod
    def trans_D(dens_l: float, dens_g: float, sigma_l: float, g=9.81) -> float:
        return 3.1 * (g * sigma_l * (dens_l - dens_g) / (dens_l ** 2)) ** (1 / 4)

    @staticmethod
    def check_if_bubble_flow(dens_l: float, dens_g: float, sigma_l: float, d_cas: float, d_tub:float, g=9.81) -> bool:
        # checks if the bubble flow pattern region in annuli exists
        d_ep = d_cas + d_tub
        return d_ep >= 19.7 * ((dens_l - dens_g) * sigma_l / (g * dens_l ** 2)) ** (1 / 2)

    @staticmethod
    def check_if_annular_flow(v_Sg: float, dens_l: float, dens_g: float, sigma_l: float) -> bool:
        return v_Sg >= MechMarquezCorr.trans_D(dens_l, dens_g, sigma_l)

    @staticmethod
    def check_if_disp_bubble(v_Sl: float, v_Sg: float, dens_l: float, dens_g: float, sigma_l: float, dens_mix: float,
                             mu_mix: float, d_tub: float, d_cas: float, ann_type: str, eccentricity: float) -> bool:
        k = d_tub / d_cas
        d_h = d_cas - d_tub
        v_mix = v_Sl + v_Sg
        r_e = MechMarquezCorr.re_number(dens_mix, v_mix, d_h, mu_mix)
        f = MechMarquezCorr.find_f(r_e, k, eccentricity, ann_type)
        v_slip = MechMarquezCorr.v_s(dens_l, dens_g, sigma_l)

        # above C trans
        above_C = v_Sg <= MechMarquezCorr.trans_C(v_Sl, v_slip)
        # above B trans
        above_B = v_Sg <= MechMarquezCorr.trans_B(dens_l, dens_g, sigma_l, v_mix, f, d_h)
        return (above_C and above_B)


    @staticmethod
    def check_ifin_bubble(v_Sl: float, v_Sg: float, dens_l: float, dens_g: float, sigma_l: float, dens_mix: float,
                          mu_mix: float, d_tub: float, d_cas: float, ann_type: str, eccentricity: float) -> bool:
        k = d_tub / d_cas
        d_h = d_cas - d_tub
        v_mix = v_Sl + v_Sg
        r_e = MechMarquezCorr.re_number(dens_mix, v_mix, d_h, mu_mix)
        f = MechMarquezCorr.find_f(r_e, k, eccentricity, ann_type)
        v_slip = MechMarquezCorr.v_s(dens_l, dens_g, sigma_l)

        # above A trans
        above_A = v_Sg <= MechMarquezCorr.trans_A(v_Sl, v_slip, ann_type)
        # below B trans
        below_B = v_Sg >= MechMarquezCorr.trans_B(dens_l, dens_g, sigma_l, v_mix, f, d_h)
        return (above_A and below_B)

    @staticmethod
    def check_if_slug(v_Sl: float, v_Sg: float, dens_l: float, dens_g: float, sigma_l: float, dens_mix: float,
                      mu_mix: float, d_tub: float, d_cas: float, ann_type: str, eccentricity: float,
                      v_Sl_CB_inter: float) -> bool:
        k = d_tub / d_cas
        d_h = d_cas - d_tub
        v_mix = v_Sl + v_Sg
        r_e = MechMarquezCorr.re_number(dens_mix, v_mix, d_h, mu_mix)
        f = MechMarquezCorr.find_f(r_e, k, eccentricity, ann_type)
        v_slip = MechMarquezCorr.v_s(dens_l, dens_g, sigma_l)
        if v_Sl < v_Sl_CB_inter:
            below_A = v_Sg >= MechMarquezCorr.trans_A(v_Sl, v_slip, ann_type)
            below_B = v_Sg >= MechMarquezCorr.trans_B(dens_l, dens_g, sigma_l, v_mix, f, d_h)
            return (below_A and below_B)

        else:
            below_S = v_Sg >= MechMarquezCorr.trans_C(v_Sl, v_slip)
            return below_S

    @staticmethod
    def define_flow_type(v_Sl: float, v_Sg: float, dens_l: float, dens_g: float, sigma_l: float, dens_mix: float,
                         mu_mix: float, d_tub: float, d_cas: float, ann_type: str, eccentricity: float,
                         v_Sl_CB_inter: float) -> str:

        bubble_check = MechMarquezCorr.check_if_bubble_flow(dens_l, dens_g, sigma_l, d_cas, d_tub)
        annular_check = MechMarquezCorr.check_if_annular_flow(v_Sg, dens_l, dens_g, sigma_l)
        if (bubble_check == False) or (annular_check == True):
            return 'out of boundaries / annular flow'

        if MechMarquezCorr.check_if_disp_bubble(v_Sl, v_Sg, dens_l, dens_g, sigma_l, dens_mix, mu_mix, d_tub, d_cas,
                                                ann_type, eccentricity):
            return 'disp_bubble'

        if MechMarquezCorr.check_ifin_bubble(v_Sl, v_Sg, dens_l, dens_g, sigma_l, dens_mix, mu_mix, d_tub, d_cas,
                                             ann_type, eccentricity):
            return 'bubble'

        if MechMarquezCorr.check_if_slug(v_Sl, v_Sg, dens_l, dens_g, sigma_l, dens_mix, mu_mix, d_tub, d_cas,
                                         ann_type, eccentricity, v_Sl_CB_inter):
            return 'slug'

    @staticmethod
    def find_r_i(dens_l: float, dens_g: float, sigma_l: float, mu_l: float,
                 q_fluid: float, q_gas: float, flow_type: str,
                 d_cas: float, d_tub: float, h_p: float,
                 g = 9.81) -> float:
        r_c = d_cas / 2
        r_p = d_tub / 2
        # h_p = d_tub + 10 * 10 ** (-3)  # м

        v_slip = MechMarquezCorr.v_s_inf(dens_l, dens_g, sigma_l, flow_type)

        vi_zl = MechMarquezCorr.v_S(q_fluid, d_cas / 2, d_tub / 2)
        vi_zg = MechMarquezCorr.v_S(q_gas, d_cas / 2, d_tub / 2)
        tan_beta = (r_c - r_p) / h_p

        r_d = MechMarquezCorr.calc_r_d(vi_zg, v_slip)

        drdz = lambda h, r: 54 * mu_l / r_d ** 2 * 1 / ((dens_l - dens_g) * g) * (r_p + h / h_p * (r_c - r_p)) * vi_zl \
                            *tan_beta * (2 / 9 * (r_d ** 2 * dens_l) / mu_l * 1 / r ** 3 * (r_p + h / h_p * (r_c - r_p))
                                         * vi_zl * tan_beta + 1 / r)

        r_list = tuple(np.linspace(r_c, r_p, num=1000, endpoint=True))
        sol_list = []
        # slo_list_to_plot = []
        for i in r_list:
            sol1 = solve_ivp(drdz, [h_p, 0], [i])
            # sol1

            sol_list.append(sol1.y[0][-1])
            # slo_list_to_plot.append(sol1)

        diff = 10 ** (6)
        r_s = None  # separation radius
        for sol in sol_list:
            if abs(r_p - sol) < diff:
                diff = abs(r_p - sol)
                r_s = sol

        return r_list[sol_list.index(r_s)]

    @staticmethod
    def calc_e_sep(d_cas: float, d_tub: float, r_s: float) -> float:
        r_c = d_cas / 2
        r_p = d_tub / 2
        return (r_c**2 - (r_s**2)) / (r_c**2 - r_p**2)

    @staticmethod
    def calc_k_sep(q_fluid: float, q_gas: float,
                   dens_l: float, dens_g: float, sigma_l: float, mu_l: float, dens_mix: float, mu_mix: float,
                   d_tub: float, d_cas: float, h_p: float, ann_type: str, eccentricity: float,
                   wct, pvt_model_data, p, t) -> float:
        # # #
        r_c = d_cas / 2
        r_t = d_tub / 2

        d_h = d_cas - d_tub

        k = d_tub / d_cas

        v_slip = MechMarquezCorr.v_s(dens_l, dens_g, sigma_l)
        v_Sliq = MechMarquezCorr.v_S(q_fluid, r_c, r_t)
        v_Sgas = MechMarquezCorr.v_S(q_gas, r_c, r_t)
        # # #

        v_Sliq_for_A = np.arange(0, 100, 1 / 1000)
        v_Sgas_for_A = MechMarquezCorr.trans_A(v_Sliq, v_slip, ann_type)

        v_Sliq_for_B = np.arange(0.001, 100, 1 / 100)
        v_Sliq_new_for_B = []
        v_Sgas_for_B = []

        for v in v_Sliq_for_B:
            q_fluid_i = v * (math.pi * (r_c ** 2 - r_t ** 2))
            fluid_flow_i = FluidFlow(q_fluid_i, wct, pvt_model_data)
            fluid_flow_i.calc_flow(p, t)

            rho_liq_i = fluid_flow_i.rl
            rho_gas_i = fluid_flow_i.rg
            sigma_l_i = fluid_flow_i.stlg

            q_mix_i = fluid_flow_i.qm
            dens_mix_i = fluid_flow_i.rm
            mu_mix_i = fluid_flow_i.mum * 10 ** (-3)

            v_mix_i = MechMarquezCorr.v_S(q_mix_i, r_c, r_t)

            r_e_i = MechMarquezCorr.re_number(dens_mix_i, v_mix_i, d_h, mu_mix_i)

            f_i = MechMarquezCorr.find_f(r_e_i, k, eccentricity, ann_type)

            v_Sg_i = MechMarquezCorr.trans_B(rho_liq_i, rho_gas_i, sigma_l_i, v_mix_i, f_i, d_h)

            v_Sliq_new_for_B.append(v)
            v_Sgas_for_B.append(v_Sg_i)

        index_for_B = v_Sgas_for_B.index(min(v_Sgas_for_B))

        v_Sliq_new_for_B = v_Sliq_new_for_B[index_for_B:]
        v_Sgas_for_B = v_Sgas_for_B[index_for_B:]

        v_Sliq_for_C = np.arange(0, 100, 1 / 1000)
        v_Sgas_for_C = MechMarquezCorr.trans_C(v_Sliq_for_C, v_slip)

        # v_Sgas_for_D = [trans_D(rho_liq, rho_gas, sigma_l), trans_D(rho_liq, rho_gas, sigma_l)]
        # v_Sliq_for_D = [0, 100]

        line_B = LineString(np.column_stack((v_Sgas_for_B, v_Sliq_new_for_B)))
        # line_A = LineString(np.column_stack((v_Sgas_for_A, v_Sliq_for_A)))
        # intersection_AB = line_B.intersection(line_A)

        line_C = LineString(np.column_stack((v_Sgas_for_C, v_Sliq_for_C)))
        intersection_CB = line_B.intersection(line_C)

        v_Sl_CB_inter = intersection_CB.xy[1][0]

        flow_type = MechMarquezCorr.define_flow_type(v_Sliq, v_Sgas, dens_l, dens_g, sigma_l, dens_mix, mu_mix,
                                                     d_tub, d_cas, ann_type, eccentricity, v_Sl_CB_inter)

        if flow_type == 'out of boundaries / annular flow':
            return 0

        # print(f'flow type = {flow_type}')

        r_s = MechMarquezCorr.find_r_i(dens_l, dens_g, sigma_l, mu_l, q_fluid, q_gas, flow_type, d_cas, d_tub, h_p)

        return MechMarquezCorr.calc_e_sep(d_cas, d_tub, r_s)


if __name__ == '__main__':
    q_fluid = 50 / 86400
    wct = 0
    pvt_model_data = {'black_oil': {'gamma_gas': 0.7, 'gamma_wat': 1, 'gamma_oil': 0.8,
                                    'rp': 50,
                                    'oil_correlations':
                                        {'pb': 'Standing', 'rs': 'Standing',
                                         'rho': 'Standing', 'b': 'Standing',
                                         'mu': 'Beggs', 'compr': 'Vasquez'},
                                    'gas_correlations': {'ppc': 'Standing', 'tpc': 'Standing',
                                                         'z': 'Dranchuk', 'mu': 'Lee'},
                                    'water_correlations': {'b': 'McCain', 'compr': 'Kriel',
                                                           'rho': 'Standing', 'mu': 'McCain'},
                                    'rsb': {'value': 50, 'p': 10000000, 't': 303.15},
                                    'muob': {'value': 0.5, 'p': 10000000, 't': 303.15},
                                    'bob': {'value': 1.5, 'p': 10000000, 't': 303.15},
                                    'table_model_data': None, 'use_table_model': False}}

    # Инициализация исходных данных метода расчета pvt-свойств флюидов
    p = 4 * (10 ** 6)
    t = 350

    # Инициализация объекта pvt-модели
    fluid_flow = FluidFlow(q_fluid, wct, pvt_model_data)

    # Пересчет всех свойств для данного давления и температуры
    fluid_flow.calc_flow(p, t)

    e_sep = MechMarquezCorr.calc_k_sep(q_fluid=fluid_flow.ql,
                                       q_gas=fluid_flow.qg,
                                       dens_l=fluid_flow.rl,
                                       dens_g=fluid_flow.rg,
                                       sigma_l=fluid_flow.stlg,
                                       mu_l=fluid_flow.mul * 10**(-3),
                                       dens_mix=fluid_flow.rm,
                                       mu_mix=fluid_flow.mum * 10**(-3),
                                       d_tub=0.063,
                                       d_cas=0.130,
                                       h_p=0.063 + 10*10**(-3),
                                       ann_type='con',
                                       eccentricity=0.8,
                                       wct=wct,
                                       pvt_model_data=pvt_model_data,
                                       p=p,
                                       t=t)

    print(f'E_sep = {e_sep: .4f}')
