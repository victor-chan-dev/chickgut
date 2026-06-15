import math 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from chickgut.utils.performance import time_it, time_block

import jax
import jax.numpy as jnp
import diffrax
jax.config.update("jax_enable_x64", True)

# --- Pure JAX Vector Fields (Extracted from Ileum Class) ---

@jax.jit
def _method_of_lines_Ileum_CPu_jax(t, y, args):
    (Jej_UD_sol, VF_Il_CPu, IlV_cm3, Kp_Il_min, CP_jej_lastnode_vol_cm3, P_CPe_Il_gmincm3) = args
    QCPu_Il_g = y
    
    val0 = Jej_UD_sol.evaluate(t)[-1]
    QCPu_Il_g = QCPu_Il_g.at[0].set(val0)
    QCPu_Il_g = jnp.maximum(QCPu_Il_g, 1e-10)
    
    U_CPu_Il_psg_gmincm3 = VF_Il_CPu * jnp.diff(QCPu_Il_g) / jnp.diff(IlV_cm3)
    P_CPu_JejIl_gmincm3 = (Kp_Il_min * val0) / CP_jej_lastnode_vol_cm3
    
    Il_UD_Diff = P_CPe_Il_gmincm3 + P_CPu_JejIl_gmincm3 - U_CPu_Il_psg_gmincm3
    return jnp.concatenate([jnp.array([0.0]), Il_UD_Diff])

@jax.jit
def _method_of_lines_Ileum_CPu_metrics(t, y, args):
    (Jej_UD_sol, VF_Il_CPu, IlV_cm3, Kp_Il_min, CP_jej_lastnode_vol_cm3, P_CPe_Il_gmincm3, Il_single_node_V) = args
    val0 = Jej_UD_sol.evaluate(t)[-1]
    QCPu_Il_g = y.at[0].set(val0)
    QCPu_Il_g = jnp.maximum(QCPu_Il_g, 1e-10)
    
    U_CPu_Il_psg_gmincm3 = VF_Il_CPu * jnp.diff(QCPu_Il_g) / jnp.diff(IlV_cm3)
    P_CPu_JejIl_gmincm3 = (Kp_Il_min * val0) / CP_jej_lastnode_vol_cm3
    Il_UD_Diff = P_CPe_Il_gmincm3 + P_CPu_JejIl_gmincm3 - U_CPu_Il_psg_gmincm3
    dUIdt = jnp.concatenate([jnp.array([0.0]), Il_UD_Diff])
    
    flux_CPu_Ileum_psg = (QCPu_Il_g[-1] / Il_single_node_V) * Kp_Il_min
    return dUIdt, flux_CPu_Ileum_psg

@jax.jit
def _method_of_lines_Ileum_CPsl_jax(t, y, args):
    (Jej_SlD_sol, VF, IlV_cm3, Kp_Jej_min, Il_single_node_V, CP_jej_lastnode_vol_cm3, k_digestrate) = args
    QCPsl_Il_g = y
    val0 = Jej_SlD_sol.evaluate(t)[-1]
    QCPsl_Il_g = QCPsl_Il_g.at[0].set(val0)
    QCPsl_Il_g = jnp.maximum(QCPsl_Il_g, 1e-10)
    
    Il_CPsl_dis_gmincm3 = k_digestrate * (QCPsl_Il_g[1:]**2 / Il_single_node_V)
    U_CPsl_Il_psg_gmincm3 = VF * jnp.diff(QCPsl_Il_g) / jnp.diff(IlV_cm3)
    U_CPsl_Il_psg_dis_gmincm3 = U_CPsl_Il_psg_gmincm3 + Il_CPsl_dis_gmincm3
    
    P_CPsl_JejIl_gmincm3 = (Kp_Jej_min * val0) / CP_jej_lastnode_vol_cm3
    
    Il_SlD_Diff = P_CPsl_JejIl_gmincm3 - U_CPsl_Il_psg_dis_gmincm3
    return jnp.concatenate([jnp.array([0.0]), Il_SlD_Diff])

@jax.jit
def _method_of_lines_Ileum_CPsl_metrics(t, y, args):
    (Jej_SlD_sol, VF, IlV_cm3, Kp_Jej_min, Il_single_node_V, CP_jej_lastnode_vol_cm3, k_digestrate, Kp_Il_min) = args
    val0 = Jej_SlD_sol.evaluate(t)[-1]
    QCPsl_Il_g = y.at[0].set(val0)
    QCPsl_Il_g = jnp.maximum(QCPsl_Il_g, 1e-10)
    
    Il_CPsl_dis_gmincm3 = k_digestrate * (QCPsl_Il_g[1:]**2 / Il_single_node_V)
    U_CPsl_Il_psg_gmincm3 = VF * jnp.diff(QCPsl_Il_g) / jnp.diff(IlV_cm3)
    U_CPsl_Il_psg_dis_gmincm3 = U_CPsl_Il_psg_gmincm3 + Il_CPsl_dis_gmincm3
    
    P_CPsl_JejIl_gmincm3 = (Kp_Jej_min * val0) / CP_jej_lastnode_vol_cm3
    Il_SlD_Diff = P_CPsl_JejIl_gmincm3 - U_CPsl_Il_psg_dis_gmincm3
    dSlIdt = jnp.concatenate([jnp.array([0.0]), Il_SlD_Diff])
    
    flux_CPsl_Ileum_psg_dis = Kp_Il_min * (QCPsl_Il_g[-1] / Il_single_node_V)
    return dSlIdt, flux_CPsl_Ileum_psg_dis

@jax.jit
def _method_of_lines_CPr_Il_jax(t, y, args):
    (Jej_RD_sol, Il_CPsl_sol, VF, IlV_cm3, Kp_Jej_min, Il_single_node_V, k_absp, k_digestrate) = args
    QCPr_Il_g = y
    val0 = Jej_RD_sol.evaluate(t)[-1]
    QCPr_Il_g = QCPr_Il_g.at[0].set(val0)
    QCPr_Il_g = jnp.maximum(QCPr_Il_g, 1e-12)
    
    Il_CPr_dis_gmincm3 = (k_absp * QCPr_Il_g[1:]**2) / Il_single_node_V
    U_CPr_Il_psg_gmincm3 = VF * jnp.diff(QCPr_Il_g) / jnp.diff(IlV_cm3)
    U_CPr_Il_psg_dis_gmincm3 = jnp.maximum(U_CPr_Il_psg_gmincm3 + Il_CPr_dis_gmincm3, 1e-12)
    
    P_CPr_JejIl_gmincm3 = (Kp_Jej_min * val0) / Il_single_node_V
    
    Il_CPsl_g_last = Il_CPsl_sol.evaluate(t)[-1]
    P_CPr_IlSlIlR_gmincm3 = (k_digestrate * Il_CPsl_g_last) / Il_single_node_V
    
    Il_RD_Diff = P_CPr_JejIl_gmincm3 + P_CPr_IlSlIlR_gmincm3 - jnp.minimum(U_CPr_Il_psg_dis_gmincm3, QCPr_Il_g[1:] / Il_single_node_V)
    return jnp.concatenate([jnp.array([0.0]), Il_RD_Diff])

@jax.jit
def _method_of_lines_CPr_Il_metrics(t, y, args):
    (Jej_RD_sol, Il_CPsl_sol, VF, IlV_cm3, Kp_Jej_min, Il_single_node_V, k_absp, k_digestrate, Kp_Il_min) = args
    val0 = Jej_RD_sol.evaluate(t)[-1]
    QCPr_Il_g = y.at[0].set(val0)
    QCPr_Il_g = jnp.maximum(QCPr_Il_g, 1e-12)
    
    Il_CPr_dis_gmincm3 = (k_absp * QCPr_Il_g[1:]**2) / Il_single_node_V
    U_CPr_Il_psg_gmincm3 = VF * jnp.diff(QCPr_Il_g) / jnp.diff(IlV_cm3)
    U_CPr_Il_psg_dis_gmincm3 = jnp.maximum(U_CPr_Il_psg_gmincm3 + Il_CPr_dis_gmincm3, 1e-12)
    
    P_CPr_JejIl_gmincm3 = (Kp_Jej_min * val0) / Il_single_node_V
    
    Il_CPsl_g_last = Il_CPsl_sol.evaluate(t)[-1]
    P_CPr_IlSlIlR_gmincm3 = (k_digestrate * Il_CPsl_g_last) / Il_single_node_V
    
    Il_RD_Diff = P_CPr_JejIl_gmincm3 + P_CPr_IlSlIlR_gmincm3 - jnp.minimum(U_CPr_Il_psg_dis_gmincm3, QCPr_Il_g[1:] / Il_single_node_V)
    dRIdt = jnp.concatenate([jnp.array([0.0]), Il_RD_Diff])
    
    flux_CPr_Il_psg_dis = (QCPr_Il_g[-1] / Il_single_node_V) * Kp_Il_min
    return dRIdt, flux_CPr_Il_psg_dis

@jax.jit
def _feed_il_jax(t, y, args):
    Jej_feed_sol, Kp_Jej_min, Kp_Il_min = args
    Qfeed_Il_g = y[0]
    Qfeed_Il_g = jnp.maximum(Qfeed_Il_g, 1e-10)
    
    Q_feedJej_g_at_t = Jej_feed_sol.evaluate(t)[0]
    
    P_Il_feed = Q_feedJej_g_at_t * Kp_Jej_min
    U_IlCo_feed = Qfeed_Il_g * Kp_Il_min
    return jnp.array([P_Il_feed - U_IlCo_feed])

@jax.jit
def _feed_il_metrics(t, y, args):
    Jej_feed_sol, Kp_Jej_min, Kp_Il_min = args
    Qfeed_Il_g = jnp.maximum(y[0], 1e-10)
    
    P_Il_feed = Jej_feed_sol.evaluate(t)[0] * Kp_Jej_min
    U_IlCo_feed = Qfeed_Il_g * Kp_Il_min
    dfeedildt = P_Il_feed - U_IlCo_feed
    return dfeedildt, U_IlCo_feed, Qfeed_Il_g
class Ileum():
    
    def __init__(self, t_span, t_eval, constants, BWeight_kgb, k_absp, k_digestrate, Kp_endog_min):

        self.name = "ileum"
        self.t_span = t_span
        self.t_eval = t_eval

        self.init_Il_CPu = None
        self.init_Il_CPsl = None
        self.init_Il_CPr = None

        self.df_UP_i = None
        self.df_SlP_i = None
        self.df_RP_i = None
        self.df_feed_i = None

        self.length_cm = None
        self.r_cm = None
        self.volume_jej_cm3 = None
        self.volume_cm3 = None
        self.volume_jej_cm3 = None
        self.total_discretize = None
        self.total_node_num = None
        
        self.constants = constants
        self.BWeight_kgb = BWeight_kgb
        self.Discretize_jej = None
        self.k_absp = k_absp
        self.k_digestrate = k_digestrate
        self.Kp_endog_min = Kp_endog_min

        self.UJexit_SS = None
        self.SlJexit_SS = None
        self.RJexit_SS = None
        self.result_feed_jej = None

        self.UIexit_SS = None
        self.SlIexit_SS = None
        self.RIexit_SS = None
        self.result_feed_il = None

        self.k_absp = k_absp
    
    def calculate_V(self, duodenum_instance, jejunum_instance):
        self.volume_cm3 = duodenum_instance.volume_cm3
        self.volume_jej_cm3 = jejunum_instance.volume_jej_cm3
        self.Discretize_jej = jejunum_instance.Discretize_jej

    def calculate_Il_prop(self):
        self.Discretize_il = 101    
        self.Node_num = self.Discretize_il - 1
        self.length_cm = 34.643*self.BWeight_kgb 
        self.r_cm = 0.9/2                   
        self.volume_il_cm3 = math.pi*math.pow(self.r_cm, 2)*self.length_cm 
        
        self.IlV_cm3 = np.linspace(0, self.volume_il_cm3, self.Discretize_il)
        self.Il_single_node_V = self.volume_il_cm3 / self.Node_num
        self.MRT_min = self.constants['Il_MRT'] 
        MRT_Il_CPu_min = 49.5 

        self.Kp_Il_min =  1/self.MRT_min 
        self.Kp_Il_CPu_min = 1/MRT_Il_CPu_min 
        
        self.VF = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/self.MRT_min 
        self.VF_Il_CPu = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/MRT_Il_CPu_min
        self.init_Il_CPu = np.zeros(self.Discretize_il) 
        self.init_Il_CPsl = np.zeros(self.Discretize_il)
        self.init_Il_CPr = np.zeros(self.Discretize_il)
                

    def calculate_Il_endog(self):
        SI_volume_cm3 = self.volume_cm3 + self.volume_jej_cm3 + self.volume_il_cm3
        self.vol_prop = self.volume_il_cm3 / SI_volume_cm3
        
        Il_BasalAA = 0.799 
        self.basalaa = Il_BasalAA * self.vol_prop 

        Ileum_basalaa_single_node = self.basalaa/(self.Discretize_il - 1) 
        P_CPe_Il_gmin = self.Kp_endog_min*Ileum_basalaa_single_node
        self.P_CPe_Il_gmincm3 = P_CPe_Il_gmin/self.Il_single_node_V
        
        self.CP_jej_lastnode_vol_cm3 = self.volume_jej_cm3/self.Discretize_jej

    def Jej_results(self, jejunum_instance):
        self.jejunum = jejunum_instance
        self.UJexit_SS = jejunum_instance.UJexit_SS
        self.SlJexit_SS = jejunum_instance.SlJexit_SS
        self.RJexit_SS = jejunum_instance.RJexit_SS
        self.Jej_single_node_V = jejunum_instance.Jej_single_node_V
        self.Kp_Jej_min = jejunum_instance.Kp_Jej_min
        self.result_feed_jej = jejunum_instance.result_feed_jej

    @time_it
    def solving_il_USl(self):
        y0_u = jnp.zeros(len(self.init_Il_CPu)).at[0].set(self.UJexit_SS.ys[-1, -1] / self.Il_single_node_V)
        y0_sl = jnp.zeros(len(self.init_Il_CPsl)).at[0].set(self.SlJexit_SS.ys[-1, -1] / self.Il_single_node_V)
        
        args_u = (self.UJexit_SS, self.VF_Il_CPu, jnp.array(self.IlV_cm3), self.Kp_Il_min, self.CP_jej_lastnode_vol_cm3, self.P_CPe_Il_gmincm3)
        args_sl = (self.SlJexit_SS, self.VF, jnp.array(self.IlV_cm3), self.Kp_Jej_min, self.Il_single_node_V, self.CP_jej_lastnode_vol_cm3, self.k_digestrate)
        
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.UIexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_Ileum_CPu_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_u, args=args_u, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000, adjoint=diffrax.DirectAdjoint()
        )
        
        self.SlIexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_Ileum_CPsl_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_sl, args=args_sl, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000, adjoint=diffrax.DirectAdjoint()
        )

    def flatten_result_il_CPu(self):
        args_u = (self.UJexit_SS, self.VF_Il_CPu, jnp.array(self.IlV_cm3), self.Kp_Il_min, self.CP_jej_lastnode_vol_cm3, self.P_CPe_Il_gmincm3, self.Il_single_node_V)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_Ileum_CPu_metrics(t, y, args_u)
            
        v_compute = jax.vmap(compute_metrics)
        dUIdt_all, flux_all = v_compute(self.UIexit_SS.ts, self.UIexit_SS.ys)
        
        self.df_UP_i = {
            't': self.UIexit_SS.ts,
            'dUIdt': dUIdt_all[:, -1],
            'flux_CPu_Ileum_psg': flux_all,
            'CPu_il': self.UIexit_SS.ys[:, -1]
        }
        return self.df_UP_i, None
    
    def flatten_result_il_CPsl(self):
        args_sl = (self.SlJexit_SS, self.VF, jnp.array(self.IlV_cm3), self.Kp_Jej_min, self.Il_single_node_V, self.CP_jej_lastnode_vol_cm3, self.k_digestrate, self.Kp_Il_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_Ileum_CPsl_metrics(t, y, args_sl)
            
        v_compute = jax.vmap(compute_metrics)
        dSlIdt_all, flux_all = v_compute(self.SlIexit_SS.ts, self.SlIexit_SS.ys)
        
        self.df_SlP_i = {
            't': self.SlIexit_SS.ts,
            'dSlIdt': dSlIdt_all[:, -1],
            'flux_CPsl_Ileum_psg_dis': flux_all,
            'CPsl_il': self.SlIexit_SS.ys[:, -1]
        }
        return self.df_SlP_i, None

    @time_it
    def SlP_for_RP_i(self):
        self.flatten_result_il_CPsl()
        Il_CPsl_Q = self.df_SlP_i['CPsl_il']
        return None, Il_CPsl_Q
    
    @time_it
    def solving_il_R(self):

        y0_r = jnp.zeros(len(self.init_Il_CPr)).at[0].set(self.RJexit_SS.ys[-1, -1] / self.Il_single_node_V)
        args_r = (self.RJexit_SS, self.SlIexit_SS, self.VF, jnp.array(self.IlV_cm3), self.Kp_Jej_min, self.Il_single_node_V, self.k_absp, self.k_digestrate)
        
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.RIexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_CPr_Il_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_r, args=args_r, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000, adjoint=diffrax.DirectAdjoint()
        )
    
    def flatten_result_il_CPr(self):
        args_r = (self.RJexit_SS, self.SlIexit_SS, self.VF, jnp.array(self.IlV_cm3), self.Kp_Jej_min, self.Il_single_node_V, self.k_absp, self.k_digestrate, self.Kp_Il_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_CPr_Il_metrics(t, y, args_r)
            
        v_compute = jax.vmap(compute_metrics)
        dRIdt_all, flux_all = v_compute(self.RIexit_SS.ts, self.RIexit_SS.ys)
        
        self.df_RP_i = {
            't': self.RIexit_SS.ts,
            'dRIdt': dRIdt_all[:, -1],
            'flux_CPr_Ileum_psg_dis': flux_all,
            'CPr_il': self.RIexit_SS.ys[:, -1]
        }
        return self.df_RP_i, None
    
    @time_it
    def solving_il_feed(self):
        self.iIl_g = [0.00000000001]
        y0 = jnp.array(self.iIl_g)
        args_feed = (self.result_feed_jej, self.Kp_Jej_min, self.Kp_Il_min)
        
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.result_feed_il = diffrax.diffeqsolve(
            diffrax.ODETerm(_feed_il_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_feed, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000, adjoint=diffrax.DirectAdjoint()
        )
    
    def flatten_result_il_feed(self):
        args_feed = (self.result_feed_jej, self.Kp_Jej_min, self.Kp_Il_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _feed_il_metrics(t, y, args_feed)
            
        v_compute = jax.vmap(compute_metrics)
        dfeedildt_all, flux_all, Q_all = v_compute(self.result_feed_il.ts, self.result_feed_il.ys)
        
        self.df_feed_i = {
            't': self.result_feed_il.ts,
            'dfeedildt': dfeedildt_all,
            'flux_feed_IlCo': flux_all,
            'Qfeed_il': self.result_feed_il.ys[:, 0]
        }
        return self.df_feed_i, None 

    def get_flux_sums(self, limit=2001):
        # Calculate flux from the last node of each fraction
        flux_u = (self.UIexit_SS.ys[:, -1] / self.Il_single_node_V) * self.Kp_Il_min
        flux_sl = (self.SlIexit_SS.ys[:, -1] / self.Il_single_node_V) * self.Kp_Il_min
        flux_r = (self.RIexit_SS.ys[:, -1] / self.Il_single_node_V) * self.Kp_Il_min
        
        sum_UI_flux = jnp.sum(flux_u[:limit])
        sum_SlI_flux = jnp.sum(flux_sl[:limit])
        sum_RI_flux = jnp.sum(flux_r[:limit])
        
        return sum_UI_flux, sum_SlI_flux, sum_RI_flux
