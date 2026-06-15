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

# --- Pure JAX Vector Fields (Extracted from Jejunum Class) ---
# Why JAX? We use JAX (specifically @jax.jit) to compile these mathematical ODE equations into 
# highly optimized machine code. Since these are called millions of times by the solver, 
# stripping them out of the Python class structure and making them "pure" functions allows 
# the simulation to run ~30x faster.

@jax.jit
def _method_of_lines_Jej_CPu_jax(t, y, args):
    (Duo_UD_sol, VF_CPu, JejV_cm3, Kp_Jej_min, Duo_single_node_V, P_CPe_Jej_gmincm3) = args
    QCPu_Jej_g = y
    
    val0 = Duo_UD_sol.evaluate(t)[-1]
    QCPu_Jej_g = QCPu_Jej_g.at[0].set(val0)
    QCPu_Jej_g = jnp.maximum(QCPu_Jej_g, 1e-10)
    
    U_CPu_Jej_psg_gmincm3 = VF_CPu * jnp.diff(QCPu_Jej_g) / jnp.diff(JejV_cm3)
    
    P_CPu_DuoJej_gmincm3 = (Kp_Jej_min * val0) / Duo_single_node_V
    Jej_UD_Diff = P_CPe_Jej_gmincm3 + P_CPu_DuoJej_gmincm3 - U_CPu_Jej_psg_gmincm3
    return jnp.concatenate([jnp.array([0.0]), Jej_UD_Diff])

@jax.jit
def _method_of_lines_Jej_CPu_metrics(t, y, args):
    (Duo_UD_sol, VF_CPu, JejV_cm3, Kp_Jej_min, Duo_single_node_V, P_CPe_Jej_gmincm3, Kp_Jej_CPu_min, Jej_single_node_V) = args
    val0 = Duo_UD_sol.evaluate(t)[-1]
    QCPu_Jej_g = y.at[0].set(val0)
    QCPu_Jej_g = jnp.maximum(QCPu_Jej_g, 1e-10)
    
    U_CPu_Jej_psg_gmincm3 = VF_CPu * jnp.diff(QCPu_Jej_g) / jnp.diff(JejV_cm3)
    P_CPu_DuoJej_gmincm3 = (Kp_Jej_min * val0) / Duo_single_node_V
    Jej_UD_Diff = P_CPe_Jej_gmincm3 + P_CPu_DuoJej_gmincm3 - U_CPu_Jej_psg_gmincm3
    dUJdt = jnp.concatenate([jnp.array([0.0]), Jej_UD_Diff])
    
    flux_CPu_Jej_psg_dis = (QCPu_Jej_g[-1] / Jej_single_node_V) * Kp_Jej_CPu_min
    return dUJdt, flux_CPu_Jej_psg_dis

@jax.jit
def _method_of_lines_Jej_CPsl_jax(t, y, args):
    (Duo_SlD_sol, VF, JejV_cm3, Kp_Duo_min, Jej_single_node_V, k_digestrate) = args
    QCPsl_Jej_g = y
    val0 = Duo_SlD_sol.evaluate(t)[-1]
    QCPsl_Jej_g = QCPsl_Jej_g.at[0].set(val0)
    QCPsl_Jej_g = jnp.maximum(QCPsl_Jej_g, 1e-10)
    
    Jej_CPsl_dis_gmincm3 = (k_digestrate * QCPsl_Jej_g[1:]**2) / Jej_single_node_V
    U_CPsl_Jej_psg_min = VF * jnp.diff(QCPsl_Jej_g) / jnp.diff(JejV_cm3)
    U_CPsl_Jej_psg_dis_gmincm3 = U_CPsl_Jej_psg_min + Jej_CPsl_dis_gmincm3
    
    P_CPsl_DuoJej_gmincm3 = (Kp_Duo_min * val0) / Jej_single_node_V
    
    Jej_SlD_Diff = P_CPsl_DuoJej_gmincm3 - U_CPsl_Jej_psg_dis_gmincm3
    return jnp.concatenate([jnp.array([0.0]), Jej_SlD_Diff])

@jax.jit
def _method_of_lines_Jej_CPsl_metrics(t, y, args):
    (Duo_SlD_sol, VF, JejV_cm3, Kp_Duo_min, Jej_single_node_V, k_digestrate, Kp_Jej_min) = args
    val0 = Duo_SlD_sol.evaluate(t)[-1]
    QCPsl_Jej_g = y.at[0].set(val0)
    QCPsl_Jej_g = jnp.maximum(QCPsl_Jej_g, 1e-10)
    
    Jej_CPsl_dis_gmincm3 = (k_digestrate * QCPsl_Jej_g[1:]**2) / Jej_single_node_V
    U_CPsl_Jej_psg_min = VF * jnp.diff(QCPsl_Jej_g) / jnp.diff(JejV_cm3)
    U_CPsl_Jej_psg_dis_gmincm3 = U_CPsl_Jej_psg_min + Jej_CPsl_dis_gmincm3
    
    P_CPsl_DuoJej_gmincm3 = (Kp_Duo_min * val0) / Jej_single_node_V
    Jej_SlD_Diff = P_CPsl_DuoJej_gmincm3 - U_CPsl_Jej_psg_dis_gmincm3
    dSlJdt = jnp.concatenate([jnp.array([0.0]), Jej_SlD_Diff])
    
    flux_CPsl_Jej_psg_dis = (QCPsl_Jej_g[-1] / Jej_single_node_V) * Kp_Jej_min
    return dSlJdt, flux_CPsl_Jej_psg_dis

@jax.jit
def _method_of_lines_CPr_Jej_jax(t, y, args):
    (Duo_RD_sol, Jej_CPsl_sol, VF, JejV_cm3, Kp_Duo_min, Duo_single_node_V, Jej_single_node_V, k_absp, k_digestrate) = args
    QCPr_Jej_g = y
    val0 = Duo_RD_sol.evaluate(t)[-1]
    QCPr_Jej_g = QCPr_Jej_g.at[0].set(val0)
    QCPr_Jej_g = jnp.maximum(QCPr_Jej_g, 1e-12)
    
    Jej_CPr_dis_gmincm3 = k_absp * (QCPr_Jej_g[1:]**2 / Jej_single_node_V)
    U_CPr_Jej_psg_mincm3 = VF * jnp.diff(QCPr_Jej_g) / jnp.diff(JejV_cm3)
    U_CPr_Jej_psg_dis_gmincm3 = jnp.maximum(U_CPr_Jej_psg_mincm3 + Jej_CPr_dis_gmincm3, 1e-12)
    
    P_CPr_DuoJej_gmincm3 = (Kp_Duo_min * val0) / Duo_single_node_V
    
    Jej_CPsl_g_last = Jej_CPsl_sol.evaluate(t)[-1]
    P_CPr_JejDJejS_gmincm3 = (k_digestrate * Jej_CPsl_g_last) / Jej_single_node_V
    
    Jej_RD_Diff = P_CPr_DuoJej_gmincm3 + P_CPr_JejDJejS_gmincm3 - jnp.minimum(U_CPr_Jej_psg_dis_gmincm3, QCPr_Jej_g[1:] / Jej_single_node_V)
    return jnp.concatenate([jnp.array([0.0]), Jej_RD_Diff])

@jax.jit
def _method_of_lines_CPr_Jej_metrics(t, y, args):
    (Duo_RD_sol, Jej_CPsl_sol, VF, JejV_cm3, Kp_Duo_min, Duo_single_node_V, Jej_single_node_V, k_absp, k_digestrate, Kp_Jej_min) = args
    val0 = Duo_RD_sol.evaluate(t)[-1]
    QCPr_Jej_g = y.at[0].set(val0)
    QCPr_Jej_g = jnp.maximum(QCPr_Jej_g, 1e-12)
    
    Jej_CPr_dis_gmincm3 = k_absp * (QCPr_Jej_g[1:]**2 / Jej_single_node_V)
    U_CPr_Jej_psg_mincm3 = VF * jnp.diff(QCPr_Jej_g) / jnp.diff(JejV_cm3)
    U_CPr_Jej_psg_dis_gmincm3 = jnp.maximum(U_CPr_Jej_psg_mincm3 + Jej_CPr_dis_gmincm3, 1e-12)
    
    P_CPr_DuoJej_gmincm3 = (Kp_Duo_min * val0) / Duo_single_node_V
    
    Jej_CPsl_g_last = Jej_CPsl_sol.evaluate(t)[-1]
    P_CPr_JejDJejS_gmincm3 = (k_digestrate * Jej_CPsl_g_last) / Jej_single_node_V
    
    Jej_RD_Diff = P_CPr_DuoJej_gmincm3 + P_CPr_JejDJejS_gmincm3 - jnp.minimum(U_CPr_Jej_psg_dis_gmincm3, QCPr_Jej_g[1:] / Jej_single_node_V)
    dRJdt = jnp.concatenate([jnp.array([0.0]), Jej_RD_Diff])
    
    flux_CPr_Jej_psg_dis = (QCPr_Jej_g[-1] / Jej_single_node_V) * Kp_Jej_min
    return dRJdt, flux_CPr_Jej_psg_dis

@jax.jit
def _feed_jej_jax(t, y, args):
    Duo_feed_sol, Kp_Duo_min, Kp_Jej_min = args
    Qfeed_Jej_g = y[0]
    
    # CRITICAL MATH FIX: We must clamp the feed amount to a tiny positive number (1e-10).
    Qfeed_Jej_g = jnp.maximum(Qfeed_Jej_g, 1e-10)
    
    Q_feedDuo_g_at_t = Duo_feed_sol.evaluate(t)[0]
    
    P_Jej_feed = Q_feedDuo_g_at_t * Kp_Duo_min
    U_JejIl_feed = Qfeed_Jej_g * Kp_Jej_min
    return jnp.array([P_Jej_feed - U_JejIl_feed])

@jax.jit
def _feed_jej_metrics(t, y, args):
    Duo_feed_sol, Kp_Duo_min, Kp_Jej_min = args
    Qfeed_Jej_g = jnp.maximum(y[0], 1e-10)
    
    P_Jej_feed = Duo_feed_sol.evaluate(t)[0] * Kp_Duo_min
    U_JejIl_feed = Qfeed_Jej_g * Kp_Jej_min
    dfeedjejdt = P_Jej_feed - U_JejIl_feed
    return dfeedjejdt, U_JejIl_feed, Qfeed_Jej_g

class Jejunum():

    def __init__(self, t_span,  t_eval, constants, BWeight_kgb, k_absp, k_digestrate, Kp_endog_min):

        self.name = "jejunum"
        self.t_span = t_span
        self.t_eval = t_eval
        
        self.init_Jej_CPu = None
        self.init_Jej_CPsl = None
        self.init_Jej_CPr = None       
        
        self.df_UP_j = None
        self.df_SlP_j = None
        self.df_RP_j = None
        self.df_feed_j = None

        self.length_cm = None
        self.r_cm = None
        self.volume_jej_cm3 = None
        self.volume_cm3 = None
        self.volume_il_cm3 = None
        self.total_discretize = None
        self.total_node_num = None

        self.constants = constants
        self.BWeight_kgb = BWeight_kgb
        self.k_absp = k_absp
        self.k_digestrate = k_digestrate
        self.Kp_endog_min = Kp_endog_min

        self.UDexit_SS = None
        self.SlDexit_SS = None
        self.RDexit_SS = None
        self.result_feed_duo = None
        self.Duo_single_node_V = None
        self.Kp_Duo_min = None

        self.UJexit_SS = None
        self.SlJexit_SS = None
        self.RJexit_SS = None
        self.result_feed_jej = None

    def calculate_V(self, duodenum_instance, ileum_instance):
        self.volume_cm3 = duodenum_instance.volume_cm3
        self.volume_il_cm3 = ileum_instance.volume_il_cm3

    def calculate_jej_prop(self):
        self.length_cm = 33.166*self.BWeight_kgb  
        self.r_cm = 1.14/2                   
        self.volume_jej_cm3 = math.pi*math.pow(self.r_cm, 2)*self.length_cm  

        self.Discretize_jej = 101    
        self.Node_num_jej = self.Discretize_jej - 1
        self.JejV_cm3 = np.linspace(0, self.volume_jej_cm3, self.Discretize_jej)
        self.Jej_single_node_V = self.volume_jej_cm3/self.Node_num_jej

        MRT_Jej_CPu_min = 16.0 
        self.MRT_Jej_min = self.constants['Jej_MRT']

        self.Kp_Jej_min = 1/self.MRT_Jej_min 
        self.Kp_Jej_CPu_min = 1/MRT_Jej_CPu_min 
        
        self.VF = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/self.MRT_Jej_min 
        self.VF_CPu = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/MRT_Jej_CPu_min 
        self.init_Jej_CPu = np.zeros(self.Discretize_jej) 
        self.init_Jej_CPsl = np.zeros(self.Discretize_jej)
        self.init_Jej_CPr = np.zeros(self.Discretize_jej)


    def calculate_jej_endog(self):
        SI_volume_cm3 = self.volume_cm3 + self.volume_jej_cm3 + self.volume_il_cm3
    
        self.vol_prop = self.volume_jej_cm3 / SI_volume_cm3
        
        Il_BasalAA = 0.799 
        self.basalaa = Il_BasalAA * self.vol_prop 

        self.basalaa_single_node = self.basalaa/(self.Discretize_jej - 1)  
        P_CPe_Jej_gmin = self.Kp_endog_min*self.basalaa_single_node 
        self.P_CPe_Jej_gmincm3 = P_CPe_Jej_gmin/self.Jej_single_node_V
        

    def Duo_results(self, duodenum_instance):
        self.duodenum = duodenum_instance
        self.UDexit_SS = duodenum_instance.UDexit_SS
        self.SlDexit_SS = duodenum_instance.SlDexit_SS
        self.RDexit_SS = duodenum_instance.RDexit_SS
        self.Duo_single_node_V = duodenum_instance.Duo_single_node_V
        self.Kp_Duo_min = duodenum_instance.Kp_Duo_min
        self.result_feed_duo = duodenum_instance.result_feed_duo
       

    @time_it
    def solving_jej_USl(self):
        # Diffrax .ys has shape (num_time_steps, num_variables)
        y0_u = jnp.zeros(len(self.init_Jej_CPu)).at[0].set(self.UDexit_SS.ys[-1, -1] / self.Jej_single_node_V)
        y0_sl = jnp.zeros(len(self.init_Jej_CPsl)).at[0].set(self.SlDexit_SS.ys[-1, -1] / self.Jej_single_node_V)
        
        args_u = (self.UDexit_SS, self.VF_CPu, jnp.array(self.JejV_cm3), self.Kp_Jej_min, self.Duo_single_node_V, self.P_CPe_Jej_gmincm3)
        args_sl = (self.SlDexit_SS, self.VF, jnp.array(self.JejV_cm3), self.Kp_Duo_min, self.Jej_single_node_V, self.k_digestrate)
        
        # Why Diffrax & Kvaerno5? 
        # The biological equations here are "stiff" (they change very rapidly, especially absorption).
        # Standard solvers (like Tsit5 or standard SciPy) struggle to keep up and take tiny steps, 
        # making the simulation crawl. Kvaerno5 is an "implicit" solver designed specifically 
        # to power through these stiff, rapid biological spikes smoothly and quickly.
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.UJexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_Jej_CPu_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_u, args=args_u, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000, adjoint=diffrax.DirectAdjoint()
        )
        
        self.SlJexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_Jej_CPsl_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_sl, args=args_sl, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000, adjoint=diffrax.DirectAdjoint()
        )
        
    def flatten_result_jej_CPu(self):
        args_u = (self.UDexit_SS, self.VF_CPu, jnp.array(self.JejV_cm3), self.Kp_Jej_min, self.Duo_single_node_V, self.P_CPe_Jej_gmincm3, self.Kp_Jej_CPu_min, self.Jej_single_node_V)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_Jej_CPu_metrics(t, y, args_u)
            
        # Why jax.vmap?
        # Instead of using a slow Python "for" loop to calculate the metrics (flux, concentrations)
        # for each of the 2,800 time steps one by one, vmap "vectorizes" the operation.
        # It calculates all time steps simultaneously using the compiled code, saving massive time.
        v_compute = jax.vmap(compute_metrics)
        dUJdt_all, flux_all = v_compute(self.UJexit_SS.ts, jnp.maximum(self.UJexit_SS.ys, 0.0))
        
        self.df_UP_j = {
            't': self.UJexit_SS.ts,
            'dUJdt': dUJdt_all[:, -1],
            'flux_CPu_Jej_psg': flux_all,
            'CPu_jej': jnp.maximum(self.UJexit_SS.ys[:, -1], 0.0)
        }
        return self.df_UP_j, None

    def flatten_result_jej_CPsl(self):
        args_sl = (self.SlDexit_SS, self.VF, jnp.array(self.JejV_cm3), self.Kp_Duo_min, self.Jej_single_node_V, self.k_digestrate, self.Kp_Jej_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_Jej_CPsl_metrics(t, y, args_sl)
            
        v_compute = jax.vmap(compute_metrics)
        dSlJdt_all, flux_all = v_compute(self.SlJexit_SS.ts, jnp.maximum(self.SlJexit_SS.ys, 0.0))
        
        self.df_SlP_j = {
            't': self.SlJexit_SS.ts,
            'dSlJdt': dSlJdt_all[:, -1],
            'flux_CPsl_Jej_psg_dis': flux_all,
            'CPsl_jej': jnp.maximum(self.SlJexit_SS.ys[:, -1], 0.0)
        }
        return self.df_SlP_j, None

    @time_it
    def SlP_for_RP_j(self):
        self.flatten_result_jej_CPsl()
        Jej_CPsl_Q = self.df_SlP_j['CPsl_jej']
        return None, Jej_CPsl_Q
    
    @time_it
    def solving_jej_R(self):
        self.df_Q_CPsl_Jej, self.Jej_CPsl = self.SlP_for_RP_j()
        y0_r = jnp.zeros(len(self.init_Jej_CPr)).at[0].set(self.RDexit_SS.ys[-1, -1] / self.Jej_single_node_V)
        args_r = (self.RDexit_SS, self.SlJexit_SS, self.VF, jnp.array(self.JejV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.Jej_single_node_V, self.k_absp, self.k_digestrate)
        
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.RJexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_CPr_Jej_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_r, args=args_r, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000, adjoint=diffrax.DirectAdjoint()
        )

    def flatten_result_jej_CPr(self):
        args_r = (self.RDexit_SS, self.SlJexit_SS, self.VF, jnp.array(self.JejV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.Jej_single_node_V, self.k_absp, self.k_digestrate, self.Kp_Jej_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_CPr_Jej_metrics(t, y, args_r)
            
        v_compute = jax.vmap(compute_metrics)
        dRJdt_all, flux_all = v_compute(self.RJexit_SS.ts, jnp.maximum(self.RJexit_SS.ys, 0.0))
        
        self.df_RP_j = {
            't': self.RJexit_SS.ts,
            'dRJdt': dRJdt_all[:, -1],
            'flux_CPr_Jej_psg_dis': flux_all,
            'CPr_jej': jnp.maximum(self.RJexit_SS.ys[:, -1], 0.0)
        }
        return self.df_RP_j, None
    
    @time_it
    def solving_jej_feed(self):
        self.iJej_g = [0.00000000001]
        y0 = jnp.array(self.iJej_g)
        args_feed = (self.result_feed_duo, self.Kp_Duo_min, self.Kp_Jej_min)
        
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.result_feed_jej = diffrax.diffeqsolve(
            diffrax.ODETerm(_feed_jej_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_feed, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000, adjoint=diffrax.DirectAdjoint()
        )
    
    def flatten_result_jej_feed(self):
        args_feed = (self.result_feed_duo, self.Kp_Duo_min, self.Kp_Jej_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _feed_jej_metrics(t, y, args_feed)
            
        v_compute = jax.vmap(compute_metrics)
        dfeedjejdt_all, flux_all, Q_all = v_compute(self.result_feed_jej.ts, jnp.maximum(self.result_feed_jej.ys, 0.0))
        
        self.df_feed_j = {
            't': self.result_feed_jej.ts,
            'dfeedjejdt': dfeedjejdt_all,
            'flux_feed_JejIl': flux_all,
            'Qfeed_jej': jnp.maximum(self.result_feed_jej.ys[:, 0], 0.0)
        }
        return self.df_feed_j, None 

    def plot_jej(self):
        
        plt.clf() 
        plt.plot(self.df_UP_j['t'], self.df_UP_j['CPu_jej'], color= 'blue') #, label='undegradable_jej g'
        plt.plot(self.df_SlP_j['t'], self.df_SlP_j['CPsl_jej'], color= 'red')#, label='slowly-digestible_jej g'
        plt.plot(self.df_RP_j['t'], self.df_RP_j['CPr_jej'], color= 'green')#, label='rapidly-digestible_jej g'
        #plt.legend()
        plt.tight_layout
        plt.xlabel('Time')
        plt.ylabel('Protein (g) in the jejunum')
        #plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
        plt.savefig('images/B-protein-over-time-at-last-node-of-jejunum.png', bbox_inches='tight')
        
        plt.clf() 
        plt.plot(self.df_UP_j['t'], self.df_UP_j['flux_CPu_Jej_psg'], color='blue')#, label='Undegradable Protein_jejunum flux'
        plt.plot(self.df_SlP_j['t'], self.df_SlP_j['flux_CPsl_Jej_psg_dis'], color='red')#, label='Slowly-digestible Protein_jejunum flux'
        plt.plot(self.df_RP_j['t'], self.df_RP_j['flux_CPr_Jej_psg_dis'], color='green')#, label='Rapidly-digestible Protein_jejunum flux'
        #plt.legend()
        plt.tight_layout
        plt.xlabel('Time')
        plt.ylabel('Flux of protein (g/min) at the end of the jejunum')
        #plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
        plt.savefig('images/B-protein-over-time-at-last-node-of-jejunum_flux.png', bbox_inches='tight')
        
        plt.clf() 
        plt.plot(self.df_feed_j['t'], self.df_feed_j['Qfeed_jej'], color='red', label='Q feed jejunum')
        plt.plot(self.df_feed_j['t'], self.df_feed_j['flux_feed_JejIl'], color='blue', label='Flux feed jejunum')
        #plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
        plt.xlabel('Time')
        plt.ylabel('Feed (g) and Flux (g/min) in the jejunum')
        plt.tight_layout(pad=2.0)
        plt.savefig('images/B-feed-over-time-jejunum.png', bbox_inches='tight')
    
