import math 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from chickgut.utils.performance import time_it, time_block

import jax
import jax.numpy as jnp
import diffrax
# JAX defaults to 32-bit floats. We need 64-bit for biological stability.
jax.config.update("jax_enable_x64", True)

# --- Pure JAX Vector Fields (Extracted from Duodenum Class) ---
# Why JAX? We use JAX (specifically @jax.jit) to compile these mathematical ODE equations into 
# highly optimized machine code. Since these are called millions of times by the solver, 
# stripping them out of the Python class structure and making them "pure" functions allows 
# the simulation to run ~30x faster.

@jax.jit
def _solve_node0_jax(t, y, args):
    fore_t, fore_y, constant_Fr, Kp_PVG_min, Kp_Duo_min = args
    QCPu_Duonode0_g = y[0]
    index_PVGDuo = jnp.minimum(jnp.searchsorted(fore_t, t, side='left'), len(fore_t) - 1)
    Q_CP_g_at_t = fore_y[index_PVGDuo, 1]
    P_CP_PVGDuo_gmin = Kp_PVG_min * (Q_CP_g_at_t * constant_Fr)
    U_CP_Duonode0Duo_gmin = Kp_Duo_min * QCPu_Duonode0_g
    return jnp.array([P_CP_PVGDuo_gmin - U_CP_Duonode0Duo_gmin])

@jax.jit
def _method_of_lines_CPu_Duo_jax(t, y, args):
    (node0_sol, VF, DuoV_cm3, Kp_Duo_min, Duo_single_node_V, P_CPe_gmincm3) = args
    QCPu_Duo_g = y
    val0 = node0_sol.evaluate(t)[0]
    QCPu_Duo_g = QCPu_Duo_g.at[0].set(val0)
    QCPu_Duo_g = jnp.maximum(QCPu_Duo_g, 1e-10)
    
    U_CPu_Duo_psg_gmincm3 = VF * jnp.diff(QCPu_Duo_g) / jnp.diff(DuoV_cm3)
    P_CPu_Duo0Duo_gmincm3 = (Kp_Duo_min * val0) / Duo_single_node_V
    UD_Diff = P_CPe_gmincm3 + P_CPu_Duo0Duo_gmincm3 - U_CPu_Duo_psg_gmincm3
    return jnp.concatenate([jnp.array([0.0]), UD_Diff])

@jax.jit
def _method_of_lines_CPu_metrics(t, y, args):
    (node0_sol, VF, DuoV_cm3, Kp_Duo_min, Duo_single_node_V, P_CPe_gmincm3) = args
    val0 = node0_sol.evaluate(t)[0]
    QCPu_Duo_g = y.at[0].set(val0)
    QCPu_Duo_g = jnp.maximum(QCPu_Duo_g, 1e-10)
    U_CPu_Duo_psg_gmincm3 = VF * jnp.diff(QCPu_Duo_g) / jnp.diff(DuoV_cm3)
    P_CPu_Duo0Duo_gmincm3 = (Kp_Duo_min * val0) / Duo_single_node_V
    UD_Diff = P_CPe_gmincm3 + P_CPu_Duo0Duo_gmincm3 - U_CPu_Duo_psg_gmincm3
    dUDdt = jnp.concatenate([jnp.array([0.0]), UD_Diff])
    
    flux_CPu_DuoJej = QCPu_Duo_g[-1] * Kp_Duo_min
    C_CPu_Duolast_gcm3 = QCPu_Duo_g[-1] / Duo_single_node_V
    return dUDdt, flux_CPu_DuoJej, C_CPu_Duolast_gcm3

@jax.jit
def _method_of_lines_CPsl_Duo_jax(t, y, args):
    (node0_sol, VF, DuoV_cm3, Kp_Duo_min, Duo_single_node_V, k_digestrate) = args
    QCPsl_Duo_g = y
    val0 = node0_sol.evaluate(t)[0]
    QCPsl_Duo_g = QCPsl_Duo_g.at[0].set(val0)
    QCPsl_Duo_g = jnp.maximum(QCPsl_Duo_g, 1e-10)
    
    Duo_CPsl_dis_gmincm3 = (k_digestrate * QCPsl_Duo_g[1:]**2) / Duo_single_node_V
    U_CPsl_Duo_psg_gmincm3 = VF * jnp.diff(QCPsl_Duo_g) / jnp.diff(DuoV_cm3)
    P_CPsl_Duo0Duo_gmincm3 = (Kp_Duo_min * val0) / Duo_single_node_V
    
    SlD_Diff = P_CPsl_Duo0Duo_gmincm3 - (U_CPsl_Duo_psg_gmincm3 + Duo_CPsl_dis_gmincm3)
    return jnp.concatenate([jnp.array([0.0]), SlD_Diff])

@jax.jit
def _method_of_lines_CPsl_metrics(t, y, args):
    (node0_sol, VF, DuoV_cm3, Kp_Duo_min, Duo_single_node_V, k_digestrate) = args
    val0 = node0_sol.evaluate(t)[0]
    QCPsl_Duo_g = y.at[0].set(val0)
    QCPsl_Duo_g = jnp.maximum(QCPsl_Duo_g, 1e-10)
    
    Duo_CPsl_dis_gmincm3 = (k_digestrate * QCPsl_Duo_g[1:]**2) / Duo_single_node_V
    U_CPsl_Duo_psg_gmincm3 = VF * jnp.diff(QCPsl_Duo_g) / jnp.diff(DuoV_cm3)
    P_CPsl_Duo0Duo_gmincm3 = (Kp_Duo_min * val0) / Duo_single_node_V
    
    SlD_Diff = P_CPsl_Duo0Duo_gmincm3 - (U_CPsl_Duo_psg_gmincm3 + Duo_CPsl_dis_gmincm3)
    dSlDdt = jnp.concatenate([jnp.array([0.0]), SlD_Diff])
    
    flux_CPsl_DuoJej = QCPsl_Duo_g[-1] * Kp_Duo_min
    C_CPsl_Duolast_gcm3 = QCPsl_Duo_g[-1] / Duo_single_node_V
    return dSlDdt, flux_CPsl_DuoJej, C_CPsl_Duolast_gcm3

@jax.jit
def _method_of_lines_CPr_Duo_jax(t, y, args):
    (node0_sol, CPsl_sol, VF, DuoV_cm3, Kp_Duo_min, Duo_single_node_V, k_absp, k_digestrate) = args
    QCPr_Duo_g = y
    val0 = node0_sol.evaluate(t)[0]
    QCPr_Duo_g = QCPr_Duo_g.at[0].set(val0)
    
    # CRITICAL MATH FIX: We must clamp the protein amount to a tiny positive number (1e-12).
    # In nature, protein can't be negative. But numerical ODE solvers might slightly overshoot 0.0.
    # Because the digestion equation squares this value (QCPr**2), a slight negative number 
    # would suddenly become a massive positive digestion rate, causing a "positive feedback loop" 
    # that crashes the simulation. Clamping prevents this physical impossibility!
    QCPr_Duo_g = jnp.maximum(QCPr_Duo_g, 1e-12)

    
    Duo_CPr_dis = (k_absp * QCPr_Duo_g[1:]**2) / Duo_single_node_V
    U_CPr_Duo_psg_mincm3 = VF * jnp.diff(QCPr_Duo_g) / jnp.diff(DuoV_cm3)
    U_CPr_Duo_psg_dis_gmincm3 = jnp.maximum(U_CPr_Duo_psg_mincm3 + Duo_CPr_dis, 1e-12)
    
    P_CPr_Duo0Duo_gmincm3 = (Kp_Duo_min * val0) / Duo_single_node_V
    
    QCPsl_Duo_g_last = CPsl_sol.evaluate(t)[-1]
    P_CPr_DuoSlDuoR_gmincm3 = (k_digestrate * QCPsl_Duo_g_last) / Duo_single_node_V
    
    RD_Diff = P_CPr_DuoSlDuoR_gmincm3 + P_CPr_Duo0Duo_gmincm3 - U_CPr_Duo_psg_dis_gmincm3
    return jnp.concatenate([jnp.array([0.0]), RD_Diff])

@jax.jit
def _method_of_lines_CPr_metrics(t, y, args):
    (node0_sol, CPsl_sol, VF, DuoV_cm3, Kp_Duo_min, Duo_single_node_V, k_absp, k_digestrate) = args
    QCPr_Duo_g = y.at[0].set(node0_sol.evaluate(t)[0])
    QCPr_Duo_g = jnp.maximum(QCPr_Duo_g, 1e-12)
    
    Duo_CPr_dis = (k_absp * QCPr_Duo_g[1:]**2) / Duo_single_node_V
    U_CPr_Duo_psg_mincm3 = VF * jnp.diff(QCPr_Duo_g) / jnp.diff(DuoV_cm3)
    U_CPr_Duo_psg_dis_gmincm3 = jnp.maximum(U_CPr_Duo_psg_mincm3 + Duo_CPr_dis, 1e-12)
    
    P_CPr_Duo0Duo_gmincm3 = (Kp_Duo_min * node0_sol.evaluate(t)[0]) / Duo_single_node_V
    
    QCPsl_Duo_g_last = CPsl_sol.evaluate(t)[-1]
    P_CPr_DuoSlDuoR_gmincm3 = (k_digestrate * QCPsl_Duo_g_last) / Duo_single_node_V
    
    RD_Diff = P_CPr_DuoSlDuoR_gmincm3 + P_CPr_Duo0Duo_gmincm3 - U_CPr_Duo_psg_dis_gmincm3
    dRDdt = jnp.concatenate([jnp.array([0.0]), RD_Diff])
    
    flux_CPr_DuoJej = QCPr_Duo_g[-1] * Kp_Duo_min
    C_CPr__Duolast_gcm3 = QCPr_Duo_g[-1] / Duo_single_node_V
    return dRDdt, flux_CPr_DuoJej, C_CPr__Duolast_gcm3

@jax.jit
def _feed_duo_jax(t, y, args):
    fore_t, fore_y, Kp_PVG_min, Kp_Duo_min = args
    Qfeed_Duo_g = y[0]
    Qfeed_Duo_g = jnp.maximum(Qfeed_Duo_g, 1e-10)
    
    index_PVGDuo = jnp.minimum(jnp.searchsorted(fore_t, t, side='left'), len(fore_t) - 1)
    Q_feedPVG_g_at_t = fore_y[index_PVGDuo, -1]
    
    P_PVG_feed = Q_feedPVG_g_at_t * Kp_PVG_min
    U_DuoJej_feed = Qfeed_Duo_g * Kp_Duo_min
    return jnp.array([P_PVG_feed - U_DuoJej_feed])

@jax.jit
def _feed_duo_metrics(t, y, args):
    fore_t, fore_y, Kp_PVG_min, Kp_Duo_min = args
    Qfeed_Duo_g = jnp.maximum(y[0], 1e-10)
    index_PVGDuo = jnp.minimum(jnp.searchsorted(fore_t, t, side='left'), len(fore_t) - 1)
    P_PVG_feed = fore_y[index_PVGDuo, -1] * Kp_PVG_min
    U_DuoJej_feed = Qfeed_Duo_g * Kp_Duo_min
    dfeedduodt = P_PVG_feed - U_DuoJej_feed
    return dfeedduodt, U_DuoJej_feed, Qfeed_Duo_g

class Duodenum():
        
    def __init__(self, t_span, init_Duo_CPu, t_eval, fore_t, fore_y, BWeight_kgb, Kp_PVG_min, constants, k_absp, k_digestrate, Kp_endog_min):
        self.name = "duodenum"
        self.t_span = t_span
        self.t_eval = t_eval
        self.fore_t = fore_t
        self.fore_y = fore_y
        self.iDuo_g = init_Duo_CPu
        self.init_Duo_CPu = None
        self.init_Duo_CPsl = None
        self.init_Duo_CPr = None

        self.df_UP_d = None
        self.df_SlP_d = None
        self.df_RP_d = None
        self.df_feed_d = None

        self.length_cm = None
        self.r_cm = None
        self.volume_cm3 = None
        self.volume_jej_cm3 = None
        self.volume_il_cm3 = None
        self.total_discretize = None
        self.total_node_num = None

        self.Kp_PVG_min = Kp_PVG_min
        self.constants = constants
        self.BWeight_kgb = BWeight_kgb
        self.k_absp = k_absp
        self.k_digestrate = k_digestrate
        self.Kp_endog_min = Kp_endog_min

        self.result_UDnode0 = None
        self.result_SlDnode0 = None
        self.result_RDnode0 = None
        self.UDexit_SS = None
        self.SlDexit_SS = None
        self.RDexit_SS = None
        self.result_feed_duo = None
        
        self.k_absp = k_absp
    
    def calculate_V(self, jejunum_instance, ileum_instance):
        self.volume_jej_cm3 = jejunum_instance.volume_jej_cm3
        self.volume_il_cm3 = ileum_instance.volume_il_cm3

    def calculate_duo_prop(self):
        self.length_cm = 14.437*self.BWeight_kgb  
        self.r_cm = 1.18/2                   
        self.volume_cm3 = math.pi*math.pow(self.r_cm, 2)*self.length_cm  

        self.total_discretize = 101    
        self.total_node_num = self.total_discretize - 1 
        Duo_exclude_node0_discretize = self.total_discretize - 1
        DuoV_cm3_total = np.linspace(0, self.volume_cm3, self.total_discretize) 

        self.DuoV_cm3 = DuoV_cm3_total[1:] 
        self.Duo_single_node_V = self.volume_cm3/self.total_node_num

        self.MRT_min = 2.808 
        self.Kp_Duo_min = 1/self.MRT_min   

        self.VF = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/self.MRT_min 
        self.init_Duo_CPu = np.zeros(Duo_exclude_node0_discretize) 
        self.init_Duo_CPsl = np.zeros(Duo_exclude_node0_discretize)
        self.init_Duo_CPr = np.zeros(Duo_exclude_node0_discretize)
    
    def calculate_duo_endog(self):
        SI_volume_cm3 = self.volume_cm3 + self.volume_jej_cm3 + self.volume_il_cm3
        self.vol_prop = self.volume_cm3 / SI_volume_cm3

        Il_BasalAA = 0.799 
        self.basalaa = Il_BasalAA * self.vol_prop 

        Duo_basalaa_single_node = self.basalaa/self.total_node_num
        P_CPe_Duo_gmin = self.Kp_endog_min*Duo_basalaa_single_node
        self.P_CPe_gmincm3 = P_CPe_Duo_gmin/self.Duo_single_node_V

    def solving_Unode0(self):
        # Prepare JAX arrays for foregut results
        fore_t = self.fore_t
        fore_y = self.fore_y
        
        args_u = (fore_t, fore_y, self.constants['UP_Fr'], self.Kp_PVG_min, self.Kp_Duo_min)
        args_sl = (fore_t, fore_y, self.constants['SlP_Fr'], self.Kp_PVG_min, self.Kp_Duo_min)
        args_r = (fore_t, fore_y, self.constants['RP_Fr'], self.Kp_PVG_min, self.Kp_Duo_min)

        solver = diffrax.Kvaerno5()  
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]
        y0 = jnp.array(self.iDuo_g)

        self.result_UDnode0 = diffrax.diffeqsolve(
            diffrax.ODETerm(_solve_node0_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_u, saveat=saveat, stepsize_controller=stepsize_controller
        )
        self.result_SlDnode0 = diffrax.diffeqsolve(
            diffrax.ODETerm(_solve_node0_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_sl, saveat=saveat, stepsize_controller=stepsize_controller
        )
        self.result_RDnode0 = diffrax.diffeqsolve(
            diffrax.ODETerm(_solve_node0_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_r, saveat=saveat, stepsize_controller=stepsize_controller
        )

    @time_it
    def solving_duo_USl(self):
        y0_u = jnp.zeros(len(self.init_Duo_CPu)).at[0].set(self.result_UDnode0.ys[-1, 0])
        y0_sl = jnp.zeros(len(self.init_Duo_CPsl)).at[0].set(self.result_SlDnode0.ys[-1, 0])
        
        args_u = (self.result_UDnode0, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.P_CPe_gmincm3)
        args_sl = (self.result_SlDnode0, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.k_digestrate)
        
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

        self.UDexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_CPu_Duo_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_u, args=args_u, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )
        
        with time_block("Solving duodenum slowly-digested protein equations"):
            self.SlDexit_SS = diffrax.diffeqsolve(
                diffrax.ODETerm(_method_of_lines_CPsl_Duo_jax), solver, t0=t0, t1=t1, dt0=0.1,
                y0=y0_sl, args=args_sl, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
            )

    def flatten_result_duo_CPu(self):
        args_u = (self.result_UDnode0, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.P_CPe_gmincm3)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_CPu_metrics(t, y, args_u)
        
        # Why jax.vmap?
        # Instead of using a slow Python "for" loop to calculate the metrics (flux, concentrations)
        # for each of the 2,800 time steps one by one, vmap "vectorizes" the operation.
        # It calculates all time steps simultaneously using the compiled code, saving massive time.
        v_compute = jax.vmap(compute_metrics)
        dUDdt_all, flux_all, C_all = v_compute(self.UDexit_SS.ts, jnp.maximum(self.UDexit_SS.ys, 0.0))
        
        self.df_UP_d = {
            't': self.UDexit_SS.ts,
            'dUDdt': dUDdt_all[:, -1],
            'flux_CPu_DuoJej': flux_all,
            'Conc_CPu_Duolast': C_all,
            'QCPu_duo': jnp.maximum(self.UDexit_SS.ys[:, -1], 0.0)
        }
        return self.df_UP_d, None
    
    def flatten_result_duo_CPsl(self):
        args_sl = (self.result_SlDnode0, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.k_digestrate)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_CPsl_metrics(t, y, args_sl)
        
        v_compute = jax.vmap(compute_metrics)
        dSlDdt_all, flux_all, C_all = v_compute(self.SlDexit_SS.ts, jnp.maximum(self.SlDexit_SS.ys, 0.0))
        
        self.df_SlP_d = {
            't': self.SlDexit_SS.ts,
            'dDSldt': dSlDdt_all[:, -1],
            'flux_CPsl_DuoJej': flux_all,
            'Conc_CPsl_Duolast': C_all,
            'QCPsl_duo': jnp.maximum(self.SlDexit_SS.ys[:, -1], 0.0)
        }
        return self.df_SlP_d, None

    @time_it
    def SlP_for_RP_d(self):
        self.flatten_result_duo_CPsl()
        Duo_CPsl_Q = self.df_SlP_d['QCPsl_duo']
        return None, Duo_CPsl_Q

    @time_it
    def solving_duo_R(self):
        self.df_Q_CPsl_Duo, self.Duo_CPsl_Q = self.SlP_for_RP_d()

        y0_r = jnp.zeros(len(self.init_Duo_CPr)).at[0].set(self.result_RDnode0.ys[-1, 0])
        
        args_r = (self.result_RDnode0, self.SlDexit_SS, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.k_absp, self.k_digestrate)
        
        # Original was LSODA, Tsit5 or Dopri5 is suitable for non-stiff or moderately stiff
        solver = diffrax.Kvaerno5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.RDexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_CPr_Duo_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_r, args=args_r, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )

    def flatten_result_duo_CPr(self):
        args_r = (self.result_RDnode0, self.SlDexit_SS, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.k_absp, self.k_digestrate)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_CPr_metrics(t, y, args_r)
        
        v_compute = jax.vmap(compute_metrics)
        dRDdt_all, flux_all, C_all = v_compute(self.RDexit_SS.ts, jnp.maximum(self.RDexit_SS.ys, 0.0))
        
        self.df_RP_d = {
            't': self.RDexit_SS.ts,
            'dRDdt': dRDdt_all[:, -1],
            'flux_CPr_DuoJej': flux_all,
            'C_CPr__Duolast_gcm3': C_all,
            'QCPr_duo': jnp.maximum(self.RDexit_SS.ys[:, -1], 0.0)
        }
        return self.df_RP_d, None

    def solving_duo_feed(self):
        fore_t = self.fore_t
        fore_y = self.fore_y
        args_feed = (fore_t, fore_y, self.Kp_PVG_min, self.Kp_Duo_min)
        
        solver = diffrax.Kvaerno5()  
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]
        y0 = jnp.array(self.iDuo_g)

        self.result_feed_duo = diffrax.diffeqsolve(
            diffrax.ODETerm(_feed_duo_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_feed, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )
    
    def flatten_result_duo_feed(self):
        fore_t = self.fore_t
        fore_y = self.fore_y
        args_feed = (fore_t, fore_y, self.Kp_PVG_min, self.Kp_Duo_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _feed_duo_metrics(t, y, args_feed)
            
        v_compute = jax.vmap(compute_metrics)
        dfeedduodt_all, flux_all, Q_all = v_compute(self.result_feed_duo.ts, jnp.maximum(self.result_feed_duo.ys, 0.0))
        
        self.df_feed_d = {
            't': self.result_feed_duo.ts,
            'dfeedduodt': dfeedduodt_all,
            'flux_feed_DuoJej': flux_all,
            'Qfeed_duo': jnp.maximum(self.result_feed_duo.ys[:, 0], 0.0)
        }
        return self.df_feed_d, None 

def plot_duo(self):

        plt.clf() 
        plt.plot(self.df_UP_d['t'], self.df_UP_d['QCPu_duo'], color='blue') #, label='Undigestible Protein_duodenum g'
        plt.plot(self.df_SlP_d['t'], self.df_SlP_d['QCPsl_duo'], color='red')#, label='Slowly-digested Protein_duodenum g'
        plt.plot(self.df_RP_d['t'], self.df_RP_d['QCPr_duo'], color='green')#, label='Rapidly-digested Protein_duodenum g'
        #plt.legend()
        plt.tight_layout
        plt.xlabel('Time')
        plt.ylabel('Protein (g) in the duodenum')
        #plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
        plt.savefig('images/B-protein-over-time-at-last-node-duodenum.png', bbox_inches='tight')
        
        plt.clf() 
        plt.plot(self.df_UP_d['t'], self.df_UP_d['flux_CPu_DuoJej'], color='blue') #, label='Undegradable Protein_duodenum flux'
        plt.plot(self.df_SlP_d['t'], self.df_SlP_d['flux_CPsl_DuoJej'], color='red') #, label='Slowly-digested Protein_duodenum flux'
        plt.plot(self.df_RP_d['t'], self.df_RP_d['flux_CPr_DuoJej'], color='green')#, label='Rapidly-digested Protein_duodenum flux'
        #plt.legend()
        plt.tight_layout
        plt.xlabel('Time')
        plt.ylabel('Flux of protein (g/min) at the end of the duodenum')
        #plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
        plt.savefig('images/B-protein-over-time-at-last-node-duodenum_flux.png', bbox_inches='tight')
        
        plt.clf() 
        plt.plot(self.df_feed_d['t'], self.df_feed_d['Qfeed_duo'], color='red', label = 'Q feed duodenum')
        plt.plot(self.df_feed_d['t'], self.df_feed_d['flux_feed_DuoJej'], color='blue', label='Flux feed duodenum')
        #plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
        plt.xlabel('Time')
        plt.ylabel('Feed (g) and Flux (g/min) in the duodenum')
        plt.tight_layout(pad=2.0)
        plt.savefig('images/B-feed-over-time-duodenum.png', bbox_inches='tight')
        
    

        