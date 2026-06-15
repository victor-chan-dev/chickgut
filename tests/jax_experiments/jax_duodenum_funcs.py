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
    
    Duo_CPr_dis = (k_absp * QCPr_Duo_g[1:]**2) / Duo_single_node_V
    U_CPr_Duo_psg_mincm3 = VF * jnp.diff(QCPr_Duo_g) / jnp.diff(DuoV_cm3)
    U_CPr_Duo_psg_dis_gmincm3 = jnp.maximum(U_CPr_Duo_psg_mincm3 + Duo_CPr_dis, 1e-12)
    
    P_CPr_Duo0Duo_gmincm3 = (Kp_Duo_min * val0) / Duo_single_node_V
    
    QCPsl_Duo_g_last = CPsl_sol.evaluate(t)[-1]
    P_CPr_DuoSlDuoR_gmincm3 = (k_digestrate * QCPsl_Duo_g_last) / Duo_single_node_V
    
    RD_Diff = P_CPr_DuoSlDuoR_gmincm3 + P_CPr_Duo0Duo_gmincm3 - U_CPr_Duo_psg_dis_gmincm3
    dRDdt_concat = jnp.concatenate([jnp.array([0.0]), RD_Diff])
    return jnp.maximum(dRDdt_concat, 1e-12)

@jax.jit
def _method_of_lines_CPr_metrics(t, y, args):
    (node0_sol, CPsl_sol, VF, DuoV_cm3, Kp_Duo_min, Duo_single_node_V, k_absp, k_digestrate) = args
    QCPr_Duo_g = y.at[0].set(node0_sol.evaluate(t)[0])
    
    Duo_CPr_dis = (k_absp * QCPr_Duo_g[1:]**2) / Duo_single_node_V
    U_CPr_Duo_psg_mincm3 = VF * jnp.diff(QCPr_Duo_g) / jnp.diff(DuoV_cm3)
    U_CPr_Duo_psg_dis_gmincm3 = jnp.maximum(U_CPr_Duo_psg_mincm3 + Duo_CPr_dis, 1e-12)
    
    P_CPr_Duo0Duo_gmincm3 = (Kp_Duo_min * node0_sol.evaluate(t)[0]) / Duo_single_node_V
    
    QCPsl_Duo_g_last = CPsl_sol.evaluate(t)[-1]
    P_CPr_DuoSlDuoR_gmincm3 = (k_digestrate * QCPsl_Duo_g_last) / Duo_single_node_V
    
    RD_Diff = P_CPr_DuoSlDuoR_gmincm3 + P_CPr_Duo0Duo_gmincm3 - U_CPr_Duo_psg_dis_gmincm3
    dRDdt = jnp.maximum(jnp.concatenate([jnp.array([0.0]), RD_Diff]), 1e-12)
    
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
