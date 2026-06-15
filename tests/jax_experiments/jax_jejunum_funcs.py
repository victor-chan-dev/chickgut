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
