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
