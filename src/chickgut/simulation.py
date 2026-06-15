import math
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from chickgut.anatomy.duodenum import Duodenum
from chickgut.anatomy.jejunum import Jejunum
from chickgut.anatomy.ileum import Ileum
from chickgut.utils.performance import time_it, time_block

# Time variables config
maxt = 0.0001
cint = 1.0
tstp = 2880
t_span = (0.0, tstp)
t_eval = np.arange(0.0, tstp + maxt, cint)

# Default optimized model parameters
k_absp = 63.7867
k_digestrate = 0.08788306992043704
Kp_endog_min = 0.01428015551135433

# setup output directories if needed (imported from os)
csv_datafiles = 'Python_VSCode_Files'

def run_simulation(ingr_name, constants, k_absp, k_digestrate, Kp_endog_min, t_eval, t_span, export_dataframes=True):
    """
    Solves the foregut simulation (crop, proventriculus/gizzard) and hindgut simulation.
    Returns (duodenum_instance, jejunum_instance, ileum_instance, df_fore).
    """
    def foregut_simulation(t_eval, t_span, constants):
        BWeight_kgb = constants['birdBW_kg']                      # Body weight (kg/bird) of the bird from: Bryan et al. 2019 
        BWeight_gb = BWeight_kgb*100             # Body weight (g/bird) of the bird based on kg/bird
        
        feed_times = [0, 480, 720, 960, 1200, 1440]  # Define when boluses are given (in minutes)
        meal_num = len(feed_times)
        feed_duration = 230.4/meal_num
        bolus_feed_g = constants['FI_24h_gbird_onDM']/meal_num  # Define size of the bolus in grams
        Kp_feed_intake_gmin = bolus_feed_g/feed_duration
        Kp_meal_intake = 1/feed_duration 
        
        pellet_density = (0.726 - 0.363)/2
        crop_volume = 67.35*BWeight_kgb
        max_g_Crop = crop_volume*pellet_density

        # Initialize feed intake as zeros
        feed_intake_g = np.zeros(len(t_eval)) 
        
        def feeding():
            for t in feed_times:
                for CP_intake_interval in range (int(t), int(t+feed_duration)):
                    FT_index = np.where(t_eval == CP_intake_interval)[0][0]
                    feed_intake_g[FT_index] = Kp_feed_intake_gmin
            return feed_intake_g

        feed_intake_g = feeding()
        df_Q_meal_CP_g = pd.DataFrame({'Time': t_eval, 'Q_meal_g': feed_intake_g})
        meal_intake_Q_g = df_Q_meal_CP_g['Q_meal_g'].values 

        iCr_g = 0.000001                 # initial pool size for crop (g of CP)  
        iPVG_g = 0.000001               # initial pool size for proventriculus-gizzard (g of CP)
        ifeedPVG_g = 0.000001
        y0_CPVG = [iCr_g, iPVG_g, ifeedPVG_g]
        iDuo_g = [0.000001]
        
        Kp_Cr_min = 0.025    # fractional passage rate out of the crop (/min)
        Kp_PVG_min = 0.029  # fractional passage rate out of the pvg (/min)
    
        def CPVG(t, y):
            Qfeed_Cr_g, QCP_PVG_g, Qfeed_PVG_g = y
            
            def Crop():            
                index_meal_intake = min(np.searchsorted(df_Q_meal_CP_g['Time'], t), len(df_Q_meal_CP_g['Time']) - 1)
                Q_meal_CP_g_at_t = meal_intake_Q_g[index_meal_intake]            
                if Qfeed_Cr_g < max_g_Crop:            
                    P_meal_CP_gmin = Kp_meal_intake*(Q_meal_CP_g_at_t)
                else:
                    P_meal_CP_gmin = 0.000000000001*(Kp_meal_intake*(Q_meal_CP_g_at_t))
                    
                U_CP_CrPVG_gmin = Kp_Cr_min*(Qfeed_Cr_g)
                dCrdt = P_meal_CP_gmin - U_CP_CrPVG_gmin
                return dCrdt, U_CP_CrPVG_gmin, Qfeed_Cr_g, P_meal_CP_gmin
            
            def PVGizzard():
                __, _, Qfeed_Cr_g, _ = Crop()
                QCP_Cr_g = Qfeed_Cr_g*(constants['CP_diet_p']/100)
                
                if Qfeed_Cr_g < 7.0:
                    P_CP_PVG_gmin = Kp_Cr_min*(QCP_Cr_g)
                    P_feed_Cr = Qfeed_Cr_g*Kp_Cr_min
                else:
                    P_CP_PVG_gmin = 0.0000000000001*(Kp_Cr_min*(QCP_Cr_g))
                    P_feed_Cr = 0.0000000000001*(Qfeed_Cr_g*Kp_Cr_min)
                    
                U_CP_PVGDuo_UP_gmin = Kp_Cr_min*(QCP_PVG_g*constants['UP_Fr'])
                U_CP_PVGDuo_SlP_gmin = Kp_Cr_min*(QCP_PVG_g*constants['SlP_Fr'])
                U_CP_PVGDuo_RP_gmin = Kp_Cr_min*(QCP_PVG_g*constants['RP_Fr'])
                U_feed_PVGDuo = Qfeed_PVG_g*Kp_PVG_min
                
                dPVGdt = P_CP_PVG_gmin - U_CP_PVGDuo_UP_gmin - U_CP_PVGDuo_SlP_gmin - U_CP_PVGDuo_RP_gmin
                dPVG_feeddt = P_feed_Cr - U_feed_PVGDuo
            
                return dPVGdt, dPVG_feeddt, U_CP_PVGDuo_UP_gmin, U_CP_PVGDuo_SlP_gmin, U_CP_PVGDuo_RP_gmin, QCP_PVG_g, Qfeed_PVG_g, P_CP_PVG_gmin
        
            dCrdt, U_CP_CrPVG_gmin, QCP_Cr_g, P_meal_CP_gmin = Crop()
            dPVGdt, dPVG_feeddt, U_CP_PVGDuo_UP_gmin, U_CP_PVGDuo_SlP_gmin, U_CP_PVGDuo_RP_gmin, QCP_PVG_g, Qfeed_PVG_g, P_CP_PVG_gmin = PVGizzard()
            
            return dCrdt, dPVGdt, dPVG_feeddt, U_CP_CrPVG_gmin, U_CP_PVGDuo_UP_gmin, U_CP_PVGDuo_SlP_gmin, U_CP_PVGDuo_RP_gmin, QCP_Cr_g, QCP_PVG_g, Qfeed_PVG_g
        
        def Solve_CPVG(t, y):
            dCrdt, dPVGdt, dPVG_feeddt, _, _, _, _, _, _, _ = CPVG(t, y)
            return [dCrdt, dPVGdt, dPVG_feeddt]
        
        result_fore = solve_ivp(Solve_CPVG, t_span, y0=y0_CPVG, t_eval=t_eval, dense_output=True, method='Radau') 

        def flatten_result_fore(result_fore):
            flattened_data = []
            for t, y in zip(result_fore.t, result_fore.y.T):
                entry = CPVG(t, y)
                
                dCrdt = entry[0]
                dPVGdt = entry[1]
                dPVG_feeddt = entry[2]
                U_CP_CrPVG_gmin = entry[4]
                U_CP_PVGDuo_gmin = entry[5]
                QCP_Cr_g = y[0]
                QCP_PVG_g = y[1]
                Qfeed_PVG_g = y[2]
                flattened_entry = {
                    't': t, 
                    'dCrdt': dCrdt,
                    'dPVGdt': dPVGdt,
                    'dPVG_feeddt': dPVG_feeddt,
                    'CrPVG_flux': U_CP_CrPVG_gmin,
                    'PVGDuo_flux': U_CP_PVGDuo_gmin,
                    'Cr': QCP_Cr_g,
                    'PVG': QCP_PVG_g,
                    'feed_PVG': Qfeed_PVG_g,
                }
                flattened_data.append(flattened_entry)
            df_cpvg = pd.DataFrame(flattened_data)
            return df_cpvg, flattened_data
        
        df_cpvg, _ = flatten_result_fore(result_fore)
        df_fore = pd.DataFrame(df_cpvg)
        
        # Save output inside the simulation directory
        os.makedirs(csv_datafiles, exist_ok=True)
        df_fore.to_csv(os.path.join(csv_datafiles, 'model_output_fore-Bryan.csv'), index=False)
        
        return result_fore, iDuo_g, BWeight_kgb, Kp_PVG_min, df_fore

    result_fore, iDuo_g, BWeight_kgb, Kp_PVG_min, df_fore = foregut_simulation(t_eval, t_span, constants)
    print("finished foregut")
    
    duodenum_instance, jejunum_instance, ileum_instance = HindGIT(
        ingr_name, t_eval, t_span, constants, k_absp, k_digestrate, Kp_endog_min, 
        result_fore.t, result_fore.y, iDuo_g, BWeight_kgb, Kp_PVG_min, export_dataframes
    )
    return duodenum_instance, jejunum_instance, ileum_instance, df_fore

@time_it
def HindGIT(ingr_name, t_eval, t_span, constants, k_absp, k_digestrate, Kp_endog_min, fore_t, fore_y, iDuo_g, BWeight_kgb, Kp_PVG_min, export_dataframes=True):
    # 1. creating duodenum, jejunum and ileum
    duodenum_instance = Duodenum(t_span, iDuo_g, t_eval, fore_t, fore_y, BWeight_kgb, Kp_PVG_min, constants, k_absp, k_digestrate, Kp_endog_min)
    jejunum_instance = Jejunum(t_span, t_eval, constants, BWeight_kgb, k_absp, k_digestrate, Kp_endog_min)
    ileum_instance = Ileum(t_span, t_eval, constants, BWeight_kgb, k_absp, k_digestrate, Kp_endog_min)
    
    # 2. calculate basic properties
    duodenum_instance.calculate_duo_prop()
    jejunum_instance.calculate_jej_prop()
    ileum_instance.calculate_Il_prop()

    # 3. calculate volumes for all compartments to be used to calculate endogenous 
    duodenum_instance.calculate_V(jejunum_instance, ileum_instance)
    jejunum_instance.calculate_V(duodenum_instance, ileum_instance)
    ileum_instance.calculate_V(duodenum_instance, jejunum_instance)

    # 4. calculate endogenous values
    duodenum_instance.calculate_duo_endog()
    jejunum_instance.calculate_jej_endog()
    ileum_instance.calculate_Il_endog()

    with time_block("\n~~~Calculating duodenum properties"):
        # 5. run duodenum to get exit values
        with time_block("Solving duodenum equations"):
            duodenum_instance.solving_Unode0()
            duodenum_instance.solving_duo_USl()
            duodenum_instance.solving_duo_R()
            duodenum_instance.solving_duo_feed()
        with time_block("Flattening duodenum results"):
            if export_dataframes:
                duodenum_instance.flatten_result_duo_CPu()
                duodenum_instance.flatten_result_duo_CPsl()
                duodenum_instance.flatten_result_duo_CPr()
                duodenum_instance.flatten_result_duo_feed()
        Duo_single_node_V = duodenum_instance.calculate_duo_prop()
        Kp_Duo_min = duodenum_instance.calculate_duo_prop()
        print("finished duodenum")

    with time_block("\n~~~Calculating jejunum properties"):
        # 6. set exit values for jejunum from duo and run jej to get jej exit values
        with time_block("Solving jejunum equations"):
            jejunum_instance.Duo_results(duodenum_instance)
            jejunum_instance.solving_jej_USl()
            jejunum_instance.solving_jej_R()
            jejunum_instance.solving_jej_feed()
        with time_block("Flattening jejunum results"):
            if export_dataframes:
                jejunum_instance.flatten_result_jej_CPu()
                jejunum_instance.flatten_result_jej_CPsl()
                jejunum_instance.flatten_result_jej_CPr()
                jejunum_instance.flatten_result_jej_feed()
        Discretize_jej = jejunum_instance.calculate_jej_prop()
        print("finished jejunum")

    with time_block("\n~~~Calculating ileum properties"):
        Kp_Jej_min = jejunum_instance.calculate_jej_prop()
        # 7. set exit values for ileum from jej and run ileum to get ileum exit values
        with time_block("Solving ileum equations"):
            ileum_instance.Jej_results(jejunum_instance)
            ileum_instance.solving_il_USl()
            ileum_instance.solving_il_R()
            ileum_instance.solving_il_feed()
        with time_block("Flattening ileum results"):
            if export_dataframes:
                ileum_instance.flatten_result_il_CPu()
                ileum_instance.flatten_result_il_CPsl()
                ileum_instance.flatten_result_il_CPr()
                ileum_instance.flatten_result_il_feed()
        print("finished ileum")
    
    return duodenum_instance, jejunum_instance, ileum_instance

def precompute_foregut(constants, t_eval, t_span):
    # Body weight
    BWeight_kgb = constants['birdBW_kg']
    
    feed_times = [0, 480, 720, 960, 1200, 1440]  
    meal_num = len(feed_times)
    feed_duration = 230.4/meal_num
    bolus_feed_g = constants['FI_24h_gbird_onDM']/meal_num 
    Kp_feed_intake_gmin = bolus_feed_g/feed_duration
    Kp_meal_intake = 1/feed_duration 
    
    pellet_density = (0.726 - 0.363)/2
    crop_volume = 67.35*BWeight_kgb
    max_g_Crop = crop_volume*pellet_density

    feed_intake_g = np.zeros(len(t_eval)) 
    for t in feed_times:
        for CP_intake_interval in range(int(t), int(t+feed_duration)):
            FT_index = np.where(t_eval == CP_intake_interval)[0][0]
            feed_intake_g[FT_index] = Kp_feed_intake_gmin
    
    df_Q_meal_CP_g = pd.DataFrame({'Time': t_eval, 'Q_meal_g': feed_intake_g})
    meal_intake_Q_g = df_Q_meal_CP_g['Q_meal_g'].values 

    y0_CPVG = [0.000001, 0.000001, 0.000001]
    iDuo_g = [0.000001]
    
    Kp_Cr_min = 0.025   
    Kp_PVG_min = 0.029  

    def CPVG(t, y):
        Qfeed_Cr_g, QCP_PVG_g, Qfeed_PVG_g = y
        
        index_meal_intake = min(np.searchsorted(df_Q_meal_CP_g['Time'], t), len(df_Q_meal_CP_g['Time']) - 1)
        Q_meal_CP_g_at_t = meal_intake_Q_g[index_meal_intake]            
        if Qfeed_Cr_g < max_g_Crop:            
            P_meal_CP_gmin = Kp_meal_intake*(Q_meal_CP_g_at_t)
        else:
            P_meal_CP_gmin = 0.000000000001*(Kp_meal_intake*(Q_meal_CP_g_at_t))
            
        U_CP_CrPVG_gmin = Kp_Cr_min*(Qfeed_Cr_g)
        dCrdt = P_meal_CP_gmin - U_CP_CrPVG_gmin

        QCP_Cr_g = Qfeed_Cr_g*(constants['CP_diet_p']/100)
        
        if Qfeed_Cr_g < 7.0:
            P_CP_PVG_gmin = Kp_Cr_min*(QCP_Cr_g)
            P_feed_Cr = Qfeed_Cr_g*Kp_Cr_min
        else:
            P_CP_PVG_gmin = 0.0000000000001*(Kp_Cr_min*(QCP_Cr_g))
            P_feed_Cr = 0.0000000000001*(Qfeed_Cr_g*Kp_Cr_min)
            
        U_CP_PVGDuo_UP_gmin = Kp_Cr_min*(QCP_PVG_g*constants['UP_Fr'])
        U_CP_PVGDuo_SlP_gmin = Kp_Cr_min*(QCP_PVG_g*constants['SlP_Fr'])
        U_CP_PVGDuo_RP_gmin = Kp_Cr_min*(QCP_PVG_g*constants['RP_Fr'])
        U_feed_PVGDuo = Qfeed_PVG_g*Kp_PVG_min
        
        dPVGdt = P_CP_PVG_gmin - U_CP_PVGDuo_UP_gmin - U_CP_PVGDuo_SlP_gmin - U_CP_PVGDuo_RP_gmin
        dPVG_feeddt = P_feed_Cr - U_feed_PVGDuo

        return [dCrdt, dPVGdt, dPVG_feeddt]
    
    result_fore = solve_ivp(CPVG, t_span, y0=y0_CPVG, t_eval=t_eval, dense_output=True, method='Radau')
    return result_fore, iDuo_g, BWeight_kgb, Kp_PVG_min

def pure_sim_flux(params, t_eval, t_span, constants, fore_t, fore_y, iDuo_g, BWeight_kgb, Kp_PVG_min):
    """
    A pure JAX-compatible simulation that takes parameters and returns the final flux sum.
    """
    k_absp, k_digestrate, Kp_endog_min = params[0], params[1], params[2]
    
    duo = Duodenum(t_span, iDuo_g, t_eval, fore_t, fore_y, BWeight_kgb, Kp_PVG_min, constants, k_absp, k_digestrate, Kp_endog_min)
    jej = Jejunum(t_span, t_eval, constants, BWeight_kgb, k_absp, k_digestrate, Kp_endog_min)
    il = Ileum(t_span, t_eval, constants, BWeight_kgb, k_absp, k_digestrate, Kp_endog_min)
    
    duo.calculate_duo_prop()
    jej.calculate_jej_prop()
    il.calculate_Il_prop()
    
    duo.calculate_V(jej, il)
    jej.calculate_V(duo, il)
    il.calculate_V(duo, jej)
    
    duo.calculate_duo_endog()
    jej.calculate_jej_endog()
    il.calculate_Il_endog()
    
    duo.solving_Unode0()
    duo.solving_duo_USl()
    duo.solving_duo_R()
    duo.solving_duo_feed()
    
    jej.Duo_results(duo)
    jej.solving_jej_USl()
    jej.solving_jej_R()
    jej.solving_jej_feed()
    
    il.Jej_results(jej)
    il.solving_il_USl()
    il.solving_il_R()
    il.solving_il_feed()
    
    sum_UI_flux, sum_SlI_flux, sum_RI_flux = il.get_flux_sums(limit=2001)
    return sum_UI_flux + sum_SlI_flux + sum_RI_flux
