"""
by Grace Hong, University of Guelph

(1) This model simulates CP digestion in a broiler chicken, within the GROWER phase (d 14-28)
(2) the plug-flow reactor code is based off of: https://kitchingroup.cheme.cmu.edu/blog/2013/03/06/Modeling-a-transient-plug-flow-reactor/

UNITS:
g, min, cm^3

Variable Abbreviations start with:
Q - quantity (g)
P - production (pool inflow) (g/min)
U - utilization (pool outflow) (g/min)
C - concentration (g/cm^3)
"""


"""   Import Packages """
import math
import pandas as pd
import numpy as np 
import pickle
import os
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

"Import classes"
from Duodenum_Class import Duodenum
from Jejunum_Class import Jejunum
from Ileum_Class import Ileum

csv_datafiles = 'Python_VSCode_Files' #setup for saving files later with relative paths
os.makedirs(csv_datafiles, exist_ok=True) #makes sure that this folder exists

"""   Set Time Variables """
maxt = 0.0001                        #max integration interval
cint=1.0                     #communication interval
tstp=2880                           #simulation time - minutes
t_span = (0.0, tstp)                #timespan for simulation (start, stop)
t_eval = np.arange(0.0,tstp+maxt,cint)
#time = np.arange(0.0, tstp+1, 1)
#                                    solve_ivp will call dynamic() multiple times over the t_span, sending it the
#                                    appropriate t and statevars values; simresults are reported every cint

#---------Importing diet constant variables-----------------------#
script_dir = os.path.abspath("Plug_Flow_Diet_Charac.csv")

#df_ing = pd.read_csv('E:/PhD Animal Biosciences/Plug_Flow/Plug_Flow_Diet_Charac.csv')
df_ing = pd.read_csv(script_dir)
print(df_ing)
ingr_const = df_ing.set_index('Ingredient').T.to_dict() #sets index to apply to each ingredient and then transposes it, creating dictionaries
def get_ing_const(ingr_name):
    if ingr_name in ingr_const:             #checks if ingr_name matches an existing key(i.e. ingredient) in ingr_cost
        return ingr_const[ingr_name]        #if ingr_name in ingr_cost is true, it returns the value (the dictonary) associated with the ingredient
ingr_name = input("enter ingredient name: ").lower() #type in ingredient here, .lower converts all input letters to lowercase for matching 
#V: Run a loop for all feeds
constants = get_ing_const(ingr_name)

k_absp = 63.7867#49.173624234454486 #63.7867 #9.832828 #3.5988575 #7.92 #previously optimized number, not correct  #388.565621
#k_digestrate = constants['k_digestrate_SlP_min']
k_digestrate = 0.08788306992043704 #0.0606653 #0.0091656011 #0.0087809 #0.00973365 #0.44746083
Kp_endog_min = 0.01428015551135433 #0.000527035 #0.00107543685 #0.00779597 #0.00694351 #0.742780423


def GIT(t_eval, t_span):
    #----- Bird Description & Intake ------#
    #Per bird, grower stage
    Breed = "Ross"
    Age = 21
    #---phase---#
    # if 0 <= Age <= 10:
    #     Phase = "Starter"
    # elif 11 <= Age <=25:
    #     Phase = "Grower"
    # elif 26 <= Age <=45:
    #     Phase = "Finisher"
    # else:
    #     Phase = "Breeder"
        
    BWeight_kgb = constants['birdBW_kg']                      # Body weight (kg/bird) of the bird from: Bryan et al. 2019 
    BWeight_gb = BWeight_kgb*100             # Body weight (g/bird) of the bird based on kg/bird
    # FI_gbmin = constants['FI_diet_gbirdday']/1440                    # Feed intake (g/bird/day) from 14-21DOA Blood Meal from: Bryan et al. 2019 - 
    # CP_intake_gmin = (constants['CP_diet_gkg']*FI_gbmin)/1000   #CP intake (g of CP/min) from: Bryan et al. 2019 for blood meal Table 3
    
    #------ Bolus feeding setup: assuming bolus of feed is given at specific time points-------#
    feed_times = [0, 480, 720, 960, 1200, 1440]  # , 720, 960, 1200, 1440, 1680, 1920, 2160, 2400, 2640 Define when boluses are given (in minutes) (8am, 12pm, 4pm, 8pm)
    meal_num = len(feed_times)
    feed_duration = 230.4/meal_num #(3.84h zhou et al. / 4 feeding times)
    bolus_feed_g = constants['FI_24h_gbird_onDM']/meal_num  # Define size of the bolus in grams, total day feed intake/time spent at feeder (3.84h)Zhou et al.
    #bolus_feed_g = 86.705/meal_num #constants['FI_24h_gbird']/meal_num
    #CP_intake_g = ((constants['CP_diet_p']/100)*bolus_feed_g) # g of CP per g of feed total per bolus
    #print(f"CP intake: {CP_intake_g}")
                    #g/kg of CP/DM divide by 1000 to get g/g 
    #Kp_CP_intake_gmin = CP_intake_g/feed_duration #g/day to g/min CP intake per min
    Kp_feed_intake_gmin = bolus_feed_g/feed_duration
    Kp_meal_intake = 1/feed_duration 
    #print(Kp_CP_intake_gmin)
    
    # Pellet size and g calculation #
    # pellet_radius_cm = 1.0
    # pellet_length_cm = 2.0
    # pellet_volume = math.pi*math.pow(pellet_radius_cm, 2)*pellet_length_cm
    pellet_density = (0.726 - 0.363)/2 #range 0.363 g/cm3 to 0.726g/cm3 ie average, Rasyid et al. 2018
    crop_volume = 67.35*BWeight_kgb #cm3
    max_g_Crop = crop_volume*pellet_density

    # Initialize feed intake as zeros (for continuous time steps)
    protein_intake_g = np.zeros(len(t_eval)) 
    feed_intake_g = np.zeros(len(t_eval)) 
    Q_meal_CP_gmin = np.zeros(len(t_eval))
    
    def feeding():
        for t in feed_times:
            # print(f"Processing feed time {t}, duration {feed_duration}")
            for CP_intake_interval in range (int(t), int(t+feed_duration)):
                FT_index = np.where(t_eval == CP_intake_interval)[0][0] #[0][0] accesses the first match, assumes that feed_time has exactly one time match in t_eval
                #protein_intake_g[FT_index] = Kp_CP_intake_gmin  # Add bolus feed at the specified time
                feed_intake_g[FT_index] = Kp_feed_intake_gmin
            #protein_intake is a dictionary and FT is the key which calls the bolus value
        #return protein_intake_g
        return feed_intake_g

    
    #protein_intake_g = feeding()
    feed_intake_g = feeding()
    # print(f"protein_intake_g[0]: {protein_intake_g[0]}")
    # print(f"protein_intake_g[1:10]: {protein_intake_g[1:10]}")
    #df_Q_meal_CP_g = pd.DataFrame({'Time': t_eval, 'Q_meal_CP_g': protein_intake_g})
    df_Q_meal_CP_g = pd.DataFrame({'Time': t_eval, 'Q_meal_g': feed_intake_g})

    meal_intake_Q_g = df_Q_meal_CP_g['Q_meal_g'].values 
    #print(meal_intake_Q_g)
    #-----initial pool size------#
    iCr_g=0.000001                 # initial pool size for crop (g of CP)  
    iPVG_g=0.000001               # initial pool size for proventriculus-gizzard (g of CP) - undigestible fraction 
    ifeedPVG_g=0.000001
    y0_CPVG=[iCr_g, iPVG_g, ifeedPVG_g]     #   load the initial values of Cr and PVG into y0_CPVG
    iDuo_g = [0.000001]
    #----------Fractional Passage Rate Constants (/min) (volumetric flow rate)---------------------------------#
    Kp_Cr_min = 0.025    # fractional (SOLID) passage rate out of the crop (/min), from average solid mrt from Garcon et al. 
    Kp_PVG_min = 0.029  # fractional (SOLID) passage rate out of the pvg (/min), from average solid mrt from Meta-a 
 
    #-----------Crop and proventriculus/gizzard function - non-plug flow --------------------------#
    def CPVG(t, y):
        Qfeed_Cr_g, QCP_PVG_g, Qfeed_PVG_g = y # undegradable protein in crop (g) and proventriculus/gizzard (g)
        
        def Crop():
            
            index_meal_intake = np.searchsorted(df_Q_meal_CP_g['Time'], t)
            
            Q_meal_CP_g_at_t = meal_intake_Q_g[index_meal_intake]
            
            if Qfeed_Cr_g < max_g_Crop:
                # if t == 0:  # Debug at time 0
                #     print(f"At t=0: index={index_meal_intake}, Q_meal_CP_g_at_t={Q_meal_CP_g_at_t}") 
            
                P_meal_CP_gmin = Kp_meal_intake*(Q_meal_CP_g_at_t) #50% enter the crop after fasting overnight Svihus et al.2014  
            
            else:
                P_meal_CP_gmin = 0.000000000001*(Kp_meal_intake*(Q_meal_CP_g_at_t))
                
            U_CP_CrPVG_gmin = Kp_Cr_min*(Qfeed_Cr_g)

            dCrdt = P_meal_CP_gmin - U_CP_CrPVG_gmin  #bolus
            #dCrdt = CP_intake_gmin - U_CP_CrPVG_gmin #continuous
            return dCrdt, U_CP_CrPVG_gmin, Qfeed_Cr_g, P_meal_CP_gmin
        
        def PVGizzard():
            __, _, Qfeed_Cr_g, _ = Crop()

            QCP_Cr_g = Qfeed_Cr_g*(constants['CP_diet_p']/100) #in CP rather than feed 
            
            if Qfeed_Cr_g < 7.0: #5-10g of feed is capacity of PVG - Svihus 2014
                P_CP_PVG_gmin = Kp_Cr_min*(QCP_Cr_g)# Flux PV/G to Duodenum - undigestible protein fraction Blood Meal - Bryan et al. 2019b Table 2
                         #k (i.e. passage rate of pvg) * amount of protein (g) in proventriculus/gizzard 
                P_feed_Cr = Qfeed_Cr_g*Kp_Cr_min
            else:
                P_CP_PVG_gmin = 0.0000000000001*(Kp_Cr_min*(QCP_Cr_g))
                P_feed_Cr = 0.0000000000001*(Qfeed_Cr_g*Kp_Cr_min)
                
            U_CP_PVGDuo_UP_gmin = Kp_Cr_min*(QCP_PVG_g*constants['UP_Fr']) # Flux crop to PV/G; 
                         #k (i.e. passage rate of crop) * amount of protein (g) in crop
            
            U_CP_PVGDuo_SlP_gmin = Kp_Cr_min*(QCP_PVG_g*constants['SlP_Fr'])
            
            U_CP_PVGDuo_RP_gmin = Kp_Cr_min*(QCP_PVG_g*constants['RP_Fr'])

            U_feed_PVGDuo = Qfeed_PVG_g*Kp_PVG_min
            
            dPVGdt = P_CP_PVG_gmin - U_CP_PVGDuo_UP_gmin - U_CP_PVGDuo_SlP_gmin - U_CP_PVGDuo_RP_gmin
            dPVG_feeddt = P_feed_Cr - U_feed_PVGDuo
        
            return dPVGdt, dPVG_feeddt, U_CP_PVGDuo_UP_gmin, U_CP_PVGDuo_SlP_gmin, U_CP_PVGDuo_RP_gmin, QCP_PVG_g, Qfeed_PVG_g, P_CP_PVG_gmin
       
    
        dCrdt, U_CP_CrPVG_gmin, QCP_Cr_g, P_meal_CP_gmin = Crop()# P_meal_CP_gmin
        
        dPVGdt, dPVG_feeddt, U_CP_PVGDuo_UP_gmin, U_CP_PVGDuo_SlP_gmin, U_CP_PVGDuo_RP_gmin, QCP_PVG_g, Qfeed_PVG_g, P_CP_PVG_gmin = PVGizzard()
        
        return dCrdt, dPVGdt, dPVG_feeddt, U_CP_CrPVG_gmin, U_CP_PVGDuo_UP_gmin, U_CP_PVGDuo_SlP_gmin, U_CP_PVGDuo_RP_gmin, QCP_Cr_g, QCP_PVG_g, Qfeed_PVG_g
    
    def Solve_CPVG(t, y ):
        dCrdt, dPVGdt, dPVG_feeddt, _, _, _, _, _, _, _ = CPVG(t, y)
        return [dCrdt, dPVGdt, dPVG_feeddt]
    
    #----------- solve for the values of Cr and PVG (CP, g) using solve_ivp --------------------------#
    result_fore=solve_ivp(Solve_CPVG, t_span, y0=y0_CPVG, t_eval=t_eval, dense_output=True, method='Radau') 
    #method=LSODA for automatic stiffness detection and switching, used for systems with different timescales 

    #--------------Exporting CPVG data---------------------------------------#
    def flatten_result_fore(result_fore):
        flattened_data = []

        for t, y in zip(result_fore.t, result_fore.y.T):
            entry = CPVG(t, y)
            
            dCrdt = entry[0]
            dPVGdt = entry[1]
            dPVG_feeddt = entry[2]
            U_CP_CrPVG_gmin= entry[4]
            U_CP_PVGDuo_gmin = entry[5]
            QCP_Cr_g=y[0]
            QCP_PVG_g=y[1]
            Qfeed_PVG_g=y[2]
            flattened_entry ={
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

    df_fore.to_csv(os.path.join(csv_datafiles, 'model_output_fore-Bryan.csv') , index=False)
    
    # def PVG_Duo_node0():
    #     pvgu_values_g = []
    #     pvgsl_values_g = []
    #     pvgr_values_g = []
    #     pvg_time = []
        
    #     # for i, t in enumerate(np.arange(0.0,tstp+maxt, cint)): #enumerate returns both the index and value for that element
    #     #     pvg = result_fore.y[-1, i]
            
    #     #     pvgu_values_g.append(pvg*constants['UP_Fr'])
    #     #     pvgsl_values_g.append(pvg*constants['SlP_Fr'])
    #     #     pvgr_values_g.append(pvg*constants['RP_Fr'])
    #     #     pvg_time.append(t)
            
    #     return pvg_time, pvgu_values_g, pvgsl_values_g, pvgr_values_g
    
    # pvg_time, pvgu_values_g, pvgsl_values_g, pvgr_values_g = PVG_Duo_node0()

  #-------------- Plot the results from the Crop and PV/G ---------------------------------------#  
    plt.plot(df_fore['t'], df_fore['Cr'], color='black') #, label='Cr'
    plt.plot(df_fore['t'], df_fore['PVG'], color='blue') #, label='PVGu'
    # plt.plot(df_fore['t'], df_fore['PVG_SlP'], color='red') #, label='PVGsl'
    # plt.plot(df_fore['t'], df_fore['PVG_RP'], color='green') #, label='PVGr'
    #plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Protein (g) in the gizzard/proventriculus')
    plt.tight_layout
    #plt.legend(bbox_to_anchor=(1, 1), fontsize=6)
    plt.savefig('images/B-protein-over-time-proventriculus_gizzard.png')
    
    plt.clf() 
    plt.plot(df_fore['t'], df_fore['CrPVG_flux'], color='black')  #, label='Cr flux'
    plt.plot(df_fore['t'], df_fore['PVGDuo_flux'], color='blue') #, label='PVG-Duo undigestible flux'
    # plt.plot(df_fore['t'], df_fore['PVGDuo_slflux'], color='red') #, label='PVG-Duo slowly-digested flux'
    # plt.plot(df_fore['t'], df_fore['PVGDuo_rflux'], color='green') #, label='PVG-Duo rapidly-digested flux'
    #plt.legend()
    plt.subplots_adjust(left=0.15)
    plt.xlabel('Time')
    plt.ylabel('flux of protein in the crop and PVG')
    plt.tight_layout
    #plt.legend(bbox_to_anchor=(1, 1), fontsize=6)
    plt.savefig('images/B-protein-over-time-proventriculus_gizzard_flux.png')
    
    return result_fore, iDuo_g, BWeight_kgb, Kp_PVG_min

result_fore, iDuo_g, BWeight_kgb, Kp_PVG_min = GIT(t_eval, t_span)

print("finished foregut")

def HindGIT(ingr_name, t_eval, t_span, constants, k_absp, k_digestrate, Kp_endog_min):
    # opt_constants = constants.copy() #making copy of the constants for optimizing the digestion rates
    # opt_constants["k_digestrate_SlP_min"] = k_digestrate

    #1. creating duodenum, jejunum and ileum
    duodenum_instance = Duodenum(t_span, iDuo_g, t_eval, result_fore, BWeight_kgb, Kp_PVG_min, constants, k_absp, k_digestrate, Kp_endog_min)
    jejunum_instance = Jejunum(t_span,  t_eval, constants, BWeight_kgb, k_absp, k_digestrate, Kp_endog_min)
    ileum_instance = Ileum(t_span, t_eval, constants, BWeight_kgb, k_absp, k_digestrate, Kp_endog_min)
    
    #3. calculate basic properties
    duodenum_instance.calculate_duo_prop()
    jejunum_instance.calculate_jej_prop()
    ileum_instance.calculate_Il_prop()

    #2. calculate volumes for all compartments to be used to calculate endogenous 
    duodenum_instance.calculate_V(jejunum_instance, ileum_instance)
    jejunum_instance.calculate_V(duodenum_instance, ileum_instance)
    ileum_instance.calculate_V(duodenum_instance, jejunum_instance)

    #4.calculate enodgenous values
    duodenum_instance.calculate_duo_endog()
    jejunum_instance.calculate_jej_endog()
    ileum_instance.calculate_Il_endog()

    #5. run duodenum to get exit values
    duodenum_instance.solving_Unode0()
    duodenum_instance.solving_duo_USl()
    duodenum_instance.solving_duo_R()
    duodenum_instance.solving_duo_feed()
    duodenum_instance.flatten_result_duo_CPu()
    duodenum_instance.flatten_result_duo_CPsl()
    duodenum_instance.flatten_result_duo_CPr()
    duodenum_instance.flatten_result_duo_feed()
    duodenum_instance.plot_duo()

    Duo_single_node_V = duodenum_instance.calculate_duo_prop()
    Kp_Duo_min = duodenum_instance.calculate_duo_prop()
    print("finished duodenum")

    #6. set exit values for jejunum from duo and run jej to get jej exit values
    jejunum_instance.Duo_results(duodenum_instance)
    jejunum_instance.solving_jej_USl()
    jejunum_instance.solving_jej_R()
    jejunum_instance.solving_jej_feed()
    jejunum_instance.flatten_result_jej_CPu()
    jejunum_instance.flatten_result_jej_CPsl()
    jejunum_instance.flatten_result_jej_CPr()
    jejunum_instance.flatten_result_jej_feed()
    jejunum_instance.plot_jej()
    
    Discretize_jej = jejunum_instance.calculate_jej_prop()
    print("finished jejunum")

    #7. set exit values for ileum from jej and run ileum to get ileum exit values
    ileum_instance.Jej_results(jejunum_instance)
    ileum_instance.solving_il_USl()
    ileum_instance.solving_il_R()
    ileum_instance.solving_il_feed()
    ileum_instance.flatten_result_il_CPu()
    ileum_instance.flatten_result_il_CPsl()
    ileum_instance.flatten_result_il_CPr()
    ileum_instance.flatten_result_il_feed()
    ileum_instance.plot_il()
    #ileum_instance.calculate()
    print("finished ileum")
    
    return duodenum_instance, jejunum_instance, ileum_instance

duodenum_instance, jejunum_instance, ileum_instance = HindGIT(ingr_name, t_eval, t_span, constants, k_absp, k_digestrate, Kp_endog_min)

# exporting orignial results - non-optimized #
def export_duojejil_results(duodenum_instance, jejunum_instance, ileum_instance):
    def export_duo():
        UDexit_SS = duodenum_instance.UDexit_SS
        SlDexit_SS = duodenum_instance.SlDexit_SS
        RDexit_SS = duodenum_instance.RDexit_SS

        df_UP_d = duodenum_instance.df_UP_d
        df_SlP_d = duodenum_instance.df_SlP_d
        df_RP_d = duodenum_instance.df_RP_d
        
        #writing UDexit_SS results to dataframe
        df_UD = pd.DataFrame(UDexit_SS.y.T, columns=[f'Node_{i}' for i in range(UDexit_SS.y.shape[0])])
        df_UD['Time'] = UDexit_SS.t #adds the time column
        df_UD = df_UD[['Time'] + [col for col in df_UD.columns if col != 'Time']] # Reorder columns to make 'Time' the first column

        #writing DDexit_SS results to dataframe
        df_SlD = pd.DataFrame(SlDexit_SS.y.T, columns=[f'Node_{i}' for i in range(SlDexit_SS.y.shape[0])])
        df_SlD['Time'] = SlDexit_SS.t #adds the time column
        df_SlD = df_SlD[['Time'] + [col for col in df_SlD.columns if col != 'Time']]
        
        #writing SDexit_SS results to dataframe
        df_RD = pd.DataFrame(RDexit_SS.y.T, columns=[f'Node_{i}' for i in range(RDexit_SS.y.shape[0])])
        df_RD['Time'] = RDexit_SS.t #adds the time column
        df_RD = df_RD[['Time'] + [col for col in df_RD.columns if col != 'Time']]
        
        
        with pd.ExcelWriter(os.path.join(csv_datafiles, "model_results_duo-Bryan.xlsx"), engine='openpyxl') as writer:
            df_UD.to_excel(writer, sheet_name='Duodenum_UD', index=False)
            df_UP_d.to_excel(writer, sheet_name='Duodenum_UD_atexit', index=False)
            df_SlD.to_excel(writer, sheet_name='Duodenum_SlD', index=False)
            df_SlP_d.to_excel(writer, sheet_name='Duodenum_SlD_atexit', index=False)
            df_RD.to_excel(writer, sheet_name='Duodenum_RD', index=False)
            df_RP_d.to_excel(writer, sheet_name='Duodenum_RD_atexit', index=False)
   
    def export_jej():
        UJexit_SS = jejunum_instance.UJexit_SS
        SlJexit_SS = jejunum_instance.SlJexit_SS
        RJexit_SS = jejunum_instance.RJexit_SS

        df_UP_j = jejunum_instance.df_UP_j
        df_SlP_j = jejunum_instance.df_SlP_j
        df_RP_j = jejunum_instance.df_RP_j
        
        #writing UJexit_SS results to dataframe
        df_UJ = pd.DataFrame(UJexit_SS.y.T, columns=[f'Node_{i}' for i in range(UJexit_SS.y.shape[0])])
        df_UJ['Time'] = UJexit_SS.t #adds the time column
        df_UJ = df_UJ[['Time'] + [col for col in df_UJ.columns if col != 'Time']] # Reorder columns to make 'Time' the first column

        #writing DJexit_SS results to dataframe
        df_SlJ = pd.DataFrame(SlJexit_SS.y.T, columns=[f'Node_{i}' for i in range(SlJexit_SS.y.shape[0])])
        df_SlJ['Time'] = SlJexit_SS.t #adds the time column
        df_SlJ = df_SlJ[['Time'] + [col for col in df_SlJ.columns if col != 'Time']] # Reorder columns to make 'Time' the first column
                
        #writing SJexit_SS results to dataframe
        df_RJ = pd.DataFrame(RJexit_SS.y.T, columns=[f'Node_{i}' for i in range(RJexit_SS.y.shape[0])])
        df_RJ['Time'] = RJexit_SS.t #adds the time column
        df_RJ = df_RJ[['Time'] + [col for col in df_RJ.columns if col != 'Time']] # Reorder columns to make 'Time' the first column
           
        with pd.ExcelWriter(os.path.join(csv_datafiles, "model_results_jej-Bryan.xlsx"), engine='openpyxl') as writer:
            df_UJ.to_excel(writer, sheet_name='Jejunum_undigesible', index=False)
            df_UP_j.to_excel(writer, sheet_name='Jejunum_undigesible_atexit', index=False)
            df_SlJ.to_excel(writer, sheet_name='Jejunum_slowly-digested', index=False)
            df_SlP_j.to_excel(writer, sheet_name='Jejunum_slowly-digested_atexit', index=False)
            df_RJ.to_excel(writer, sheet_name='Jejunum_rapidly-digested', index=False)
            df_RP_j.to_excel(writer, sheet_name='Jejunum_rapidly-digested_atexit', index=False)
            
    def export_il():
        UIexit_SS = ileum_instance.UIexit_SS
        SlIexit_SS = ileum_instance.SlIexit_SS
        RIexit_SS = ileum_instance.RIexit_SS

        df_UP_i = ileum_instance.df_UP_i
        df_SlP_i = ileum_instance.df_SlP_i
        df_RP_i = ileum_instance.df_RP_i
        
        #writing UIexit_SS results to dataframe
        df_UI = pd.DataFrame(UIexit_SS.y.T, columns=[f'Node_{i}' for i in range(UIexit_SS.y.shape[0])])
        df_UI['Time'] = UIexit_SS.t #adds the time column
        df_UI = df_UI[['Time'] + [col for col in df_UI.columns if col != 'Time']] # Reorder columns to make 'Time' the first column
        
        #writing DIexit_SS results to dataframe
        df_SlI = pd.DataFrame(SlIexit_SS.y.T, columns=[f'Node_{i}' for i in range(SlIexit_SS.y.shape[0])])
        df_SlI['Time'] = SlIexit_SS.t #adds the time column
        df_SlI = df_SlI[['Time'] + [col for col in df_SlI.columns if col != 'Time']] # Reorder columns to make 'Time' the first column
        
        #writing DIexit_SS results to dataframe
        df_RI = pd.DataFrame(RIexit_SS.y.T, columns=[f'Node_{i}' for i in range(RIexit_SS.y.shape[0])])
        df_RI['Time'] = RIexit_SS.t #adds the time column
        df_RI = df_RI[['Time'] + [col for col in df_RI.columns if col != 'Time']] # Reorder columns to make 'Time' the first column
        
        # df_total = pd.DataFrame({'time': RIexit_SS.t,
        #                          'UIexit_SS': UIexit_SS.y[-1], 
        #                          'SlIexit_SS': SlIexit_SS.y[-1], 
        #                          'RIexit_SS': RIexit_SS.y[-1],
        #                          })
      
        with pd.ExcelWriter(os.path.join(csv_datafiles, "model_results_ileum-Bryan.xlsx"), engine='openpyxl') as writer:
            df_UI.to_excel(writer, sheet_name='Ileum_undegradable', index=False)
            df_UP_i.to_excel(writer, sheet_name='Ileum_undegrable_atexit', index=False)
            df_SlI.to_excel(writer, sheet_name='Ileum_sl_degradable', index=False)
            df_SlP_i.to_excel(writer, sheet_name='Ileum_sl_degrable_atexit', index=False)
            df_RI.to_excel(writer, sheet_name='Ileum_rapid_degrad', index=False)
            df_RP_i.to_excel(writer, sheet_name='Ileum_rapid_degrad_atexit', index=False)
            #df_total.to_excel(writer, sheet_name='Ileum_total_atexit', index=False)

    export_duo()    
    export_jej()
    export_il()
    print("done exporting")

export_duojejil_results(duodenum_instance, jejunum_instance, ileum_instance)
    #-------------OPTIMIZATION for k_absp constant---------------------------------------#

def optimize_params(t_eval, t_span, ingr_name):
    #protein_data = constants['target_v']
    protein_data = constants['target_v_sdis']
    #protein_total_data = np.array([0, constants['target_v_total']]) #data array t=0, CPtotal=0; t=114-, CPtotal=X
    #protein_total_data = constants['target_v_total']

    #print(constants['target_v_total'])
    best_current = {'x':None, 'fun':np.inf}
    w1 = 1.0
    w2 = 1.0
    evaluation_count = 0
    last_sse = None

    def SSE_function(opt_params): #the 800 values
        nonlocal evaluation_count
        evaluation_count += 1
        print(f"Evaluation {evaluation_count}")

        k_absp = opt_params[0]
        k_digestrate = opt_params[1]
        Kp_endog_min = opt_params[2]
        #passing the 800 in here
        #with pool

        try:
        # This is where the ODE solving happens - wrap it in try-except
            duodenum_instance, jejunum_instance, ileum_instance = HindGIT(ingr_name, t_eval, t_span, constants, k_absp, k_digestrate, Kp_endog_min)
            
            # # Check if the solver succeeded and returned valid shapes
            # print(f"Shape of RIexit_SS.y: {ileum_instance.RIexit_SS.y.shape}")
            
            # # Check if we have enough time points (should be 2881)
            # if ileum_instance.RIexit_SS.y.shape[1] < 1441:  # 1440 + 1
            #     print(f"Solver terminated early: only {ileum_instance.RIexit_SS.y.shape[1]} time points")
            #     return 1e10  # Large penalty value

            # last_50ileum = ileum_instance.RIexit_SS.y[:, 1440][-50:] 
            # #[:, 1440][-50:] #1440 --> the 1440 timestep
            # sum_nodes = np.sum(last_50ileum)

            # last_50ileum_UI = ileum_instance.UIexit_SS.y[:, 1440][-50:]
            # sum_nodes_UI = np.sum(last_50ileum_UI) 
            
            # last_50ileum_SlI = ileum_instance.SlIexit_SS.y[:, 1440][-50:]
            # sum_nodes_SlI = np.sum(last_50ileum_SlI)

            sum_RI_flux = np.sum(ileum_instance.df_RP_i.iloc[0:2001, 2]) 
            #sum of flux not the node pool but across 0-2000min of the last node
            #print(sum_RI_flux)

            sum_UI_flux = np.sum(ileum_instance.df_UP_i.iloc[0:2001, 2])
            #print(sum_UI_flux)
            # sum_UI_flux = ileum_instance.df_UP_i.query('0 <= t <= 2000')['flux_CPu_Ileum_psg'].sum()
            
            sum_SlI_flux = np.sum(ileum_instance.df_SlP_i.iloc[0:2001, 2])
            #print(sum_SlI_flux)

            #sum_nodes_total = sum_nodes + sum_nodes_UI + sum_nodes_SlI 
            sum_flux_total = sum_RI_flux + sum_UI_flux + sum_SlI_flux #not digested fluxes for the last 24h +

            #pred_protein = sum_nodes
            pred_protein = (1 - (sum_flux_total/constants['FI_24h_gbird']))*100
            #V: Probably want to print this out ^

            #pred_protein_total = sum_nodes_total

            #print(pred_protein)
            SSE_protein = np.sum((pred_protein - protein_data )**2) #sum of squared errors
            #SSE_protein_total = np.sum((pred_protein_total - protein_total_data)**2)
            #total_SSE = w1*SSE_protein + w2*SSE_protein_total #w1 and w2 are weights, the same bc the optimizer seems to focus on SSE_protein more
            my_callback.last_sse = SSE_protein
            return SSE_protein
        
        except Exception as e:
            print(f"Solver failed at evaluation {evaluation_count}: {e}")
            print(f"Parameters that caused failure: k_absp={k_absp}, k_digestrate={k_digestrate}, Kp_endog_min={Kp_endog_min}")
            return 1e10  # Large penalty value
        
    best_current = {'x': None, 'fun': float('inf')}  
    if os.path.exists("best_solution.pkl"):
        with open("best_solution.pkl", "rb") as f:
            best_current = pickle.load(f)
    else:
        best_current = {'x':None, 'fun':np.inf}
    
    #V: Bounds in sequential order of the parameters to be optimized
    # k_absp, k_digestrate, Kp_endog_min
    #0, 0, 0.01
    # V: Look into differential evolution
    bounds_params = [(0, 100), (0, 0.1), (0, 0.1)]    
    opt_results = []

    def my_callback(xk, convergence=0):
            #xk is the best current parameter vector for each iteration, if its better than previous best it saves the xk and SSE to best_solution.pk1

        if hasattr(my_callback, 'last_sse') and my_callback.last_sse < best_current['fun']:
            best_current['x'] = xk.copy()
            best_current['fun'] = my_callback.last_sse
            
        with open('best_solution.pkl', 'wb') as f:
            pickle.dump(best_current, f)
        print(f"saved better solution: SSE={my_callback.last_sse}")
    

    #V:  One of the main methods.
    # This has roughly 50 different combos
    # Ideally we want popsize to 100, maxiter to 1000
    # This whole things should run 100 times, since the differential evolution changes each time
    # This diff_ev takes the longest
    Optimize_params_result = differential_evolution(SSE_function, bounds_params, popsize=3, maxiter=2, callback=my_callback) #L-BFGS-B
        #seed = random number for reproducibility, diff_evol can get diff result on diff runs
        #i.e. getting same sequence of random numbers each time code is run, any integer works and doesn't have a specific meaning
        #workers ---> for parallization i.e. running multiple computations at the same time using different CPU cores for faster results
        # the -1 in workers means to use all available CPU cores
       # result_local = minimize(SSE_function, x0=[result_global.x[0]], method='L-BFGS-B', bounds=[(1e-6, 1.0)])
        
    Optimized_params = Optimize_params_result.x.copy() #extracts the optimized result to be able to use it as a number
    k_absp_opt = Optimized_params[0]
    k_digestrate_opt = Optimized_params[1]
    Kp_endog_min_opt = Optimized_params[2]
    
    #V: Ideally we'd have 100 of these values & 100 per diet to choose the best result
    print("best K_absp_opt:", k_absp_opt)
    print("best K_digestrate_opt:", k_digestrate_opt)
    print("best K_endog_opt:", Kp_endog_min_opt)
    #V:add in printing out the digestability it's predicting (Plus pred protein in a document that gets updated each 100)
    SSE_value = Optimize_params_result.fun

    #tracking best result with lowest SSE
    if SSE_value < best_current['fun']:
        best_current['fun'] = SSE_value
        best_current['x'] = Optimized_params.copy()
    
    result_row = [k_absp_opt, k_digestrate_opt, Kp_endog_min_opt]
    opt_results.append(result_row)

    # Optimize_params_result.nit #the number of iterations
    # Optimize_params_result.nfev #the nnmber of evaluations

    #refining best value after 100 loop
    #V: This might not necessarily be needed. We already have our "best results"
    # So print it out and save it to excel
    if best_current['x'] is not None:
        print("Refining best result")
        k_absp_opt = best_current['x'][0]
        k_digestrate_opt = best_current['x'][1]
        Kp_endog_min_opt = best_current['x'][2]

        duodenum_2, jejunum_2, ileum_2 = HindGIT(ingr_name, t_eval, t_span, constants, k_absp_opt, k_digestrate_opt, Kp_endog_min_opt)

        # UDexit_SS = duodenum_2.UDexit_SS
        # SlDexit_SS = duodenum_2.SlDexit_SS
        # RDexit_SS = duodenum_2.RDexit_SS
        
        # UJexit_SS = jejunum_2.UJexit_SS
        # SlJexit_SS = jejunum_2.SlJexit_SS
        # RJexit_SS = jejunum_2.RJexit_SS
        
        # last_50Il = ileum_2.RIexit_SS.y[:, 1440][-50:] 
        # last50_sum_g_value = np.sum(last_50Il)

        # last_50Il_UI = ileum_2.UIexit_SS.y[:, 1440][-50:] 
        # last_50_UI_sum_g_value = np.sum(last_50Il_UI)

        # last_50Il_SlI = ileum_2.SlIexit_SS.y[:, 1440][-50:] 
        # last_50_SlI_sum_g_value = np.sum(last_50Il_SlI)

        # last_50_sum_total_g_value = last50_sum_g_value + last_50_UI_sum_g_value + last_50_SlI_sum_g_value
        #print(f"[{i+1}] g at 1440:", last50_sum_g_value)
        #print("g at 1440:", UIexit_SS.y.T[1440, -1])
        
        result_row = [k_absp_opt] + [k_digestrate_opt] + [Kp_endog_min_opt] #+ [last50_sum_g_value] + [last_50_UI_sum_g_value] + [last_50_SlI_sum_g_value] + [last_50_sum_total_g_value]
        
        opt_results.append(result_row)
        column_names = [f"k_absp_{ingr_name}"] + [f"k_digestrate_{ingr_name}"] + [f"k_endog_{ingr_name}"] #+ ["last_50RI_at_1440"] + ["last_50UI_at_1440"] + ["last_50SlI_at_1440"] + ["last_50_total_at_1440"]
        df_opt_results = pd.DataFrame(opt_results, columns=column_names)
       
        # Save to Excel
    
        #df_opt_results.to_excel(os.path.join(csv_datafiles, "optimization_results.xlsx"), index=False)
        df_opt_results.to_excel("optimization_results.xlsx", index=False)
    
        print("All results saved to 'optimization_results.xlsx'")

        return (k_absp_opt, k_digestrate_opt, Kp_endog_min_opt, best_current['x'], duodenum_2, jejunum_2, ileum_2) 
                #last50_sum_g_value, last_50_UI_sum_g_value, last_50_SlI_sum_g_value, last_50_sum_total_g_value)
        
#     # #----------Plots results from duodenum node 0----------------------#
#     # # plt.clf() 
#     # # plt.plot(self.result_UDnode0.t, self.result_UDnode0.y[0], color='blue', label='Undigestible Protein_node0duo g')
#     # # plt.plot(self.result_SlDnode0.t, self.result_SlDnode0.y[0], color='red', label='Slowly-digested Protein_node0duo g')
#     # # plt.plot(self.result_RDnode0.t, self.result_RDnode0.y[0], color='green', label='Rapidly-digested Protein_node0duo g')
#     # # plt.legend()
#     # # plt.xlabel('Time')
#     # # plt.ylabel('Protein (g) of duodenum')
#     # # plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
#     # # plt.savefig('images/B-protein-over-time-node0-duodenum.png')

optimize_params(t_eval, t_span, ingr_name)
print ("the end")

