
import math 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from chickgut.utils.performance import time_it

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
        self.Discretize_il = 101    # number of sections to split volume into jejunum (# of nodes)
        self.Node_num = self.Discretize_il - 1
        self.length_cm = 34.643*self.BWeight_kgb # ileum length (cm) from: Novotny et al. 2023, averaged across coarse/medium/fine diets 
        self.r_cm = 0.9/2                   #ileum radius (cm), ileum diameter/2 = radius from: steczny and kokosynski 2019 - 42 DOA 
        self.volume_il_cm3 = math.pi*math.pow(self.r_cm, 2)*self.length_cm # volume of ileum (cm^3)
        
        self.IlV_cm3 = np.linspace(0, self.volume_il_cm3, self.Discretize_il)
        self.Il_single_node_V = self.volume_il_cm3 / self.Node_num
        self.MRT_min = self.constants['Il_MRT'] #Bryan et al. 2019 in vivo MRT
        MRT_Il_CPu_min = 49.5 #P5 = 49.5

        self.Kp_Il_min =  1/self.MRT_min # fractional (SOLID?) passage rate out of the ileum (/min), from meta-A , inverse mrt 1/95.094
        self.Kp_Il_CPu_min = 1/MRT_Il_CPu_min # fractional (SOLID) passage rate out of the ileum (/min), from meta-A P5 - for undigestible protein
        
        self.VF = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/self.MRT_min #ileum volumetric flow rate
        self.VF_Il_CPu = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/MRT_Il_CPu_min
        #VF_Co = (math.pi*math.pow(co.r_cm, 2))*co.length_cm/MRT_Il_CPu_min
        self.init_Il_CPu = np.zeros(self.Discretize_il) #starts all nodes at 0
        self.init_Il_CPsl = np.zeros(self.Discretize_il)
        self.init_Il_CPr = np.zeros(self.Discretize_il)
                

    def calculate_Il_endog(self):
        SI_volume_cm3 = self.volume_cm3 + self.volume_jej_cm3 + self.volume_il_cm3

        self.vol_prop = self.volume_il_cm3 / SI_volume_cm3
        #print(self.vol_prop)
        
        Il_BasalAA = 0.799 #g of basal ileal endogenous protein/g of CP from: Ravindran 2021, total of table 1 - i.e. total endogenous of SI
        self.basalaa = Il_BasalAA * self.vol_prop # basal endogenous AA input for jejunum based on proportion of volume

        Ileum_basalaa_single_node = self.basalaa/(self.Discretize_il - 1) # total jejunum basal aa divided by number of nodes 
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

    #-----------undigestible protein ileum-------------------#
    def method_of_lines_Ileum_CPu(self, t, QCPu_Il_g):
        # Find the index in pvg_values that corresponds to the closest time in df_JEJ
        # t is assumed to be a time value that matches the time points in df_JEJ
        #index_il_CPu = np.searchsorted(self.UJexit_SS.t, t, side='left')  # Find index for the time point

        #plug flow equations - Kitchin Group
        QCPu_Il_g[0] = self.UJexit_SS.sol(t)[-1]

        if QCPu_Il_g.all() < 0:
            QCPu_Il_g = 1e-10

        U_CPu_Il_psg_gmincm3 = self.VF_Il_CPu * np.diff(QCPu_Il_g) / np.diff(self.IlV_cm3)
        
            
        P_CPu_JejIl_gmin = self.Kp_Il_min*self.UJexit_SS.sol(t)[-1] #Production pool from the jejunum to the ileum
        P_CPu_JejIl_gmincm3 = P_CPu_JejIl_gmin/self.CP_jej_lastnode_vol_cm3
        
        Il_UD_Diff = self.P_CPe_Il_gmincm3 + P_CPu_JejIl_gmincm3 - U_CPu_Il_psg_gmincm3
                    #g/min*cm3                                      #g/min*cm3    
        dUIdt = np.concatenate([[0], Il_UD_Diff])
        
        flux_CPu_Ileum_psg = (QCPu_Il_g/self.Il_single_node_V)*self.Kp_Il_min
                                    #g/min*cm3
        
        #last_50_UI = np.sum(QCPu_Il_g[-50:])
        
        return dUIdt, flux_CPu_Ileum_psg #, last_50_UI
    #print(f"Shape of y0: {init_UD.shape}")
    
    def Solve_method_of_lines_Ileum_CPu(self, t, QCPu_Il_g):
        dUIdt, _ = self.method_of_lines_Ileum_CPu(t, QCPu_Il_g)
        return dUIdt
    
    #-----------slowly-digested protein ileum-------------------#
    def method_of_lines_Ileum_CPsl(self, t, QCPsl_Il_g):
        # Find the index in pvg_values that corresponds to the closest time in df_JEJ
        # t is assumed to be a time value that matches the time points in df_JEJ
        #index_il_CPsl = np.searchsorted(self.SlJexit_SS.t, t, side='left')  # Find index for the time point

        #plug flow equations
        QCPsl_Il_g[0] = self.SlJexit_SS.sol(t)[-1]

        if QCPsl_Il_g.all() < 0:
            QCPsl_Il_g = 1e-10
        
        #---flux from Ileum slowly-digested to Ileum rapidly-digested---#
        Il_CPsl_dis_gmincm3 = self.k_digestrate * (QCPsl_Il_g[1:]**2 /self.Il_single_node_V) 
        
        U_CPsl_Il_psg_gmincm3 = self.VF * np.diff(QCPsl_Il_g) / np.diff(self.IlV_cm3)
        U_CPsl_Il_psg_dis_gmincm3 = U_CPsl_Il_psg_gmincm3 + Il_CPsl_dis_gmincm3
        
        #---flux from jejunum to ileum---#
        P_CPsl_JejIl_gmin = self.Kp_Jej_min*self.SlJexit_SS.sol(t)[-1] #Production pool from the jejunum to the ileum
        P_CPsl_JejIl_gmincm3 = P_CPsl_JejIl_gmin/self.CP_jej_lastnode_vol_cm3
        
        Il_SlD_Diff = P_CPsl_JejIl_gmincm3 - U_CPsl_Il_psg_dis_gmincm3
                    #self.P_CPe_Il_gmincm3 g/min*cm3                                      #g/min*cm3    
        dSlIdt = np.concatenate([[0], Il_SlD_Diff])
        
        flux_CPsl_Ileum_psg_dis = self.Kp_Il_min*(QCPsl_Il_g/self.Il_single_node_V)

        #last_50_SlI = np.sum(QCPsl_Il_g[-50:])
        
        return dSlIdt, flux_CPsl_Ileum_psg_dis #, last_50_SlI

    def Solve_method_of_lines_Ileum_CPsl(self, t, QCPsl_Il_g):
        dSlIdt, _ = self.method_of_lines_Ileum_CPsl(t, QCPsl_Il_g)
        return dSlIdt
    
    @time_it
    def solving_il_USl(self):

        self.init_Il_CPu[0] = self.UJexit_SS.y.T[0, -1]/self.Il_single_node_V
        
        
        self.init_Il_CPsl[0] = self.SlJexit_SS.y.T[0, -1]/self.Il_single_node_V
        
        # This solves differentials which can take awhile
        self.UIexit_SS = solve_ivp(self.Solve_method_of_lines_Ileum_CPu, self.t_span, y0=self.init_Il_CPu, t_eval=self.t_eval, 
                                   dense_output=True, method='Radau')
        self.SlIexit_SS = solve_ivp(self.Solve_method_of_lines_Ileum_CPsl, self.t_span, y0=self.init_Il_CPsl, t_eval=self.t_eval, 
                                    dense_output=True, method='Radau')

    #----prepping data for next compartment undigestible protein----#
    def flatten_result_il_CPu(self):
        flattened_data_il_CPu = []

        for t, y in zip(self.UIexit_SS.t, self.UIexit_SS.y.T):
            dUIdt, flux_CPu_Ileum_psg = self.method_of_lines_Ileum_CPu(t, y)
            QCPu_Il_g=y[-1]
            
            flattened_entry_il_CPu={
                't': t, 
                'dUIdt': dUIdt[-1],
                'flux_CPu_Ileum_psg': flux_CPu_Ileum_psg[-1],
                #'last_50_UI':last_50_UI,
                'CPu_il': QCPu_Il_g,
            }
            
            flattened_data_il_CPu.append(flattened_entry_il_CPu)
        self.df_UP_i = pd.DataFrame(flattened_data_il_CPu)
        #print(f"df_ileum, {df_il}")
        return self.df_UP_i, flattened_data_il_CPu
    
    # df_UP_i, _ = flatten_result_il_CPu()

    # df_UP_IL = pd.DataFrame(df_UP_i)

    #----prepping data for next compartment slowly-digested protein----#
    def flatten_result_il_CPsl(self):
        flattened_data_il_CPsl = []

        for t, y in zip(self.SlIexit_SS.t, self.SlIexit_SS.y.T):
            dSlIdt, flux_CPsl_Ileum_psg_dis = self.method_of_lines_Ileum_CPsl(t, y)
            QCPsl_Il_g=y[-1]
            
            flattened_entry_il_CPsl={
                't': t, 
                'dSlIdt': dSlIdt[-1],
                'flux_CPsl_Ileum_psg_dis': flux_CPsl_Ileum_psg_dis[-1],                #'last_50_SlI': last_50_SlI,
                'CPsl_il': QCPsl_Il_g,
            }
            
            flattened_data_il_CPsl.append(flattened_entry_il_CPsl)
        self.df_SlP_i = pd.DataFrame(flattened_data_il_CPsl)
        #print(f"df_ileum, {df_il}")

        return self.df_SlP_i, flattened_data_il_CPsl

    # df_SlP_i, _ = flatten_result_il_CPsl()

    # df_SlP_IL = pd.DataFrame(df_SlP_i)
    
    #------slowly-digested to rapidly-digested protein in jejunum setup------------#

    def SlP_for_RP_i(self):
        self.flatten_result_il_CPsl()
        df_Q_CPsl_Il = self.df_SlP_i[['t', 'CPsl_il']] #pulling out just the relavent columns from the df_SlP dataframe
        df_Q_CPsl_Il.reset_index(drop=True, inplace=True) #resets the index numbers of df_PVG to default integer index (0,1,2,3...)
        Il_CPsl = df_Q_CPsl_Il['CPsl_il'].values  
        return df_Q_CPsl_Il, Il_CPsl
    
    #------------------------rapidly-digested protein duodenum---------------------------------------------------#
    def method_of_lines_CPr_Il(self, t, QCPr_Il_g):
        # Find the index in pvg_values that corresponds to the closest time in df_DUO
        # t is assumed to be a time value that matches the time points in df_DUO
        #index_il_CPr = np.searchsorted(self.RJexit_SS.t, t, side='left')  # Find index for the time point based on previous compartment
        index_il_SlR_flux = min(np.searchsorted(self.df_Q_CPsl_Il['t'], t, side='left'), len(self.df_Q_CPsl_Il['t']) - 1) #for slowly-digested to rapid flux
        
        QCPr_Il_g[0] = self.RJexit_SS.sol(t)[-1]
        
        if QCPr_Il_g.all() < 0:
            QCPr_Il_g = 1e-10
            
        #QCPr_Il_g = np.maximum(QCPr_Il_g , 1e-12)
        
        Il_CPr_dis_gmincm3 = (self.k_absp*QCPr_Il_g[1:]**2)/self.Il_single_node_V      #  rapid CP -> rapid CP flux (g/min)

        U_CPr_Il_psg_gmincm3 = self.VF * np.diff(QCPr_Il_g) / np.diff(self.IlV_cm3)     # calculates amount of protein at each node
            # g/min*cm^3  
            
        U_CPr_Il_psg_dis_gmincm3 = np.maximum(U_CPr_Il_psg_gmincm3 + Il_CPr_dis_gmincm3, 1e-12)
        
        #---flux from Jej to Il---#
        P_CPr_JejIl_gmin = self.Kp_Jej_min*self.RJexit_SS.sol(t)[-1]
        P_CPr_JejIl_gmincm3 = P_CPr_JejIl_gmin/self.Il_single_node_V

        #---flux from Il slowly-digested to Il rapidly-digested---#
        P_CPr_IlSlIlR_gmin = self.k_digestrate*(self.Il_CPsl[index_il_SlR_flux])
        P_CPr_IlSlIlR_gmincm3 = P_CPr_IlSlIlR_gmin/self.Il_single_node_V

        Il_RD_Diff = P_CPr_JejIl_gmincm3 + P_CPr_IlSlIlR_gmincm3 - np.minimum(U_CPr_Il_psg_dis_gmincm3, QCPr_Il_g[1:]/self.Il_single_node_V)
                    # g/min*cm3         #g/min*cm^3
        dRIdt = np.concatenate([[0], Il_RD_Diff])
        
        flux_CPr_Il_psg_dis = (QCPr_Il_g/self.Il_single_node_V)*self.Kp_Il_min
        
        #last_50_RI = np.sum(QCPr_Il_g[-50:])

        return dRIdt, flux_CPr_Il_psg_dis 
        #, last_50_RI
    
    def Solve_method_of_lines_CPr_Il(self, t, QCPr_Il_g):
        dRIdt, _ = self.method_of_lines_CPr_Il(t, QCPr_Il_g)
        return dRIdt
    
    @time_it
    def solving_il_R(self):
        self.df_Q_CPsl_Il, self.Il_CPsl = self.SlP_for_RP_i()

        self.init_Il_CPr[0] = self.RJexit_SS.y.T[0, -1]/self.Il_single_node_V

        self.RIexit_SS = solve_ivp(self.Solve_method_of_lines_CPr_Il, self.t_span, y0=self.init_Il_CPr, t_eval=self.t_eval, 
                                   dense_output=True, method='LSODA')
    
    #----prepping data for next compartment - undigestible----#  
    def flatten_result_il_CPr(self):
        flattened_data_il_CPr = []

        for t, y in zip(self.RIexit_SS.t, self.RIexit_SS.y.T):
            dRIdt, flux_CPr_Il_psg_dis = self.method_of_lines_CPr_Il(t, y)
            QCPr_Il_g=y[-1]
            
            flattened_entry_il_CPr ={
                't': t, 
                'dRIdt': dRIdt[-1],
                'flux_CPr_Ileum_psg_dis': flux_CPr_Il_psg_dis[-1], #'last_50_RI': last_50_RI,
                'CPr_il': QCPr_Il_g,
            }
            flattened_data_il_CPr.append(flattened_entry_il_CPr)
        self.df_RP_i = pd.DataFrame(flattened_data_il_CPr)
        #print(f"df_duodenum, {df_s}")
        return self.df_RP_i, flattened_data_il_CPr
        #--------Plots results for Ileum-----------------#
    
    def feed_il(self, t, Qfeed_Il_g):

        Q_feedJej_g_at_t = self.result_feed_jej.sol(t)
        
        if Qfeed_Il_g.all() < 0.0:
            Qfeed_Il_g = 1e-10
            
        P_Il_feed = Q_feedJej_g_at_t*self.Kp_Jej_min
        U_IlCo_feed = Qfeed_Il_g*self.Kp_Il_min
        
        dfeedildt = P_Il_feed - U_IlCo_feed
        return dfeedildt, U_IlCo_feed, Qfeed_Il_g
    
    def solve_feed_il(self, t, Qfeed_Il_g):
        dfeedildt, _, _ = self.feed_il(t, Qfeed_Il_g)
        return dfeedildt
    
    @time_it
    def solving_il_feed(self):
        self.iIl_g = [0.00000000001]
        self.result_feed_il = solve_ivp(self.solve_feed_il, self.t_span, y0=self.iIl_g, t_eval=self.t_eval, dense_output=True, method='RK45')
    
    def flatten_result_il_feed(self):
        flattened_data_il_feed = []

        for t, y in zip(self.result_feed_il.t, self.result_feed_il.y.T):
            dfeedildt, U_IlCo_feed, Qfeed_Il_g = self.feed_il(t, y)
            Qfeed_Il_g = y
            flattened_entry_il_feed ={
                't': t, 
                'dfeedildt': dfeedildt,
                'flux_feed_IlCo': U_IlCo_feed,
                'Qfeed_il': Qfeed_Il_g,
            }
            flattened_data_il_feed.append(flattened_entry_il_feed)
        self.df_feed_i= pd.DataFrame(flattened_data_il_feed)
        #print(f"df_duodenum, {df_s}")
        return self.df_feed_i, flattened_data_il_feed 
    

    def plot_il(self):
        plt.clf() 
        plt.plot(self.df_UP_i['t'], self.df_UP_i['CPu_il'], color='blue') #, label='undigestible_il g'
        plt.plot(self.df_SlP_i['t'], self.df_SlP_i['CPsl_il'], color='red') #, label='slowly-digested_il g'
        plt.plot(self.df_RP_i['t'], self.df_RP_i['CPr_il'], color='green') #, label='rapidly-digested_il g'
        #plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
        plt.xlabel('Time')
        plt.ylabel('Protein (g) in the ileum')
        plt.tight_layout(pad=2.0)
        plt.savefig('images/B-protein-over-time-at-last-node-of-ileum.png', bbox_inches='tight')
        
        plt.clf() 
        plt.plot(self.df_UP_i['t'], self.df_UP_i['flux_CPu_Ileum_psg'], color='blue')#, label='Undigestible Protein_Ileum flux'
        plt.plot(self.df_SlP_i['t'], self.df_SlP_i['flux_CPsl_Ileum_psg_dis'], color='red')#, label='Slowly-digested Protein_Ileum flux'
        plt.plot(self.df_RP_i['t'], self.df_RP_i['flux_CPr_Ileum_psg_dis'], color='green')#, label='Rapidly-digested Protein_Ileum flux'
        #plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
        plt.xlabel('Time')
        plt.ylabel('Flux of protein (g/min) at the end of the ileum')
        plt.tight_layout(pad=2.0)
        plt.savefig('images/B-protein-over-time-at-last-node-of-Ileum_flux.png', bbox_inches='tight')
        
        plt.clf() 
        plt.plot(self.df_feed_i['t'], self.df_feed_i['Qfeed_il'], color='red', label='Q feed ileum')
        plt.plot(self.df_feed_i['t'], self.df_feed_i['flux_feed_IlCo'], color='blue', label='Flux feed ileum')
        #plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
        plt.xlabel('Time')
        plt.ylabel('Feed (g) and Flux (g/min) in the ileum')
        plt.tight_layout(pad=2.0)
        plt.savefig('images/B-feed-over-time-ileum.png', bbox_inches='tight')
        
