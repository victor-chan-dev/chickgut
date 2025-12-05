
import math 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


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
        self.length_cm = 33.166*self.BWeight_kgb  # jejunum length (cm) from: Novotny et al. 2023, averaged across coarse/medium/fine diets
        self.r_cm = 1.14/2                   # jejunum radius (cm), jejunum diameter/2 = radius from: steczny and kokosynski 2019 - 42 DOA
        self.volume_jej_cm3 = math.pi*math.pow(self.r_cm, 2)*self.length_cm  #jejunum volume (cm^3)

        self.Discretize_jej = 101    # number of sections to split volume into jejunum (# of nodes)
        self.Node_num_jej = self.Discretize_jej - 1
        self.JejV_cm3 = np.linspace(0, self.volume_jej_cm3, self.Discretize_jej)
        self.Jej_single_node_V = self.volume_jej_cm3/self.Node_num_jej

        MRT_Jej_CPu_min = 16.0 #P5 = 27.0; P1 = 16.0
        self.MRT_Jej_min = self.constants['Jej_MRT']

        self.Kp_Jej_min = 1/self.MRT_Jej_min # fractional (SOLID) passage rate out of the jejunum (/min), from meta-A , inverse mrt 1/52.066667
        self.Kp_Jej_CPu_min = 1/MRT_Jej_CPu_min # fractional (SOLID) passage rate out of the jejunum (/min), from meta-A - for undigestible protein  
        
        self.VF = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/self.MRT_Jej_min #volumetric flow rate
        self.VF_CPu = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/MRT_Jej_CPu_min #volumetric flow rate
        self.init_Jej_CPu = np.zeros(self.Discretize_jej) #starts all nodes at 0
        self.init_Jej_CPsl = np.zeros(self.Discretize_jej)
        self.init_Jej_CPr = np.zeros(self.Discretize_jej)


    def calculate_jej_endog(self):
        SI_volume_cm3 = self.volume_cm3 + self.volume_jej_cm3 + self.volume_il_cm3
    
        self.vol_prop = self.volume_jej_cm3 / SI_volume_cm3
        
        Il_BasalAA = 0.799 #g of basal ileal endogenous protein/g of CP from: Ravindran 2021, total of table 1 - i.e. total endogenous of SI
        self.basalaa = Il_BasalAA * self.vol_prop # basal endogenous AA input for jejunum based on proportion of volume

                #---endogenous protein from ileum-----#
        self.basalaa_single_node = self.basalaa/(self.Discretize_jej - 1) # total jejunum basal aa divided by number of nodes 
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
       

    #------------------------------------undigestible protein jejunum---------------------#
    def method_of_lines_Jej_CPu(self, t, QCPu_Jej_g):

        #index_CPu_DuoJej = np.searchsorted(self.UDexit_SS.t, t, side='left')
        #plug flow equation - Kitchin Group
            # no dis equation because undigestible therefore no conversion to rapidly-digested protein
        
        #quantity in the last node of duodenum is the same as first node of the jejunum
        #fetching data from the timestep (row) index_CPu_DuoJej and from the last column i.e. last node of duo
        QCPu_Jej_g [0] = self.UDexit_SS.sol(t)[-1] 

        if QCPu_Jej_g.all() < 0:
            QCPu_Jej_g = 1e-10

        U_CPu_Jej_psg_gmincm3 = self.VF_CPu * np.diff(QCPu_Jej_g) / np.diff(self.JejV_cm3) #calculates amount of protein at each node
            #g/min*cm^3
            
        #---flux from duodenum to jejunum---#

        P_CPu_DuoJej_gmin = self.Kp_Jej_min*self.UDexit_SS.sol(t)[-1] #last node value per timepoint
        P_CPu_DuoJej_gmincm3 = P_CPu_DuoJej_gmin/self.Duo_single_node_V

        Jej_UD_Diff = self.P_CPe_Jej_gmincm3 + P_CPu_DuoJej_gmincm3 - U_CPu_Jej_psg_gmincm3
                                            # g/min*cm3                                                                                    
        dUJdt = np.concatenate([[0], Jej_UD_Diff])
        
        flux_CPu_Jej_psg_dis = (QCPu_Jej_g[-1]/self.Jej_single_node_V)* self.Kp_Jej_CPu_min

        
        return dUJdt, flux_CPu_Jej_psg_dis

    #print(f"Shape of y0: {init_UD.shape}")
    def Solve_method_of_lines_Jej_CPu(self, t, QCPu_Jej_g):
        dUJdt, _ = self.method_of_lines_Jej_CPu(t, QCPu_Jej_g)
        return dUJdt

    #---------------------digestible protein jejunum-------------------------------------------#
    def method_of_lines_Jej_CPsl(self, t, QCPsl_Jej_g):
        #index_CPsl_Q_time = np.searchsorted(Duo_CPsl_time, t, side='left')
        #index_jej_CPsl = np.searchsorted(duo_time, t)  # Find index for the time point based on previous compartment
        #index_CPsl_DuoJej = np.searchsorted(self.SlDexit_SS.t, t, side='left')

        QCPsl_Jej_g[0] = self.SlDexit_SS.sol(t)[-1]
        
        if QCPsl_Jej_g.all() < 0:
            QCPsl_Jej_g = 1e-10
            
        #---flux from Jej slowly-digested to Jej rapidly-digested 
        Jej_CPsl_dis_gmincm3 = (self.k_digestrate* QCPsl_Jej_g[1:]**2)/self.Jej_single_node_V
            
        #plug flow equation - Kitchin Group
        U_CPsl_Jej_psg_min = self.VF * np.diff(QCPsl_Jej_g) / np.diff(self.JejV_cm3) #calculates amount of protein at each node
        #g/min*cm^3
    
        U_CPsl_Jej_psg_dis_gmincm3 = U_CPsl_Jej_psg_min + Jej_CPsl_dis_gmincm3
            
        #---flux from duo to jej---#
        #P_CPsl_DuoJej_gmin = Kp_Duo_min*CPsl_duo_lastnode_g[index_CPsl_Q_time]
        P_CPsl_DuoJej_gmin = self.Kp_Duo_min*self.SlDexit_SS.sol(t)[-1]
        P_CPsl_DuoJej_gmincm3 = P_CPsl_DuoJej_gmin/self.Jej_single_node_V
            
        #---flux from jej slowly-digested to jej rapidly-digested---#
        # Jej_CPd_dis_lastnode = Jej_CPd_dis[-1] 
        # U_CPd_JejDJejS_gmin = constants['k_dr_DP_min']*(Jej_CPd_dis_lastnode*jej_enzymes)

        Jej_SlD_Diff = P_CPsl_DuoJej_gmincm3 - U_CPsl_Jej_psg_dis_gmincm3
                                            # g/min*cm3                                #g/min*cm3     # changes everything beyond node 0                                                          
        #self.P_CPe_Jej_gmincm3 

        dSlJdt = np.concatenate([[0], Jej_SlD_Diff])
        flux_CPsl_Jej_psg_dis = (QCPsl_Jej_g[-1]/self.Jej_single_node_V)*self.Kp_Jej_min
                                # g/min*cm3
        return dSlJdt, flux_CPsl_Jej_psg_dis

    def Solve_method_of_lines_Jej_CPsl(self, t, QCPsl_Jej_g):
        dDJdt, _ = self.method_of_lines_Jej_CPsl(t, QCPsl_Jej_g)
        return dDJdt

    def solving_jej_USl(self):

        self.init_Jej_CPu[0] = self.UDexit_SS.y.T[0, -1]/self.Jej_single_node_V

        self.init_Jej_CPsl[0] = self.SlDexit_SS.y.T[0, -1]/self.Jej_single_node_V

    #---------Solves for value of jejunum crude protein (CP, g) using solve_ivp-----------------#
        self.UJexit_SS = solve_ivp(self.Solve_method_of_lines_Jej_CPu, self.t_span, y0=self.init_Jej_CPu, t_eval=self.t_eval, 
                                   dense_output=True, method='Radau')
        self.SlJexit_SS = solve_ivp(self.Solve_method_of_lines_Jej_CPsl, self.t_span, y0=self.init_Jej_CPsl, t_eval=self.t_eval, 
                                    dense_output=True, method='Radau')
        
    #----prepping data for next compartment - undigestible----#
    def flatten_result_jej_CPu(self):
        flattened_data_jej_CPu = []
        for t, y in zip(self.UJexit_SS.t, self.UJexit_SS.y.T):
            dUJdt, flux_CPu_Jej_psg_dis = self.method_of_lines_Jej_CPu(t, y)

            QCPu_Jej_g=y[-1]
            
            flattened_entry_jej ={
                't': t, 
                'dUJdt': dUJdt[-1],
                'flux_CPu_Jej_psg': flux_CPu_Jej_psg_dis,
                'CPu_jej': QCPu_Jej_g,
            }
            
            flattened_data_jej_CPu.append(flattened_entry_jej)
        self.df_UP_j = pd.DataFrame(flattened_data_jej_CPu)
        #print(f"df_duodenum, {df_jej}")
        return self.df_UP_j, flattened_data_jej_CPu

    # df_UP_j, _ = flatten_result_jej_CPu()
    # df_UP_JEJ = pd.DataFrame(df_UP_j)

    #----prepping data for next compartment - slowly-digestible----#
    def flatten_result_jej_CPsl(self):
        flattened_data_jej_CPsl = []
        for t, y in zip(self.SlJexit_SS.t, self.SlJexit_SS.y.T):
            dSlJdt, flux_CPsl_Jej_psg_dis = self.method_of_lines_Jej_CPsl(t, y)
            QCPsl_Jej_g = y[-1]
            
            flattened_entry_jej_CPsl ={
                't': t, 
                'dSlJdt': dSlJdt[-1],
                'flux_CPsl_Jej_psg_dis': flux_CPsl_Jej_psg_dis,
                'CPsl_jej': QCPsl_Jej_g,
            }
            
            flattened_data_jej_CPsl.append(flattened_entry_jej_CPsl)
        self.df_SlP_j = pd.DataFrame(flattened_data_jej_CPsl)
        #print(f"df_duodenum, {df_jej}")
        return self.df_SlP_j, flattened_data_jej_CPsl

    #------slowly- to rapidly-digested protein in jejunum setup------------#
    def SlP_for_RP_j(self):
        self.flatten_result_jej_CPsl()
        df_Q_CPsl_Jej = self.df_SlP_j[['t', 'CPsl_jej']] #pulling out just the relavent columns from the df_cpvg dataframe
        df_Q_CPsl_Jej.reset_index(drop=True, inplace=True) #resets the index numbers of df_PVG to default integer index (0,1,2,3...)
        Jej_CPsl = df_Q_CPsl_Jej['CPsl_jej'].values 
        return df_Q_CPsl_Jej, Jej_CPsl
    
    #------------------------rapidly-digested protein duodenum---------------------------------------------------#
    def method_of_lines_CPr_Jej(self, t, QCPr_Jej_g):

        #index_jej_CPr = np.searchsorted(duo_time, t, side='left')  # Find index for the time point based on previous compartment
        #index_CPr_DuoJej = np.searchsorted(self.RDexit_SS.t, t, side='left')
        index_jej_SlR_flux = np.searchsorted(self.df_Q_CPsl_Jej['t'], t, side='left')
        
        QCPr_Jej_g[0] = self.RDexit_SS.sol(t)[-1]

        QCPr_Jej_g = np.maximum(QCPr_Jej_g , 1e-12)

        #----flux from Jej Rapid to Absorbed----#
        Jej_CPr_dis_gmincm3 = self.k_absp*(QCPr_Jej_g[1:]**2/self.Jej_single_node_V)    #  soluble CP -> soluble CP flux (g/min)

        #----plug flow----#
        U_CPr_Jej_psg_mincm3 = self.VF * np.diff(QCPr_Jej_g) / np.diff(self.JejV_cm3)     # calculates amount of protein at each node
        # g/min*cm^3  

        U_CPr_Jej_psg_dis_gmincm3 = np.maximum(U_CPr_Jej_psg_mincm3 + Jej_CPr_dis_gmincm3, 1e-12)
                   #---flux from Duo to Jej---#
    
        P_CPr_DuoJej_gmin = self.Kp_Duo_min*self.RDexit_SS.sol(t)[-1]
        P_CPr_DuoJej_gmincm3 = P_CPr_DuoJej_gmin/self.Duo_single_node_V

        #---flux from Jej slow to Jej Rapid---#
        P_CPr_JejDJejS_gmin = self.k_digestrate*(self.Jej_CPsl[index_jej_SlR_flux])
        P_CPr_JejDJejS_gmincm3 = P_CPr_JejDJejS_gmin/self.Jej_single_node_V
        
        Jej_RD_Diff = P_CPr_DuoJej_gmincm3 + P_CPr_JejDJejS_gmincm3 - np.minimum(U_CPr_Jej_psg_dis_gmincm3, QCPr_Jej_g[1:]/self.Jej_single_node_V)
                    # g/min*cm3         #g/min*cm^3
        dRJdt = np.concatenate([[0], Jej_RD_Diff])
        
        flux_CPr_Jej_psg_dis = (QCPr_Jej_g[-1]/self.Jej_single_node_V)*self.Kp_Jej_min
        return dRJdt, flux_CPr_Jej_psg_dis

    def Solve_method_of_lines_CPr_Jej(self, t, QCPr_Jej_g):
        dRDdt, _ = self.method_of_lines_CPr_Jej(t, QCPr_Jej_g)
        return dRDdt
                
    #print(f"Shape of y0: {init_UD.shape}")
    def solving_jej_R(self):
        self.df_Q_CPsl_Jej, self.Jej_CPsl = self.SlP_for_RP_j()
        self.init_Jej_CPr[0] = self.RDexit_SS.y.T[0, -1]/self.Jej_single_node_V
            
        self.RJexit_SS = solve_ivp(self.Solve_method_of_lines_CPr_Jej,self.t_span, y0=self.init_Jej_CPr, t_eval=self.t_eval, 
                                   dense_output=True, method='LSODA')

    #----prepping data for next compartment - undegradable----#  
    def flatten_result_jej_CPr(self):
        flattened_data_jej_CPr = []

        for t, y in zip(self.RJexit_SS.t, self.RJexit_SS.y.T):
            dRJdt, flux_CPr_Jej_psg_dis = self.method_of_lines_CPr_Jej(t, y)
            QCPr_Jej_g = y[-1]
            
            flattened_entry_jej_CPr ={
                't': t, 
                'dRJdt': dRJdt[-1],
                'flux_CPr_Jej_psg_dis': flux_CPr_Jej_psg_dis,
                'CPr_jej': QCPr_Jej_g,
            }
            flattened_data_jej_CPr.append(flattened_entry_jej_CPr)
        self.df_RP_j = pd.DataFrame(flattened_data_jej_CPr)
        #print(f"df_duodenum, {df_s}")
        return self.df_RP_j, flattened_data_jej_CPr
    
    def feed_jej(self, t, Qfeed_Jej_g):

        Q_feedDuo_g_at_t = self.result_feed_duo.sol(t)
        
        if Qfeed_Jej_g.all() < 0.0:
            Qfeed_Jej_g = 1e-10
            
        P_Jej_feed = Q_feedDuo_g_at_t*self.Kp_Duo_min
        U_JejIl_feed = Qfeed_Jej_g*self.Kp_Jej_min
        
        dfeedjejdt = P_Jej_feed - U_JejIl_feed
        return dfeedjejdt, U_JejIl_feed, Qfeed_Jej_g
    
    def solve_feed_jej(self, t, Qfeed_Jej_g):
        dfeedjejdt, _, _ = self.feed_jej(t, Qfeed_Jej_g)
        return dfeedjejdt
    
    def solving_jej_feed(self):
        self.iJej_g = [0.00000000001]
        self.result_feed_jej = solve_ivp(self.solve_feed_jej, self.t_span, y0=self.iJej_g, t_eval=self.t_eval, dense_output=True, method='RK45')
    
    def flatten_result_jej_feed(self):
        flattened_data_jej_feed = []

        for t, y in zip(self.result_feed_jej.t, self.result_feed_jej.y.T):
            dfeedjejdt, U_JejIl_feed, Qfeed_Jej_g = self.feed_jej(t, y)
            Qfeed_Jej_g = y
            
            flattened_entry_jej_feed ={
                't': t, 
                'dfeedjejdt': dfeedjejdt,
                'flux_feed_JejIl': U_JejIl_feed,
                'Qfeed_jej': Qfeed_Jej_g,
            }
            flattened_data_jej_feed.append(flattened_entry_jej_feed)
        self.df_feed_j= pd.DataFrame(flattened_data_jej_feed)
        #print(f"df_duodenum, {df_s}")
        return self.df_feed_j, flattened_data_jej_feed 
    
        #----------plots results from Jejunum-------------------------------#
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
    
