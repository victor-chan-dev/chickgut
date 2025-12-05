
import math 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Duodenum():
        
    def __init__(self, t_span, iDuo_g, t_eval, result_fore, BWeight_kgb, Kp_PVG_min, constants, k_absp, k_digestrate, Kp_endog_min):
        self.name = "duodenum"
        self.t_span = t_span
        self.t_eval = t_eval

        self.iDuo_g = iDuo_g
        self.init_Duo_CPu= None
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

        self.result_fore = result_fore
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
        self.length_cm = 14.437*self.BWeight_kgb  # duodenum length (cm) from: Novotny et al. 2023, averaged across coarse/medium/fine diets
        self.r_cm = 1.18/2                   # duodenum radius (cm), duodenum diameter/2 = radius from: steczny and kokosynski 2019 - 42 DOA
        self.volume_cm3 = math.pi*math.pow(self.r_cm, 2)*self.length_cm  # duodneum volume (cm^3)

        #--for Duo_node0--#
        self.total_discretize = 101    # number of sections to split volume into duodenum (# of nodes)
        self.total_node_num = self.total_discretize - 1 # number of total nodes includ node 0 that is it's own pool
        Duo_exclude_node0_discretize = self.total_discretize - 1
        DuoV_cm3_total = np.linspace(0, self.volume_cm3, self.total_discretize) # generating evenly spaced numbers (cm^3) over range of the volume 

        self.DuoV_cm3 = DuoV_cm3_total[1:] # list of node volumes excluding node 0

        self.Duo_single_node_V = self.volume_cm3/self.total_node_num

        self.MRT_min = 2.808 #from MRT Meta-analysis
        self.Kp_Duo_min = 1/self.MRT_min   # fractional (SOLID) passage rate out of the duodenum (/min), from meta-A 
        #Kp_Duo_CPu_min = 2.0 #fractional (SOLID) passage rate out of duodenum (/min) for undigestible, from meta-A - inverse mrt 1/0.5

        self.VF = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/self.MRT_min #volumetric flow rate
        self.init_Duo_CPu = np.zeros(Duo_exclude_node0_discretize) # initial values for nodes of the duodenum excluding first discretized point i.e. node0
        self.init_Duo_CPsl = np.zeros(Duo_exclude_node0_discretize)
        self.init_Duo_CPr = np.zeros(Duo_exclude_node0_discretize)
    
    def calculate_duo_endog(self):
        SI_volume_cm3 = self.volume_cm3 + self.volume_jej_cm3 + self.volume_il_cm3
    
        self.vol_prop = self.volume_cm3 / SI_volume_cm3

        Il_BasalAA = 0.799 #g of basal ileal endogenous protein/g of CP from: Ravindran 2021, total of table 1 - i.e. total endogenous of SI
        self.basalaa = Il_BasalAA * self.vol_prop # basal endogenous AA input for duodenum based on proportion of volume 

        #---endogenous protein from duodenum-----#
        Duo_basalaa_single_node = self.basalaa/self.total_node_num
        P_CPe_Duo_gmin = self.Kp_endog_min*Duo_basalaa_single_node
        self.P_CPe_gmincm3 = P_CPe_Duo_gmin/self.Duo_single_node_V

    
    def Duo_node0_CPu (self, t, QCPu_Duonode0_g):
        index_PVGDuo = np.searchsorted(self.result_fore.t, t, side='left')
        
        Q_CPu_g_at_t = self.result_fore.y.T[index_PVGDuo, 1]
        #print(Q_CPu_g_at_t)
        P_CPu_PVGDuo_gmin = self.Kp_PVG_min*(Q_CPu_g_at_t*self.constants['UP_Fr']) #g/min

        U_CPu_Duonode0Duo_gmin = self.Kp_Duo_min*QCPu_Duonode0_g #g/min
        # if QCPu_Duonode0_g < 0:
        #     QCPu_Duonode0_g = 1e-10
        dUDnode0dt = P_CPu_PVGDuo_gmin - U_CPu_Duonode0Duo_gmin #g/min
        
        return QCPu_Duonode0_g, U_CPu_Duonode0Duo_gmin, dUDnode0dt
    
    def Duo_node0_CPsl (self, t, QCPsl_Duonode0_g):
        index_PVGDuo = np.searchsorted(self.result_fore.t, t, side='left')
        
        Q_CPsl_g_at_t = self.result_fore.y.T[index_PVGDuo, 1]
        
        P_CPsl_PVGDuo_gmin = self.Kp_PVG_min*(Q_CPsl_g_at_t*self.constants['SlP_Fr'])
        
        U_CPsl_Duonode0Duo_gmin = self.Kp_Duo_min*QCPsl_Duonode0_g
        # if QCPsl_Duonode0_g < 0:
        #     QCPsl_Duonode0_g = 1e-10
        dSlDnode0dt = P_CPsl_PVGDuo_gmin - U_CPsl_Duonode0Duo_gmin
        
        return QCPsl_Duonode0_g, U_CPsl_Duonode0Duo_gmin, dSlDnode0dt
    
    def Duo_node0_CPr (self, t, QCPr_Duonode0_g):
        index_PVGDuo = np.searchsorted(self.result_fore.t, t, side='left')
        
        Q_CPr_g_at_t = self.result_fore.y.T[index_PVGDuo, 1]
        
        P_CPr_PVGDuo_gmin = self.Kp_PVG_min*(Q_CPr_g_at_t*self.constants['RP_Fr'])
        
        U_CPr_Duonode0Duo_gmin = self.Kp_Duo_min*QCPr_Duonode0_g
        # if QCPr_Duonode0_g < 0:
        #     QCPr_Duonode0_g = 1e-10
        dRDnode0dt = P_CPr_PVGDuo_gmin - U_CPr_Duonode0Duo_gmin
        
        return QCPr_Duonode0_g, U_CPr_Duonode0Duo_gmin, dRDnode0dt
    
    def Solve_Duonode0_CPu(self, t, QCPu_Duonode0_g):
        _, _, dUDnode0dt = self.Duo_node0_CPu( t, QCPu_Duonode0_g)
        return dUDnode0dt
    
    def Solve_Duonode0_CPsl(self, t, QCPsl_Duonode0_g):
        _, _, dSlDnode0dt = self.Duo_node0_CPsl(t, QCPsl_Duonode0_g)
        return dSlDnode0dt
    
    def Solve_Duonode0_CPr(self, t, QCPr_Duonode0_g):
        _, _, dRDnode0dt = self.Duo_node0_CPr(t, QCPr_Duonode0_g)
        return dRDnode0dt       

    #----------- solve for the values of duodenum node 0 crude protein (CP, g) using solve_ivp --------------------------#
    def solving_Unode0(self):
        self.result_UDnode0 = solve_ivp(self.Solve_Duonode0_CPu, self.t_span, y0=self.iDuo_g, t_eval=self.t_eval, dense_output=True, method='RK45')
        self.result_SlDnode0 = solve_ivp(self.Solve_Duonode0_CPsl, self.t_span, y0=self.iDuo_g, t_eval=self.t_eval, dense_output=True, method='RK45')
        self.result_RDnode0 = solve_ivp(self.Solve_Duonode0_CPr, self.t_span, y0=self.iDuo_g, t_eval=self.t_eval, dense_output=True, method='RK45')

    #-------------------undigestible protein duodenum------------------------------------------------------#
    def method_of_lines_CPu_Duo(self, t, QCPu_Duo_g):
        #index_Duo0Duo = np.searchsorted(self.result_UDnode0.t, t, side='left') #index of first suitable location found
        
        QCPu_Duo_g[0] = self.result_UDnode0.sol(t)  
        #QCPu_Duo_g = np.clip(QCPu_Duo_g, 0, None) #applying lower limit cap of 0, none is no upper cap
        
        QCPu_Duo_g = np.maximum(QCPu_Duo_g , 1e-10) #checks negatives
    
        
        U_CPu_Duo_psg_gmincm3 = self.VF * np.diff(QCPu_Duo_g)/ np.diff(self.DuoV_cm3)     # calculates amount of protein at each node
        # g/min*cm^3  prepend attaches the value before the list of QCPu_Duo_g, so use [1:] to access from discretize point 1 and onwards
        
        P_CPu_Duo0Duo_gmin = self.Kp_Duo_min*self.result_UDnode0.sol(t)

        P_CPu_Duo0Duo_gmincm3 = P_CPu_Duo0Duo_gmin/self.Duo_single_node_V

        UD_Diff = self.P_CPe_gmincm3 + P_CPu_Duo0Duo_gmincm3 - U_CPu_Duo_psg_gmincm3 
                    # g/min*cm3         #g/min*cm^3
                    
        dUDdt = np.concatenate([[0], UD_Diff]) #dUDdt is one less bc of np.diff, need to add a 0 to avoid broadcasting error
                                                #Kitchin explains it as concentration is constant at entrance
    
        flux_CPu_DuoJej = QCPu_Duo_g[-1]*self.Kp_Duo_min
        C_CPu_Duolast_gcm3 = QCPu_Duo_g[-1]/self.Duo_single_node_V
        # print("P_CPu_Duo0Duo_gmincm3:", P_CPu_Duo0Duo_gmincm3)
        return dUDdt, flux_CPu_DuoJej, C_CPu_Duolast_gcm3,

    def Solve_method_of_lines_CPu_Duo(self, t, QCPu_Duo_g):
        dUDdt, _, _ = self.method_of_lines_CPu_Duo(t, QCPu_Duo_g)
        return dUDdt
    
    #------------------------slowly-digested protein duodenum---------------------------------------------------#
    def method_of_lines_CPsl_Duo(self, t, QCPsl_Duo_g):
        #index_Duo0Duo = np.searchsorted(self.result_SlDnode0.t, t, side='left')
        QCPsl_Duo_g[0] = self.result_SlDnode0.sol(t) 
        
        QCPsl_Duo_g = np.maximum(QCPsl_Duo_g, 1e-10)
        # if QCPsl_Duo_g.all() < 0.0:
        #     QCPsl_Duo_g = 1e-10
                    
        #---flux from Duo slowly- to Duo rapidly-digested protein---#
        Duo_CPsl_dis_gmincm3 = (self.k_digestrate* QCPsl_Duo_g[1:]**2)/self.Duo_single_node_V #  undigested CP -> soluble CP flux (g/min), *note - Kdd_CPu= 0
        # g/min*cm^3
        
        #---Kitchin Group plug-flow---#
        U_CPsl_Duo_psg_gmincm3 = self.VF * np.diff(QCPsl_Duo_g) / np.diff(self.DuoV_cm3) # calculates amount of protein at each node
            # g/min*cm^3  
            
        #---flux from Duo node 0  to Duo PF ---#
        P_CPsl_Duo0Duo_gmin = self.Kp_Duo_min*self.result_SlDnode0.sol(t)
        P_CPsl_Duo0Duo_gmincm3 = P_CPsl_Duo0Duo_gmin/self.Duo_single_node_V
        
        SlD_Diff =  P_CPsl_Duo0Duo_gmincm3 - (U_CPsl_Duo_psg_gmincm3 + Duo_CPsl_dis_gmincm3) 
                    # self.P_CPe_gmincm3 g/min*cm3                                 #g/min*cm^3              #g/min*cm3
        
        dSlDdt = np.concatenate([[0], SlD_Diff])
        flux_CPsl_DuoJej = (QCPsl_Duo_g[-1])*self.Kp_Duo_min
        C_CPsl_Duolast_gcm3 = (QCPsl_Duo_g[-1])/self.Duo_single_node_V
        
        return dSlDdt, flux_CPsl_DuoJej, C_CPsl_Duolast_gcm3
    
    def Solve_method_of_lines_CPsl_Duo(self, t, QCPsl_Duo_g):
        dSlDdt, _, _ = self.method_of_lines_CPsl_Duo(t, QCPsl_Duo_g)
        return dSlDdt
    
    #----------- solve for the values of duodenum crude protein (CP, g) using solve_ivp --------------------------#
    def solving_duo_USl(self):
        self.init_Duo_CPu[0] = self.result_UDnode0.y.T[0, -1]
        
        self.init_Duo_CPsl[0] = self.result_SlDnode0.y.T[0, -1]
  
        self.UDexit_SS = solve_ivp(self.Solve_method_of_lines_CPu_Duo, self.t_span, self.init_Duo_CPu, t_eval=self.t_eval, 
                                   dense_output=True, method='Radau')
        #Radau implicit runge kutta method, good for stiff differential-algebraic equations, stability and accuracy

        self.SlDexit_SS = solve_ivp(self.Solve_method_of_lines_CPsl_Duo, self.t_span, self.init_Duo_CPsl, t_eval=self.t_eval, 
                                    dense_output=True, method='Radau')

    
    #----prepping data for next compartment - undigestible----#  
    def flatten_result_duo_CPu(self):
        flattened_data_duo_CPu = []
        for t, y in zip(self.UDexit_SS.t, self.UDexit_SS.y.T):
            dUDdt, flux_CPu_DuoJej, C_CPu_Duolast_gcm3 = self.method_of_lines_CPu_Duo(t, y)
            QCPu_Duo_g=y[-1] # asks for values from the last node
            flattened_entry_duo_CPu ={
                't': t, 
                'dUDdt': dUDdt[-1],
                'flux_CPu_DuoJej': flux_CPu_DuoJej,
                'Conc_CPu_Duolast':C_CPu_Duolast_gcm3,
                'QCPu_duo': QCPu_Duo_g
            }
            flattened_data_duo_CPu.append(flattened_entry_duo_CPu)
        self.df_UP_d = pd.DataFrame(flattened_data_duo_CPu)

        #print(f"df_duodenum, {df_d}")
        return self.df_UP_d, flattened_data_duo_CPu
    
    #df_UP_DUO = self.df_UP_d  
    
    #df_Q_CPu_Duo = df_UP_DUO[['t', 'QCPu_duo']] #pulling out just the relavent columns from the df_cpvg dataframe
    
    #----prepping data for next compartment - slowly-digested fraction----#   
    def flatten_result_duo_CPsl(self):
        flattened_data_duo_CPsl = []
        
        for t, y in zip(self.SlDexit_SS.t, self.SlDexit_SS.y.T):
            dDSldt, flux_CPsl_DuoJej, C_CPsl_Duolast_gcm3 = self.method_of_lines_CPsl_Duo(t, y)
            QCPsl_Duo_g=y[-1] # asks for values from the last node
            
            flattened_entry_duo_CPsl ={
                't': t, 
                'dDSldt': dDSldt[-1],
                'flux_CPsl_DuoJej': flux_CPsl_DuoJej,
                'Conc_CPsl_Duolast': C_CPsl_Duolast_gcm3,
                'QCPsl_duo':QCPsl_Duo_g
            }
            flattened_data_duo_CPsl.append(flattened_entry_duo_CPsl)
        self.df_SlP_d = pd.DataFrame(flattened_data_duo_CPsl)
        #print(f"df_duodenum, {df_DP_d}")
        return self.df_SlP_d, flattened_data_duo_CPsl
    
    # df_SlP_d, _ = flatten_result_duo_CPsl()

    # df_SlP_DUO = pd.DataFrame(df_SlP_d)  

    #------slowly- to rapidly-digested protein in duodenum setup------------#
    def SlP_for_RP_d(self):
        self.flatten_result_duo_CPsl()
        df_Q_CPsl_Duo = self.df_SlP_d[['t', 'QCPsl_duo']] #pulling out just the relavent columns from the df_cpvg dataframe
        df_Q_CPsl_Duo.reset_index(drop=True, inplace=True) #resets the index numbers of df_PVG to default integer index (0,1,2,3...)
        Duo_CPsl_Q = df_Q_CPsl_Duo['QCPsl_duo'].values 
        return df_Q_CPsl_Duo, Duo_CPsl_Q
    
    #------------------------Rapidly-digested protein duodenum---------------------------------------------------#
    def method_of_lines_CPr_Duo(self, t, QCPr_Duo_g):

        #index_Duo0Duo = np.searchsorted(self.result_RDnode0.t, t, side='left')
        index_duo_SlR_flux = np.searchsorted(self.df_Q_CPsl_Duo['t'], t, side='left')
        
        QCPr_Duo_g[0] = self.result_RDnode0.sol(t) 
        
        #QCPr_Duo_g = np.maximum(QCPr_Duo_g, 1e-12)
        
        Duo_CPr_dis = (self.k_absp*QCPr_Duo_g[1:]**2)/self.Duo_single_node_V     #  rapid CP -> rapid CP flux (g/min)
        
        #QCPr_Duo_g = np.clip(QCPr_Duo_g, 0, None)
        U_CPr_Duo_psg_mincm3 = self.VF * np.diff(QCPr_Duo_g) / np.diff(self.DuoV_cm3)     # calculates amount of protein at each node
            # g/min*cm^3  
        
        U_CPr_Duo_psg_dis_gmincm3 = np.maximum(U_CPr_Duo_psg_mincm3 + Duo_CPr_dis, 1e-12) # avoiding negatives

        #---flux from Duo node 0  to Duo PF ---#
        P_CPr_Duo0Duo_gmin = self.Kp_Duo_min*self.result_RDnode0.sol(t) 
        P_CPr_Duo0Duo_gmincm3 = P_CPr_Duo0Duo_gmin/self.Duo_single_node_V     
            
        #---flux from Duo slowly- to Duo rapidly-digested protein---#
        P_CPr_DuoSlDuoR_gmin = self.k_digestrate*(self.Duo_CPsl_Q[index_duo_SlR_flux])
        P_CPr_DuoSlDuoR_gmincm3 = P_CPr_DuoSlDuoR_gmin/self.Duo_single_node_V
        #Duo_CPd is from the slowly-digested protein function, [index_duo_SlR_flux] helps match the values that correspond to each time point
        
        #---flux from Duo Rapidly-digested to Absorbed---#
        #U_CPr_DuoRDuoA_gmin = k_Duo_diffusion*QCPr_Duo_g[1:]
        
       # RD_Diff =  P_CPr_DuoSlDuoR_gmincm3 + P_CPr_Duo0Duo_gmincm3 - U_CPr_Duo_psg_dis_gmincm3
        RD_Diff = P_CPr_DuoSlDuoR_gmincm3 + P_CPr_Duo0Duo_gmincm3 - U_CPr_Duo_psg_dis_gmincm3
                    # g/min*cm3         #g/min*cm^3
                    
        dRDdt_concat = np.concatenate([[0], RD_Diff])

        dRDdt = np.maximum(dRDdt_concat, 1e-12, None) #clamped at minimum value is 0, no upper bound
                
        flux_CPr_DuoJej = (QCPr_Duo_g[-1])*self.Kp_Duo_min
        C_CPr__Duolast_gcm3 = (QCPr_Duo_g[-1])/self.Duo_single_node_V
        return dRDdt, flux_CPr_DuoJej, C_CPr__Duolast_gcm3
    
    def Solve_method_of_lines_CPr_Duo(self, t, QCPr_Duo_g):
        dRDdt, _, _ = self.method_of_lines_CPr_Duo(t, QCPr_Duo_g)
        return dRDdt
    
    def solving_duo_R(self):
        self.df_Q_CPsl_Duo, self.Duo_CPsl_Q = self.SlP_for_RP_d()

        self.init_Duo_CPr[0] = self.result_RDnode0.y.T[0, -1]

        self.RDexit_SS = solve_ivp(self.Solve_method_of_lines_CPr_Duo, self.t_span, self.init_Duo_CPr, t_eval=self.t_eval, 
                                   dense_output=True, method='LSODA')

    #----prepping data for next compartment - undigestible----#  
    def flatten_result_duo_CPr(self):
        flattened_data_duo_CPr = []

        for t, y in zip(self.RDexit_SS.t, self.RDexit_SS.y.T):
            dRDdt, flux_CPr_DuoJej, C_CPr__Duolast_gcm3 = self.method_of_lines_CPr_Duo(t, y)
            QCPr_Duo_g = y[-1] # asks for values from the last node
            
            flattened_entry_duo_CPr ={
                't': t, 
                'dRDdt': dRDdt[-1],
                'flux_CPr_DuoJej': flux_CPr_DuoJej,
                'C_CPr__Duolast_gcm3': C_CPr__Duolast_gcm3,
                'QCPr_duo': QCPr_Duo_g,
            }
            flattened_data_duo_CPr.append(flattened_entry_duo_CPr)
        self.df_RP_d= pd.DataFrame(flattened_data_duo_CPr)
        #print(f"df_duodenum, {df_s}")
        return self.df_RP_d, flattened_data_duo_CPr
    
    def feed_duo(self, t, Qfeed_Duo_g):
        index_PVGDuo = np.searchsorted(self.result_fore.t, t, side='left')
        
        Q_feedPVG_g_at_t = self.result_fore.y.T[index_PVGDuo, -1]
        
        if Qfeed_Duo_g.all() < 0.0:
            Qfeed_Duo_g = 1e-10
        
        P_PVG_feed = Q_feedPVG_g_at_t*self.Kp_PVG_min
        U_DuoJej_feed = Qfeed_Duo_g*self.Kp_Duo_min
        
        dfeedduodt = P_PVG_feed - U_DuoJej_feed
        return dfeedduodt, U_DuoJej_feed, Qfeed_Duo_g
    
    def solve_feed_duo(self, t, Qfeed_Duo_g):
        dfeedduodt, _, _ = self.feed_duo(t, Qfeed_Duo_g)
        return dfeedduodt
    
    def solving_duo_feed(self):
        self.result_feed_duo = solve_ivp(self.solve_feed_duo, self.t_span, y0=self.iDuo_g, t_eval=self.t_eval, dense_output=True, method='RK45')
    
    def flatten_result_duo_feed(self):
        flattened_data_duo_feed = []

        for t, y in zip(self.result_feed_duo.t, self.result_feed_duo.y.T):
            dfeedduodt, U_DuoJej_feed, Qfeed_Duo_g = self.feed_duo(t, y)
            Qfeed_Duo_g = y
            
            flattened_entry_duo_feed ={
                't': t, 
                'dfeedduodt': dfeedduodt,
                'flux_feed_DuoJej': U_DuoJej_feed,
                'Qfeed_duo': Qfeed_Duo_g,
            }
            flattened_data_duo_feed.append(flattened_entry_duo_feed)
        self.df_feed_d = pd.DataFrame(flattened_data_duo_feed)
        #print(f"df_duodenum, {df_s}")
        return self.df_feed_d, flattened_data_duo_feed 
    
        #----------Plots results from duodenum----------------------#
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
        
    

        