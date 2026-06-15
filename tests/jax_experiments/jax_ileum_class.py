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
        self.Discretize_il = 101    
        self.Node_num = self.Discretize_il - 1
        self.length_cm = 34.643*self.BWeight_kgb 
        self.r_cm = 0.9/2                   
        self.volume_il_cm3 = math.pi*math.pow(self.r_cm, 2)*self.length_cm 
        
        self.IlV_cm3 = np.linspace(0, self.volume_il_cm3, self.Discretize_il)
        self.Il_single_node_V = self.volume_il_cm3 / self.Node_num
        self.MRT_min = self.constants['Il_MRT'] 
        MRT_Il_CPu_min = 49.5 

        self.Kp_Il_min =  1/self.MRT_min 
        self.Kp_Il_CPu_min = 1/MRT_Il_CPu_min 
        
        self.VF = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/self.MRT_min 
        self.VF_Il_CPu = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/MRT_Il_CPu_min
        self.init_Il_CPu = np.zeros(self.Discretize_il) 
        self.init_Il_CPsl = np.zeros(self.Discretize_il)
        self.init_Il_CPr = np.zeros(self.Discretize_il)
                

    def calculate_Il_endog(self):
        SI_volume_cm3 = self.volume_cm3 + self.volume_jej_cm3 + self.volume_il_cm3
        self.vol_prop = self.volume_il_cm3 / SI_volume_cm3
        
        Il_BasalAA = 0.799 
        self.basalaa = Il_BasalAA * self.vol_prop 

        Ileum_basalaa_single_node = self.basalaa/(self.Discretize_il - 1) 
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

    @time_it
    def solving_il_USl(self):
        self.init_Il_CPu[0] = float(self.UJexit_SS.ys[-1, -1]) / self.Il_single_node_V
        self.init_Il_CPsl[0] = float(self.SlJexit_SS.ys[-1, -1]) / self.Il_single_node_V
        
        y0_u = jnp.array(self.init_Il_CPu)
        y0_sl = jnp.array(self.init_Il_CPsl)
        
        args_u = (self.UJexit_SS, self.VF_Il_CPu, jnp.array(self.IlV_cm3), self.Kp_Il_min, self.CP_jej_lastnode_vol_cm3, self.P_CPe_Il_gmincm3)
        args_sl = (self.SlJexit_SS, self.VF, jnp.array(self.IlV_cm3), self.Kp_Jej_min, self.Il_single_node_V, self.CP_jej_lastnode_vol_cm3, self.k_digestrate)
        
        solver = diffrax.Kvaerno5(root_finder=diffrax.VeryChord())
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.UIexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_Ileum_CPu_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_u, args=args_u, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )
        
        self.SlIexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_Ileum_CPsl_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_sl, args=args_sl, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )

    def flatten_result_il_CPu(self):
        args_u = (self.UJexit_SS, self.VF_Il_CPu, jnp.array(self.IlV_cm3), self.Kp_Il_min, self.CP_jej_lastnode_vol_cm3, self.P_CPe_Il_gmincm3, self.Il_single_node_V)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_Ileum_CPu_metrics(t, y, args_u)
            
        v_compute = jax.vmap(compute_metrics)
        dUIdt_all, flux_all = v_compute(self.UIexit_SS.ts, self.UIexit_SS.ys)
        
        self.df_UP_i = pd.DataFrame({
            't': np.array(self.UIexit_SS.ts),
            'dUIdt': np.array(dUIdt_all[:, -1]),
            'flux_CPu_Ileum_psg': np.array(flux_all),
            'CPu_il': np.array(self.UIexit_SS.ys[:, -1])
        })
        return self.df_UP_i, None
    
    def flatten_result_il_CPsl(self):
        args_sl = (self.SlJexit_SS, self.VF, jnp.array(self.IlV_cm3), self.Kp_Jej_min, self.Il_single_node_V, self.CP_jej_lastnode_vol_cm3, self.k_digestrate, self.Kp_Il_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_Ileum_CPsl_metrics(t, y, args_sl)
            
        v_compute = jax.vmap(compute_metrics)
        dSlIdt_all, flux_all = v_compute(self.SlIexit_SS.ts, self.SlIexit_SS.ys)
        
        self.df_SlP_i = pd.DataFrame({
            't': np.array(self.SlIexit_SS.ts),
            'dSlIdt': np.array(dSlIdt_all[:, -1]),
            'flux_CPsl_Ileum_psg_dis': np.array(flux_all),
            'CPsl_il': np.array(self.SlIexit_SS.ys[:, -1])
        })
        return self.df_SlP_i, None

    def SlP_for_RP_i(self):
        self.flatten_result_il_CPsl()
        df_Q_CPsl_Il = self.df_SlP_i[['t', 'CPsl_il']] 
        df_Q_CPsl_Il.reset_index(drop=True, inplace=True) 
        Il_CPsl = df_Q_CPsl_Il['CPsl_il'].values  
        return df_Q_CPsl_Il, Il_CPsl
    
    @time_it
    def solving_il_R(self):
        self.df_Q_CPsl_Il, self.Il_CPsl = self.SlP_for_RP_i()

        self.init_Il_CPr[0] = float(self.RJexit_SS.ys[-1, -1]) / self.Il_single_node_V
            
        y0_r = jnp.array(self.init_Il_CPr)
        args_r = (self.RJexit_SS, self.SlIexit_SS, self.VF, jnp.array(self.IlV_cm3), self.Kp_Jej_min, self.Il_single_node_V, self.k_absp, self.k_digestrate)
        
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.RIexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_CPr_Il_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_r, args=args_r, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )
    
    def flatten_result_il_CPr(self):
        args_r = (self.RJexit_SS, self.SlIexit_SS, self.VF, jnp.array(self.IlV_cm3), self.Kp_Jej_min, self.Il_single_node_V, self.k_absp, self.k_digestrate, self.Kp_Il_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_CPr_Il_metrics(t, y, args_r)
            
        v_compute = jax.vmap(compute_metrics)
        dRIdt_all, flux_all = v_compute(self.RIexit_SS.ts, self.RIexit_SS.ys)
        
        self.df_RP_i = pd.DataFrame({
            't': np.array(self.RIexit_SS.ts),
            'dRIdt': np.array(dRIdt_all[:, -1]),
            'flux_CPr_Ileum_psg_dis': np.array(flux_all),
            'CPr_il': np.array(self.RIexit_SS.ys[:, -1])
        })
        return self.df_RP_i, None
    
    @time_it
    def solving_il_feed(self):
        self.iIl_g = [0.00000000001]
        y0 = jnp.array(self.iIl_g)
        args_feed = (self.result_feed_jej, self.Kp_Jej_min, self.Kp_Il_min)
        
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.result_feed_il = diffrax.diffeqsolve(
            diffrax.ODETerm(_feed_il_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_feed, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )
    
    def flatten_result_il_feed(self):
        args_feed = (self.result_feed_jej, self.Kp_Jej_min, self.Kp_Il_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _feed_il_metrics(t, y, args_feed)
            
        v_compute = jax.vmap(compute_metrics)
        dfeedildt_all, flux_all, Q_all = v_compute(self.result_feed_il.ts, self.result_feed_il.ys)
        
        self.df_feed_i = pd.DataFrame({
            't': np.array(self.result_feed_il.ts),
            'dfeedildt': np.array(dfeedildt_all[:, 0]),
            'flux_feed_IlCo': np.array(flux_all[:, 0]),
            'Qfeed_il': np.array(self.result_feed_il.ys[:, 0])
        })
        return self.df_feed_i, None 
