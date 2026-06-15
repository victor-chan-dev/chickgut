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
        self.length_cm = 33.166*self.BWeight_kgb  
        self.r_cm = 1.14/2                   
        self.volume_jej_cm3 = math.pi*math.pow(self.r_cm, 2)*self.length_cm  

        self.Discretize_jej = 101    
        self.Node_num_jej = self.Discretize_jej - 1
        self.JejV_cm3 = np.linspace(0, self.volume_jej_cm3, self.Discretize_jej)
        self.Jej_single_node_V = self.volume_jej_cm3/self.Node_num_jej

        MRT_Jej_CPu_min = 16.0 
        self.MRT_Jej_min = self.constants['Jej_MRT']

        self.Kp_Jej_min = 1/self.MRT_Jej_min 
        self.Kp_Jej_CPu_min = 1/MRT_Jej_CPu_min 
        
        self.VF = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/self.MRT_Jej_min 
        self.VF_CPu = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/MRT_Jej_CPu_min 
        self.init_Jej_CPu = np.zeros(self.Discretize_jej) 
        self.init_Jej_CPsl = np.zeros(self.Discretize_jej)
        self.init_Jej_CPr = np.zeros(self.Discretize_jej)


    def calculate_jej_endog(self):
        SI_volume_cm3 = self.volume_cm3 + self.volume_jej_cm3 + self.volume_il_cm3
    
        self.vol_prop = self.volume_jej_cm3 / SI_volume_cm3
        
        Il_BasalAA = 0.799 
        self.basalaa = Il_BasalAA * self.vol_prop 

        self.basalaa_single_node = self.basalaa/(self.Discretize_jej - 1)  
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
       

    @time_it
    def solving_jej_USl(self):
        # Diffrax .ys has shape (num_time_steps, num_variables)
        self.init_Jej_CPu[0] = float(self.UDexit_SS.ys[-1, -1]) / self.Jej_single_node_V
        self.init_Jej_CPsl[0] = float(self.SlDexit_SS.ys[-1, -1]) / self.Jej_single_node_V

        y0_u = jnp.array(self.init_Jej_CPu)
        y0_sl = jnp.array(self.init_Jej_CPsl)
        
        args_u = (self.UDexit_SS, self.VF_CPu, jnp.array(self.JejV_cm3), self.Kp_Jej_min, self.Duo_single_node_V, self.P_CPe_Jej_gmincm3)
        args_sl = (self.SlDexit_SS, self.VF, jnp.array(self.JejV_cm3), self.Kp_Duo_min, self.Jej_single_node_V, self.k_digestrate)
        
        solver = diffrax.Kvaerno5(root_finder=diffrax.VeryChord())
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.UJexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_Jej_CPu_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_u, args=args_u, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )
        
        self.SlJexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_Jej_CPsl_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_sl, args=args_sl, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )
        
    def flatten_result_jej_CPu(self):
        args_u = (self.UDexit_SS, self.VF_CPu, jnp.array(self.JejV_cm3), self.Kp_Jej_min, self.Duo_single_node_V, self.P_CPe_Jej_gmincm3, self.Kp_Jej_CPu_min, self.Jej_single_node_V)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_Jej_CPu_metrics(t, y, args_u)
            
        v_compute = jax.vmap(compute_metrics)
        dUJdt_all, flux_all = v_compute(self.UJexit_SS.ts, self.UJexit_SS.ys)
        
        self.df_UP_j = pd.DataFrame({
            't': np.array(self.UJexit_SS.ts),
            'dUJdt': np.array(dUJdt_all[:, -1]),
            'flux_CPu_Jej_psg': np.array(flux_all),
            'CPu_jej': np.array(self.UJexit_SS.ys[:, -1])
        })
        return self.df_UP_j, None

    def flatten_result_jej_CPsl(self):
        args_sl = (self.SlDexit_SS, self.VF, jnp.array(self.JejV_cm3), self.Kp_Duo_min, self.Jej_single_node_V, self.k_digestrate, self.Kp_Jej_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_Jej_CPsl_metrics(t, y, args_sl)
            
        v_compute = jax.vmap(compute_metrics)
        dSlJdt_all, flux_all = v_compute(self.SlJexit_SS.ts, self.SlJexit_SS.ys)
        
        self.df_SlP_j = pd.DataFrame({
            't': np.array(self.SlJexit_SS.ts),
            'dSlJdt': np.array(dSlJdt_all[:, -1]),
            'flux_CPsl_Jej_psg_dis': np.array(flux_all),
            'CPsl_jej': np.array(self.SlJexit_SS.ys[:, -1])
        })
        return self.df_SlP_j, None

    def SlP_for_RP_j(self):
        self.flatten_result_jej_CPsl()
        df_Q_CPsl_Jej = self.df_SlP_j[['t', 'CPsl_jej']] 
        df_Q_CPsl_Jej.reset_index(drop=True, inplace=True) 
        Jej_CPsl = df_Q_CPsl_Jej['CPsl_jej'].values 
        return df_Q_CPsl_Jej, Jej_CPsl
    
    @time_it
    def solving_jej_R(self):
        self.df_Q_CPsl_Jej, self.Jej_CPsl = self.SlP_for_RP_j()
        self.init_Jej_CPr[0] = float(self.RDexit_SS.ys[-1, -1]) / self.Jej_single_node_V
            
        y0_r = jnp.array(self.init_Jej_CPr)
        args_r = (self.RDexit_SS, self.SlJexit_SS, self.VF, jnp.array(self.JejV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.Jej_single_node_V, self.k_absp, self.k_digestrate)
        
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.RJexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_CPr_Jej_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_r, args=args_r, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )

    def flatten_result_jej_CPr(self):
        args_r = (self.RDexit_SS, self.SlJexit_SS, self.VF, jnp.array(self.JejV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.Jej_single_node_V, self.k_absp, self.k_digestrate, self.Kp_Jej_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_CPr_Jej_metrics(t, y, args_r)
            
        v_compute = jax.vmap(compute_metrics)
        dRJdt_all, flux_all = v_compute(self.RJexit_SS.ts, self.RJexit_SS.ys)
        
        self.df_RP_j = pd.DataFrame({
            't': np.array(self.RJexit_SS.ts),
            'dRJdt': np.array(dRJdt_all[:, -1]),
            'flux_CPr_Jej_psg_dis': np.array(flux_all),
            'CPr_jej': np.array(self.RJexit_SS.ys[:, -1])
        })
        return self.df_RP_j, None
    
    @time_it
    def solving_jej_feed(self):
        self.iJej_g = [0.00000000001]
        y0 = jnp.array(self.iJej_g)
        args_feed = (self.result_feed_duo, self.Kp_Duo_min, self.Kp_Jej_min)
        
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.result_feed_jej = diffrax.diffeqsolve(
            diffrax.ODETerm(_feed_jej_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_feed, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )
    
    def flatten_result_jej_feed(self):
        args_feed = (self.result_feed_duo, self.Kp_Duo_min, self.Kp_Jej_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _feed_jej_metrics(t, y, args_feed)
            
        v_compute = jax.vmap(compute_metrics)
        dfeedjejdt_all, flux_all, Q_all = v_compute(self.result_feed_jej.ts, self.result_feed_jej.ys)
        
        self.df_feed_j = pd.DataFrame({
            't': np.array(self.result_feed_jej.ts),
            'dfeedjejdt': np.array(dfeedjejdt_all[:, 0]),
            'flux_feed_JejIl': np.array(flux_all[:, 0]),
            'Qfeed_jej': np.array(self.result_feed_jej.ys[:, 0])
        })
        return self.df_feed_j, None 
