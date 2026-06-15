class Duodenum():
        
    def __init__(self, t_span, iDuo_g, t_eval, result_fore, BWeight_kgb, Kp_PVG_min, constants, k_absp, k_digestrate, Kp_endog_min):
        self.name = "duodenum"
        self.t_span = t_span
        self.t_eval = t_eval

        self.iDuo_g = iDuo_g
        self.init_Duo_CPu = None
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
        self.length_cm = 14.437*self.BWeight_kgb  
        self.r_cm = 1.18/2                   
        self.volume_cm3 = math.pi*math.pow(self.r_cm, 2)*self.length_cm  

        self.total_discretize = 101    
        self.total_node_num = self.total_discretize - 1 
        Duo_exclude_node0_discretize = self.total_discretize - 1
        DuoV_cm3_total = np.linspace(0, self.volume_cm3, self.total_discretize) 

        self.DuoV_cm3 = DuoV_cm3_total[1:] 
        self.Duo_single_node_V = self.volume_cm3/self.total_node_num

        self.MRT_min = 2.808 
        self.Kp_Duo_min = 1/self.MRT_min   

        self.VF = (math.pi*math.pow(self.r_cm, 2))*self.length_cm/self.MRT_min 
        self.init_Duo_CPu = np.zeros(Duo_exclude_node0_discretize) 
        self.init_Duo_CPsl = np.zeros(Duo_exclude_node0_discretize)
        self.init_Duo_CPr = np.zeros(Duo_exclude_node0_discretize)
    
    def calculate_duo_endog(self):
        SI_volume_cm3 = self.volume_cm3 + self.volume_jej_cm3 + self.volume_il_cm3
        self.vol_prop = self.volume_cm3 / SI_volume_cm3

        Il_BasalAA = 0.799 
        self.basalaa = Il_BasalAA * self.vol_prop 

        Duo_basalaa_single_node = self.basalaa/self.total_node_num
        P_CPe_Duo_gmin = self.Kp_endog_min*Duo_basalaa_single_node
        self.P_CPe_gmincm3 = P_CPe_Duo_gmin/self.Duo_single_node_V

    def solving_Unode0(self):
        # Prepare JAX arrays for foregut results
        fore_t = jnp.array(self.result_fore.t)
        fore_y = jnp.array(self.result_fore.y.T)
        
        args_u = (fore_t, fore_y, self.constants['UP_Fr'], self.Kp_PVG_min, self.Kp_Duo_min)
        args_sl = (fore_t, fore_y, self.constants['SlP_Fr'], self.Kp_PVG_min, self.Kp_Duo_min)
        args_r = (fore_t, fore_y, self.constants['RP_Fr'], self.Kp_PVG_min, self.Kp_Duo_min)

        solver = diffrax.Tsit5()  
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]
        y0 = jnp.array(self.iDuo_g)

        self.result_UDnode0 = diffrax.diffeqsolve(
            diffrax.ODETerm(_solve_node0_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_u, saveat=saveat, stepsize_controller=stepsize_controller
        )
        self.result_SlDnode0 = diffrax.diffeqsolve(
            diffrax.ODETerm(_solve_node0_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_sl, saveat=saveat, stepsize_controller=stepsize_controller
        )
        self.result_RDnode0 = diffrax.diffeqsolve(
            diffrax.ODETerm(_solve_node0_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_r, saveat=saveat, stepsize_controller=stepsize_controller
        )

    @time_it
    def solving_duo_USl(self):
        self.init_Duo_CPu[0] = float(self.result_UDnode0.ys[-1, 0])
        self.init_Duo_CPsl[0] = float(self.result_SlDnode0.ys[-1, 0])
        
        y0_u = jnp.array(self.init_Duo_CPu)
        y0_sl = jnp.array(self.init_Duo_CPsl)
        
        args_u = (self.result_UDnode0, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.P_CPe_gmincm3)
        args_sl = (self.result_SlDnode0, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.k_digestrate)
        
        solver = diffrax.Kvaerno5(root_finder=diffrax.VeryChord())
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.UDexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_CPu_Duo_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_u, args=args_u, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )
        
        with time_block("Solving duodenum slowly-digested protein equations"):
            self.SlDexit_SS = diffrax.diffeqsolve(
                diffrax.ODETerm(_method_of_lines_CPsl_Duo_jax), solver, t0=t0, t1=t1, dt0=0.1,
                y0=y0_sl, args=args_sl, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
            )

    def flatten_result_duo_CPu(self):
        args_u = (self.result_UDnode0, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.P_CPe_gmincm3)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_CPu_metrics(t, y, args_u)
        
        v_compute = jax.vmap(compute_metrics)
        dUDdt_all, flux_all, C_all = v_compute(self.UDexit_SS.ts, self.UDexit_SS.ys)
        
        self.df_UP_d = pd.DataFrame({
            't': np.array(self.UDexit_SS.ts),
            'dUDdt': np.array(dUDdt_all[:, -1]),
            'flux_CPu_DuoJej': np.array(flux_all),
            'Conc_CPu_Duolast': np.array(C_all),
            'QCPu_duo': np.array(self.UDexit_SS.ys[:, -1])
        })
        return self.df_UP_d, None
    
    def flatten_result_duo_CPsl(self):
        args_sl = (self.result_SlDnode0, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.k_digestrate)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_CPsl_metrics(t, y, args_sl)
        
        v_compute = jax.vmap(compute_metrics)
        dSlDdt_all, flux_all, C_all = v_compute(self.SlDexit_SS.ts, self.SlDexit_SS.ys)
        
        self.df_SlP_d = pd.DataFrame({
            't': np.array(self.SlDexit_SS.ts),
            'dDSldt': np.array(dSlDdt_all[:, -1]),
            'flux_CPsl_DuoJej': np.array(flux_all),
            'Conc_CPsl_Duolast': np.array(C_all),
            'QCPsl_duo': np.array(self.SlDexit_SS.ys[:, -1])
        })
        return self.df_SlP_d, None

    @time_it
    def SlP_for_RP_d(self):
        self.flatten_result_duo_CPsl()
        # Not strictly needed for diffrax anymore, but returning just in case anything else uses it
        df_Q_CPsl_Duo = self.df_SlP_d[['t', 'QCPsl_duo']] 
        df_Q_CPsl_Duo.reset_index(drop=True, inplace=True) 
        Duo_CPsl_Q = df_Q_CPsl_Duo['QCPsl_duo'].values 
        return df_Q_CPsl_Duo, Duo_CPsl_Q

    @time_it
    def solving_duo_R(self):
        self.df_Q_CPsl_Duo, self.Duo_CPsl_Q = self.SlP_for_RP_d()

        self.init_Duo_CPr[0] = float(self.result_RDnode0.ys[-1, 0])
        y0_r = jnp.array(self.init_Duo_CPr)
        
        args_r = (self.result_RDnode0, self.SlDexit_SS, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.k_absp, self.k_digestrate)
        
        # Original was LSODA, Tsit5 or Dopri5 is suitable for non-stiff or moderately stiff
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]

        self.RDexit_SS = diffrax.diffeqsolve(
            diffrax.ODETerm(_method_of_lines_CPr_Duo_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0_r, args=args_r, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )

    def flatten_result_duo_CPr(self):
        args_r = (self.result_RDnode0, self.SlDexit_SS, self.VF, jnp.array(self.DuoV_cm3), self.Kp_Duo_min, self.Duo_single_node_V, self.k_absp, self.k_digestrate)
        
        @jax.jit
        def compute_metrics(t, y):
            return _method_of_lines_CPr_metrics(t, y, args_r)
        
        v_compute = jax.vmap(compute_metrics)
        dRDdt_all, flux_all, C_all = v_compute(self.RDexit_SS.ts, self.RDexit_SS.ys)
        
        self.df_RP_d = pd.DataFrame({
            't': np.array(self.RDexit_SS.ts),
            'dRDdt': np.array(dRDdt_all[:, -1]),
            'flux_CPr_DuoJej': np.array(flux_all),
            'C_CPr__Duolast_gcm3': np.array(C_all),
            'QCPr_duo': np.array(self.RDexit_SS.ys[:, -1])
        })
        return self.df_RP_d, None

    def solving_duo_feed(self):
        fore_t = jnp.array(self.result_fore.t)
        fore_y = jnp.array(self.result_fore.y.T)
        args_feed = (fore_t, fore_y, self.Kp_PVG_min, self.Kp_Duo_min)
        
        solver = diffrax.Tsit5()  
        stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
        saveat = diffrax.SaveAt(ts=jnp.array(self.t_eval), dense=True)
        t0 = self.t_span[0]
        t1 = self.t_span[1]
        y0 = jnp.array(self.iDuo_g)

        self.result_feed_duo = diffrax.diffeqsolve(
            diffrax.ODETerm(_feed_duo_jax), solver, t0=t0, t1=t1, dt0=0.1,
            y0=y0, args=args_feed, saveat=saveat, stepsize_controller=stepsize_controller, max_steps=100000
        )
    
    def flatten_result_duo_feed(self):
        fore_t = jnp.array(self.result_fore.t)
        fore_y = jnp.array(self.result_fore.y.T)
        args_feed = (fore_t, fore_y, self.Kp_PVG_min, self.Kp_Duo_min)
        
        @jax.jit
        def compute_metrics(t, y):
            return _feed_duo_metrics(t, y, args_feed)
            
        v_compute = jax.vmap(compute_metrics)
        dfeedduodt_all, flux_all, Q_all = v_compute(self.result_feed_duo.ts, self.result_feed_duo.ys)
        
        self.df_feed_d = pd.DataFrame({
            't': np.array(self.result_feed_duo.ts),
            'dfeedduodt': np.array(dfeedduodt_all[:, 0]),
            'flux_feed_DuoJej': np.array(flux_all[:, 0]),
            'Qfeed_duo': np.array(self.result_feed_duo.ys[:, 0])
        })
        return self.df_feed_d, None 
