import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_DIR / "resources"
diets_file = RESOURCE_DIR / "Plug_Flow_Diet_Charac.csv"

# Load diet constants at import time
if diets_file.exists():
    plug_flow_diets = pd.read_csv(diets_file)
    ingr_const = plug_flow_diets.set_index('Ingredient').T.to_dict()
else:
    plug_flow_diets = None
    ingr_const = {}

def get_ing_const(ingr_name):
    """
    Retrieves the constants for a specific feed ingredient.
    Returns None if the ingredient does not exist.
    """
    ingr_name_lower = ingr_name.lower()
    if ingr_name_lower in ingr_const:
        return ingr_const[ingr_name_lower]
    return None

def plot_git_results(df_fore, image_dir="images"):
    """
    Plots the crop/gizzard protein quantities and fluxes over time
    and saves the plots to the specified image directory.
    """
    os.makedirs(image_dir, exist_ok=True)
    
    plt.plot(df_fore['t'], df_fore['Cr'], color='black')
    plt.plot(df_fore['t'], df_fore['PVG'], color='blue')
    plt.xlabel('Time')
    plt.ylabel('Protein (g) in the gizzard/proventriculus')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'B-protein-over-time-proventriculus_gizzard.png'))
    
    plt.clf() 
    plt.plot(df_fore['t'], df_fore['CrPVG_flux'], color='black')
    plt.plot(df_fore['t'], df_fore['PVGDuo_flux'], color='blue')
    plt.subplots_adjust(left=0.15)
    plt.xlabel('Time')
    plt.ylabel('flux of protein in the crop and PVG')
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'B-protein-over-time-proventriculus_gizzard_flux.png'))

def export_duojejil_results(duodenum_instance, jejunum_instance, ileum_instance, output_dir="Python_VSCode_Files"):
    """
    Exports the simulation results for Duodenum, Jejunum, and Ileum 
    to Excel spreadsheets in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    def export_duo():
        UDexit_SS = duodenum_instance.UDexit_SS
        SlDexit_SS = duodenum_instance.SlDexit_SS
        RDexit_SS = duodenum_instance.RDexit_SS

        df_UP_d = duodenum_instance.df_UP_d
        df_SlP_d = duodenum_instance.df_SlP_d
        df_RP_d = duodenum_instance.df_RP_d
        
        df_UD = pd.DataFrame(UDexit_SS.y.T, columns=[f'Node_{i}' for i in range(UDexit_SS.y.shape[0])])
        df_UD['Time'] = UDexit_SS.t
        df_UD = df_UD[['Time'] + [col for col in df_UD.columns if col != 'Time']]

        df_SlD = pd.DataFrame(SlDexit_SS.y.T, columns=[f'Node_{i}' for i in range(SlDexit_SS.y.shape[0])])
        df_SlD['Time'] = SlDexit_SS.t
        df_SlD = df_SlD[['Time'] + [col for col in df_SlD.columns if col != 'Time']]
        
        df_RD = pd.DataFrame(RDexit_SS.y.T, columns=[f'Node_{i}' for i in range(RDexit_SS.y.shape[0])])
        df_RD['Time'] = RDexit_SS.t
        df_RD = df_RD[['Time'] + [col for col in df_RD.columns if col != 'Time']]
        
        with pd.ExcelWriter(os.path.join(output_dir, "model_results_duo-Bryan.xlsx"), engine='openpyxl') as writer:
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
        
        df_UJ = pd.DataFrame(UJexit_SS.y.T, columns=[f'Node_{i}' for i in range(UJexit_SS.y.shape[0])])
        df_UJ['Time'] = UJexit_SS.t
        df_UJ = df_UJ[['Time'] + [col for col in df_UJ.columns if col != 'Time']]

        df_SlJ = pd.DataFrame(SlJexit_SS.y.T, columns=[f'Node_{i}' for i in range(SlJexit_SS.y.shape[0])])
        df_SlJ['Time'] = SlJexit_SS.t
        df_SlJ = df_SlJ[['Time'] + [col for col in df_SlJ.columns if col != 'Time']]
                
        df_RJ = pd.DataFrame(RJexit_SS.y.T, columns=[f'Node_{i}' for i in range(RJexit_SS.y.shape[0])])
        df_RJ['Time'] = RJexit_SS.t
        df_RJ = df_RJ[['Time'] + [col for col in df_RJ.columns if col != 'Time']]
           
        with pd.ExcelWriter(os.path.join(output_dir, "model_results_jej-Bryan.xlsx"), engine='openpyxl') as writer:
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
        
        df_UI = pd.DataFrame(UIexit_SS.y.T, columns=[f'Node_{i}' for i in range(UIexit_SS.y.shape[0])])
        df_UI['Time'] = UIexit_SS.t
        df_UI = df_UI[['Time'] + [col for col in df_UI.columns if col != 'Time']]
        
        df_SlI = pd.DataFrame(SlIexit_SS.y.T, columns=[f'Node_{i}' for i in range(SlIexit_SS.y.shape[0])])
        df_SlI['Time'] = SlIexit_SS.t
        df_SlI = df_SlI[['Time'] + [col for col in df_SlI.columns if col != 'Time']]
        
        df_RI = pd.DataFrame(RIexit_SS.y.T, columns=[f'Node_{i}' for i in range(RIexit_SS.y.shape[0])])
        df_RI['Time'] = RIexit_SS.t
        df_RI = df_RI[['Time'] + [col for col in df_RI.columns if col != 'Time']]
      
        with pd.ExcelWriter(os.path.join(output_dir, "model_results_ileum-Bryan.xlsx"), engine='openpyxl') as writer:
            df_UI.to_excel(writer, sheet_name='Ileum_undegradable', index=False)
            df_UP_i.to_excel(writer, sheet_name='Ileum_undegrable_atexit', index=False)
            df_SlI.to_excel(writer, sheet_name='Ileum_sl_degradable', index=False)
            df_SlP_i.to_excel(writer, sheet_name='Ileum_sl_degrable_atexit', index=False)
            df_RI.to_excel(writer, sheet_name='Ileum_rapid_degrad', index=False)
            df_RP_i.to_excel(writer, sheet_name='Ileum_rapid_degrad_atexit', index=False)

    export_duo()    
    export_jej()
    export_il()
    print(f"done exporting to {output_dir}")
