#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 22 May 10:11:26 2023

Calculate motion modes and generate timeseries plots of these

@author: tobrien

"""

import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from tierpsytools.hydra.platechecker import fix_dtypes
import time

sys.path.insert(0, '/Users/tobrien/analysis_repo/DiseaseModelling/hydra_screen/phenotype_summary')
from helper import (select_strains,
                    make_colormaps,
                    strain_gene_dict)
from ts_helper import (align_bluelight_meta,
                       make_feats_abs,
                       plot_strains_ts,
                       get_motion_modes,
                       plot_frac_all_modes_coloured_by_motion_mode,
                       plot_frac_by_mode,
                       plot_frac_all_modes,
                       MODECOLNAMES,
                       short_plot_frac_by_mode)

from luigi_helper import load_bluelight_timeseries_from_results

#%%
# Choose whether to recalculate motion modes or load from existing .hdf5 file 
is_reload_timeseries_from_results = True
is_recalculate_frac_motion_modes = True

# Choose control and mutant strains
CONTROL_STRAIN = 'WT'
candidate_gene = 'H828Q'
gene_list = [candidate_gene]

# Set paths to data
TS_METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/cua-1/AuxiliaryFiles/wells_updated_metadata.csv')
RAW_DATA_DIR = Path('/Volumes/Ashur Pro2/cua-1')
ANALYSIS_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/cua-1/Paper_figures/Updated_strain_TimeSeries/Gen2_100uM')
saveto = ANALYSIS_DIR
timeseries_fname = Path('/Users/tobrien/Documents/Imperial : MRC/cua-1/Paper_figures/Updated_strain_TimeSeries/Gen2_100uM') / '{}_timeseries.hdf5'.format(candidate_gene)

# Read in the metadata file
meta_ts = pd.read_csv(TS_METADATA_FILE,
                                  index_col=None)

# Some functions rely on having date file with different headers- I just copy info
imaging_date_yyyymmdd = meta_ts['date_yyyymmdd']
imaging_date_yyyymmdd = pd.DataFrame(imaging_date_yyyymmdd)
meta_ts['imaging_date_yyyymmdd'] = imaging_date_yyyymmdd 
meta_ts.loc[:,'imaging_date_yyyymmdd'] = meta_ts['imaging_date_yyyymmdd'].apply(lambda x: str(int(x)))

# Drop wells with nan values
meta_ts.dropna(axis=0,
               subset=['worm_gene'],
               inplace=True) 
            
# Remove bad wells
n_samples = meta_ts.shape[0]
bad_well_cols = [col for col in meta_ts.columns if 'is_bad' in col]
bad = meta_ts[bad_well_cols].any(axis=1)
meta_ts = meta_ts.loc[~bad,:]
            
# I'm only interested in plotting one condition, this could be changed for any
treatment_to_keep = ['gen2_reared_on_100uM_CuCl2']
meta_ts = meta_ts[meta_ts['treatment_condition'].isin(treatment_to_keep)]
            
# Filtering out missing videos from analysis
vids_to_drop = ['20230208/cucl2_run1_bluelight_20230208_143105.22956839']
meta_ts = meta_ts[~meta_ts['imgstore_name'].isin(vids_to_drop)] 

# Update number of worms per well and drop columns not required
meta_ts['number_worms_per_well'] = 3
meta_ts = meta_ts.drop(columns= ['date_bleached_yyyymmdd', 'date_refed_yyyymmdd'])
meta_ts = fix_dtypes(meta_ts)

#Replace gene name for paper
meta_ts.replace({'N2':'WT',
                 'cua-1':'H828Q'},
                inplace=True)
           
# Use helper function to select the data that we're interested in
meta_ts, idx, gene_list = select_strains([candidate_gene],
                                         CONTROL_STRAIN,
                                         meta_df=meta_ts)
# Make strain and stimuli colour maps
strain_lut_ts, stim_lut = make_colormaps(gene_list,
                                         [],
                                         idx,
                                         [candidate_gene])
  
# Here I actually hardcode the colour map to fit with other paper figs
strain_lut_ts = {'WT':'darkgrey',
                  'H828Q' :'salmon'}

# Make a dictionary of the strain names
strain_dict = strain_gene_dict(meta_ts)
gene_dict = {v:k for k,v in strain_dict.items()}

# Align by bluelight conditions 
meta_ts = align_bluelight_meta(meta_ts)

if is_reload_timeseries_from_results:
                # Uses tierpsy under hood to calculate timeseries and motion modes
                timeseries_df, hires_df  = load_bluelight_timeseries_from_results(
                                    meta_ts,
                                    RAW_DATA_DIR / 'Results',
                                    saveto=None)
                                    # save to disk when calculating (takes long time to run)
                try:
                    timeseries_df.to_hdf(timeseries_fname, 'timeseries_df', format='table')
                    hires_df.to_hdf(timeseries_fname, 'hires_df', format='table') #'fixed' may help hires_df be copied correctly, issues with files not being saved properly due to time it takes and HDD connection
                except Exception:
                    print ('error creating {} HDF5 file'.format(candidate_gene))
                    
else:  
    # Read hdf5 data and make DF
    timeseries_df = pd.read_hdf(timeseries_fname, 'timeseries_df')
    # hires_df = pd.read_hdf(timeseries_fname, 'hires_df')

#%% Add info about the replicates for each day/plate
date_to_repl = pd.DataFrame({
                            'date_yyyymmdd': [
                                             '20230208',
                                             '20230209',
                                             '20230210',
                                             '20230213',
                                             '20230214',
                                             '20230215'
                                             ],
                                         'replicate': [1, 2, 3, 1, 2 ,3]
                                         })  
   
# Merge replicates into single df
timeseries_df = pd.merge(timeseries_df, date_to_repl,
                                         how='left',
                                         on='date_yyyymmdd')
    
timeseries_df['worm_strain'] = timeseries_df['worm_gene'].map(gene_dict)

    
#make d/v signed features absolute as in hydra d/v is not assigned
timeseries_df = make_feats_abs(timeseries_df)

# %%% Plot hand-picked features from the downsampled dataframe

plt.close('all')
(saveto / 'ts_plots').mkdir(exist_ok=True)
feats_toplot = ['speed',
                'abs_speed',
                'angular_velocity',
                'abs_angular_velocity',
                'relative_to_body_speed_midbody',
                'abs_relative_to_body_speed_midbody',
                'abs_relative_to_neck_angular_velocity_head_tip',
                'speed_tail_base',
                'length',
                'major_axis',
                'd_speed',
                'head_tail_distance',
                'abs_angular_velocity_neck',
                'abs_angular_velocity_head_base',
                'abs_angular_velocity_hips',
                'abs_angular_velocity_tail_base',
                'abs_angular_velocity_midbody',
                'abs_angular_velocity_head_tip',
                'abs_angular_velocity_tail_tip',
                'd_curvature_std_head'
                ]

# Uses plotting helper to make lineplots with confidence intervals
# of handpicked features over time (includes bluelight bursts on fig)
plot_strains_ts(timeseries_df=timeseries_df,
                            strain_lut=strain_lut_ts,
                            CONTROL_STRAIN=CONTROL_STRAIN,
                            features=feats_toplot,
                            SAVETO=saveto / 'ts_plots')
plt.close('all')
#%% Plot entire motion modes
# Calculates motion modes (fraction of paused, fws/bck worms) and save as .hdf5
tic = time.time()
if is_recalculate_frac_motion_modes:
    motion_modes, frac_motion_modes_with_ci = get_motion_modes(hires_df,
                                                               saveto=timeseries_fname
                                                               )
# If motion modes already calculated, reload them 
else:
    frac_motion_modes_with_ci = pd.read_hdf(timeseries_fname,
                                            'frac_motion_mode_with_ci')

    frac_motion_modes_with_ci['worm_strain'] = frac_motion_modes_with_ci['worm_gene'].map(gene_dict)

    fps = 25
    frac_motion_modes_with_ci = frac_motion_modes_with_ci.reset_index()
    frac_motion_modes_with_ci['time_s'] = (frac_motion_modes_with_ci['timestamp']
                                           / fps)
    print('Time elapsed: {}s'.format(time.time()-tic))
                
#%% Utilising Luigi's boostraping functions to make ts plot            
# plot forwawrd,backward and stationary on one plot for each strain
# plots are coloured by cmap defined earlier on
            
for ii, (strain, df_g) in enumerate(frac_motion_modes_with_ci.groupby('worm_gene')):
                plot_frac_all_modes(df_g, strain, strain_lut_ts)
                plt.savefig(saveto / '{}_ts_motion_modes_coloured_by_strain.png'.format(strain), dpi=200)
                
#%% Same as above, but each motion mode coloured differently
                  
for iii, (strain, df_g) in enumerate(frac_motion_modes_with_ci.groupby('worm_gene')):
                plot_frac_all_modes_coloured_by_motion_mode(df_g, strain, strain_lut_ts)
                plt.savefig(saveto / '{}_ts_coloured_by_motion_modes.png'.format(strain), dpi=200)      
                
#%% Plot each motion mode separately
            
for motion_mode in MODECOLNAMES:
                plot_frac_by_mode(df=frac_motion_modes_with_ci, 
                                  strain_lut=strain_lut_ts, 
                                  modecolname=motion_mode)
                plt.savefig(saveto / '{}_ts.png'.format(motion_mode), dpi=200)
#%% First stimuli plots
time_drop = frac_motion_modes_with_ci['time_s']>160
frac_motion_modes_with_ci = frac_motion_modes_with_ci.loc[~time_drop,:]             
for motion_mode in MODECOLNAMES:
                short_plot_frac_by_mode(df=frac_motion_modes_with_ci, 
                                        strain_lut=strain_lut_ts, 
                                        modecolname=motion_mode)
                plt.savefig(saveto / '{}_first_stimuli_ts.png'.format(motion_mode), dpi=200)     
                
                        