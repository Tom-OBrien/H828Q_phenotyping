#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 15 May 14:32:18 2023

@author: tobrien

Script for generating combined heatmap of cua-1[H828Q] mutant & N2 untreated;
exposed to 100uM copper for 4h; reared on 100uM copper for 1/2 generations

"""
# Importing modules
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from scipy import stats
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.read_data.hydra_metadata import read_hydra_metadata, align_bluelight_conditions

# Import custom functions
sys.path.insert(0, '/Users/tobrien/analysis_repo/DiseaseModelling/hydra_screen/phenotype_summary')
from helper import (
                    select_strains,
                    filter_features,
                    make_colormaps,
                    STIMULI_ORDER)
from plotting_helper import  (plot_colormap,
                              plot_cmap_text,
                              make_heatmap_df, 
                              make_barcode, 
                              make_clustermaps)

# Set paths to files with metadata and experimental results:
FEAT_FILE =  Path('/Volumes/Ashur Pro2/cua-1/Results/features_summary_tierpsy_plate_20230220_173924.csv')
FNAME_FILE = Path('/Volumes/Ashur Pro2/cua-1/Results/filenames_summary_tierpsy_plate_20230220_173924.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/cua-1/AuxiliaryFiles/wells_updated_metadata.csv')

saveto = Path('/Users/tobrien/Documents/Imperial : MRC/cua-1/Paper_figures/Combined_heatmap')

# Select group of treatments/strains to be plotted together
STRAINS = ['H828Q',
           'WT 4h Exposure',
           'H828Q 4h Exposure',
           'WT 2nd Gen',
           'H828Q 1st Gen',
           'WT 1st Gen',
           'H828 2nd Gen' ]

selected_features = [
                   'd_curvature_mean_midbody_w_paused_abs_50th_prestim',
                   'speed_w_forward_50th_prestim',
                   'length_50th_prestim',
                   'motion_mode_forward_fraction_bluelight'
                   ]

CONTROL_STRAIN = 'WT'

#%% Import my data and filter accordingly:
if __name__ == '__main__':
    
    # Tierpsy tools function to import feature/metadata files
    feat, meta = read_hydra_metadata(FEAT_FILE,
                                     FNAME_FILE,
                                     METADATA_FILE)
    # Align features by prestim, blue light and poststim conditions
    feat, meta = align_bluelight_conditions(
                                feat, meta, how='inner')

    #Filter out nan worms and print .csv of these    
    nan_worms = meta[meta.worm_gene.isna()][[
                                            'featuresN_filename',
                                            'well_name',
                                            'imaging_plate_id',
                                            'instrument_name',
                                            'date_yyyymmdd']]
    nan_worms.to_csv(
        METADATA_FILE.parent / 'my_nan_worms.csv', index=False)
    print('{} nan worms - my data'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)

    # Rename genes to be in line with paper names
    meta.worm_gene.replace({'N2':'WT',
                            'cua-1':'H828Q'},  
                           inplace=True)    
    
    # Converting date to nicer format for plotting
    meta['date_yyyymmdd'] = pd.to_datetime(
        meta['date_yyyymmdd'], format='%Y%m%d').dt.date

    # Filtering feature set with Tierpsy Tools function
    feat_df, meta_df, featsets = filter_features(feat,
                                                 meta)
    #%% Filter nans with tierpsy tools function
    feat_df = filter_nan_inf(feat_df, 0.5, axis=1, verbose=True)
    meta_df = meta.loc[feat_df.index]
    feat_df = filter_nan_inf(feat_df, 0.05, axis=0, verbose=True)
    feat_df = feat_df.fillna(feat_df.mean())

    #%% Remove wells annotated as bad
    n_samples = meta_df.shape[0]
    bad_well_cols = [col for col in meta_df.columns if 'is_bad' in col]
    bad = meta_df[bad_well_cols].any(axis=1)
    meta_df = meta_df.loc[~bad,:]
    
    #%% Combine worm gene and treatment info
    meta_df['analysis'] = meta_df['worm_gene'] + '_' + meta_df['treatment_condition']
    meta_df.analysis.replace({'WT_none':'WT',
                              'H828Q_none':'H828Q',
                              'WT_exposed_100uM_CuCl2':'WT 4h Exposure',
                              'H828Q_exposed_100uM_CuCl2': 'H828Q 4h Exposure',
                              'WT_reared_on_100uM_CuCl2': 'WT 2nd Gen',
                              'H828Q_reared_on_100uM_CuCl2': 'H828Q 1st Gen',
                              'WT_gen2_reared_on_100uM_CuCl2': 'WT 1st Gen',
                              'H828Q_gen2_reared_on_100uM_CuCl2': 'H828 2nd Gen'
                              }, inplace=True)
    # Update worm gene column with this information
    meta_df['worm_gene'] = meta_df['analysis']
    
    #%%
    # Make a list of unique genes in metadata that differ from control
    genes = [g for g in meta_df.worm_gene.unique() if g != CONTROL_STRAIN]

    # Modeule to select only the strains of interest from feature matrix
    feat_df, meta_df, idx, gene_list = select_strains(candidate_gene=STRAINS,
                                                    control_strain=CONTROL_STRAIN,
                                                    feat_df=feat_df,
                                                    meta_df=meta_df)

    
    # Make a colour map of unique colours for each strain and order
    strain_lut, stim_lut, feat_lut = make_colormaps(gene_list,
                                                    featlist=featsets['all'],
                                                    idx=idx,
                                                    candidate_gene=STRAINS
                                                    )
    
    #%% Now we plot a heatmap of the data
    
    # Hardcoded colour map to fit with existing paper figure styles
    strain_lut = {'WT':'lightgrey',
                  'H828Q':'salmon',                 
                  'WT 4h Exposure':'lightgrey',
                  'H828Q 4h Exposure':'salmon',                 
                  'WT 2nd Gen':'lightgrey',
                  'H828Q 1st Gen':'salmon',
                  'WT 1st Gen':'lightgrey',
                  'H828 2nd Gen':'salmon'
                  }
    
    # Use Tierpsy Tools to impute nans in the data
    feat_nonan = impute_nan_inf(feat_df)
    
    # Calculate Z-scores of features and save as data frame
    # featsets is an output of the filter_features module (above)
    featZ = pd.DataFrame(data=stats.zscore(feat_nonan[featsets['all']], axis=0),
                         columns=featsets['all'],
                         index=feat_nonan.index)
    
    # Double check that no nan's exist in the Z-score feature matrix
    assert featZ.isna().sum().sum() == 0    
    
    # N2_clustered_features.txt is a file made during the 'calculate stats' 
    # script, it contains how features are ordered 
    
    # Find N2clustered feats file 
    N2clustered_features = {}
    for fset in STIMULI_ORDER.keys():
        with open(saveto /  'N2_clustered_features_{}.txt'.format(fset), 'r') as fid:
            N2clustered_features[fset] = [l.rstrip() for l in fid.readlines()]            
    with open(saveto / 'N2_clustered_features_{}.txt'.format('all'), 'r') as fid:
        N2clustered_features['all'] = [l.rstrip() for l in fid.readlines()]

    N2clustered_features_copy = N2clustered_features.copy()
    (saveto / 'heatmaps').mkdir(exist_ok=True)
    
        # Creating a heatmap containing a colour-coded map of stim conditions
    for stim,fset in featsets.items():
        heatmap_df = make_heatmap_df(N2clustered_features_copy[stim],
                                     featZ[fset],
                                     meta_df)
        # Plotting heatmap and colouring according to z-score
        make_barcode(heatmap_df,
                     selected_features,
                     cm=['inferno']*(heatmap_df.shape[0]-1)+['Pastel1'],
                     vmin_max=[(-2,2)]*(heatmap_df.shape[0]-1)+[(1,3)])

        plt.savefig(saveto / 'heatmaps' / '{}_heatmap.png'.format(stim))
    # Save everything
    (saveto / 'clustermaps').mkdir(exist_ok=True)
    clustered_features = make_clustermaps(featZ,
                                          meta_df,
                                          featsets,
                                          strain_lut,
                                          feat_lut,
                                           group_vars = ['worm_gene'],
                                          saveto=saveto / 'clustermaps')
    
