#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:43:31 2023

Script for plotting the chosen behavioural features

@author: tobrien
"""
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from tierpsytools.preprocessing.filter_data import filter_nan_inf
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf
from tierpsytools.read_data.hydra_metadata import (align_bluelight_conditions,
                                                   read_hydra_metadata)

ROOT_DIR = Path('/Users/tobrien/Documents/Imperial : MRC/cua-1/Analysis')
FEAT_FILE =  Path('/Volumes/Ashur Pro2/cua-1/Results/features_summary_tierpsy_plate_20230220_173924.csv')
FNAME_FILE = Path('/Volumes/Ashur Pro2/cua-1/Results/filenames_summary_tierpsy_plate_20230220_173924.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/cua-1/AuxiliaryFiles/wells_updated_metadata.csv')

save_dir = Path('/Users/tobrien/Documents/Imperial : MRC/cua-1/Paper_figures')

CONTROL_STRAIN = 'WT'  

FEATURE = ['d_curvature_mean_midbody_w_paused_abs_50th_prestim',
           'speed_w_forward_50th_prestim',
           'length_50th_prestim']

#%%
if __name__ == '__main__':
    
    # Read in data and align by bluelight using tierpsy tools
    feat, meta = read_hydra_metadata(
        FEAT_FILE,
        FNAME_FILE,
        METADATA_FILE)

    feat, meta = align_bluelight_conditions(feat, meta, how='inner')
    
    # rename strains to fit with existing paper nomenclature:
    meta.worm_gene.replace({'N2':'WT',
                           'cua-1':'H828Q'}, inplace=True)
    
    # Only keeping data with 100uM CuCl2
    conc_to_keep = ['exposed_100uM_CuCl2',
                    'reared_on_100uM_CuCl2',
                    'gen2_reared_on_100uM_CuCl2']
    meta = meta[meta['treatment_condition'].isin(conc_to_keep)]
    feat = feat.loc[meta.index]    
    
    # I then rename for plotting
    meta.treatment_condition.replace({'exposed_100uM_CuCl2':'4h',
                                      'reared_on_100uM_CuCl2':'Gen 1',
                                      'gen2_reared_on_100uM_CuCl2': 'Gen 2'},
                                     inplace=True)

    # Converting date into nicer format for plitting
    meta['date_yyyymmdd'] = pd.to_datetime(
        meta['date_yyyymmdd'], format='%Y%m%d').dt.date
    
    # I know filter out nan values and save these as a .csv
    nan_worms = meta[meta.worm_gene.isna()][['featuresN_filename',
                                             'well_name',
                                             'imaging_plate_id',
                                             'instrument_name',
                                             'date_yyyymmdd']]

    nan_worms.to_csv(
        METADATA_FILE.parent / 'nan_worms.csv', index=False)
    print('{} nan worms'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)            

    # Print out number of features processed
    feat_filters = [line for line in open(FNAME_FILE) 
                     if line.startswith("#")]
    print ('Features summaries were processed \n {}'.format(
           feat_filters))
    
    # Make summary .txt file of feats
    with open(ROOT_DIR / 'feat_filters_applied.txt', 'w+') as fid:
        fid.writelines(feat_filters)
    
    # Extract genes in metadata different from control strain and make a list
    # of the total number of strains
    genes = [g for g in meta.worm_gene.unique() if g != CONTROL_STRAIN]
                                    
    
    #%% Filter nans with tierpsy tools function
    feat = filter_nan_inf(feat, 0.5, axis=1, verbose=True)
    meta = meta.loc[feat.index]
    feat = filter_nan_inf(feat, 0.05, axis=0, verbose=True)
    feat = feat.fillna(feat.mean())
    
    # Impute nans from feature dataframe
    feat_nonan = impute_nan_inf(feat)

    # Remove wells annotated as bad
    mask = [1.0]
    meta = meta[meta['well_label'].isin(mask)]
    feat_nonan = feat_nonan.loc[meta.index]
    
    #%% Generate a strain lut for control and mutant
    strain_lut = {'WT': 'darkgrey',
                  'H828Q' : 'salmon'}
    
    date_lut = {'2023-02-08':'white',
                '2023-02-09':'grey',
                }
    
    # concatenate feature matrix and metadata into one dataframe
    data=pd.concat([feat_nonan,meta],
                   axis=1)
    
    # Setting plot style
    plt.rcParams["font.family"] = "arial"
    for f in FEATURE:
        # First generate boxplot
        ax = sns.boxplot(y=f,
                         x='treatment_condition',
                         data=data,
                         hue='worm_gene',
                         palette=strain_lut.values(),
                         showfliers=False,
                )
        
        # Now generate strip plot of individual datapoints
        ax = sns.stripplot(y=f,
                x='treatment_condition',
                data=data,
                hue='worm_gene',
                palette=strain_lut.values(),
                alpha=0.3,
                dodge=True,
                label='')
        # Remove default seaborn legend 
        ax.get_legend().remove()       
        
        # Add in new legend using mpatches
        WT = mpatches.Patch(color='darkgrey', label='WT')
        H828Q = mpatches.Patch(color='salmon', label='H828Q')
        ax.legend(handles=[WT, H828Q], loc='upper left')
        
        # Remove x label 
        plt.xlabel('')
        
        # Rename y label according to feature
        if f=='d_curvature_mean_midbody_w_paused_abs_50th_prestim':
            plt.ylabel('Midbody curvature/Time' ' (rads ' '\u03bcm' '$^{-1}$' '/s)')
            
        if f=='speed_w_forward_50th_prestim':
            plt.ylabel('Speed ' '(\u03bcm'' s''$^{-1}$'')')
            
        if f=='length_50th_prestim':
            plt.ylabel('Length ' '(\u03bcm' ')')

        # Fit figure to size and save 
        plt.tight_layout()
        plt.savefig(save_dir/ '{}.png'.format(f), bbox_inches='tight',
                    dpi=300)
        plt.close('all')
        plt.show()
        