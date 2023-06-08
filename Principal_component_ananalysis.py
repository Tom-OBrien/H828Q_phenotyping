#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 15:02:53 2021

This script performs principal coordinate analysis and plots the first 2 
components for the first generation of H828Q and wild-type worms reared on
increasing concentrations of Cu. I also plot the 100uM CuCl2 4h an F2 data

@author: tobrien
"""
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.patches as mpatches
from tierpsytools.read_data.hydra_metadata import (read_hydra_metadata,
                                                   align_bluelight_conditions)
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from sklearn.decomposition import PCA
from tierpsytools.preprocessing.preprocess_features import impute_nan_inf

sys.path.insert(0, '/Users/tobrien/analysis_repo/DiseaseModelling/hydra_screen/phenotype_summary')
from helper import (filter_features,
                    make_colourmaps_two_controls)


#%% First set data paths and save directory
FEAT_FILE =  Path('/Volumes/Ashur Pro2/cua-1/Results/features_summary_tierpsy_plate_20230220_173924.csv')
FNAME_FILE = Path('/Volumes/Ashur Pro2/cua-1/Results/filenames_summary_tierpsy_plate_20230220_173924.csv')
METADATA_FILE = Path('/Users/tobrien/Documents/Imperial : MRC/cua-1/AuxiliaryFiles/wells_updated_metadata.csv')

saveto = Path('/Users/tobrien/Documents/Imperial : MRC/cua-1/Paper_figures/PCA/bluelight')
saveto.mkdir(exist_ok=True)

# Mutant Strain
TARGET_STRAIN = 'H828Q'
# Control strain + DMSO only
CONTROL_STRAIN = 'WT'
controls = [TARGET_STRAIN, CONTROL_STRAIN]

# List of treatment conditions we don't want to plot on these PCs
treatments_to_drop = [
        'exposed_25uM_CuCl2',
        'exposed_50uM_CuCl2',
        'exposed_75uM_CuCl2',
        'exposed_150uM_CuCl2',
        'exposed_200uM_CuCl2',
        'reared_on_75uM_CuCl2',
        'reared_on_150uM_CuCl2',
        'reared_on_200uM_CuCl2',
        'gen2_reared_on_25uM_CuCl2',
        'gen2_reared_on_50uM_CuCl2',
        'gen2_reared_on_75uM_CuCl2',
        'gen2_reared_on_150uM_CuCl2',
        'gen2_reared_on_200uM_CuCl2',
                            ]

#%% 
if __name__ == '__main__':
 
    # Read in data with Tierpsy Tools function
    feat, meta = read_hydra_metadata(FEAT_FILE,
                                     FNAME_FILE,
                                     METADATA_FILE)
    # Align feature data by period of acquisition i.e. blue light
    feat, meta = align_bluelight_conditions(
                                feat, meta, how='inner')

    #Filter out nan worms and print .csv of these    
    nan_worms = meta[meta.worm_gene.isna()][[
                                            'featuresN_filename',
                                            'well_name',
                                            'imaging_plate_id',
                                            'instrument_name',
                                            'date_yyyymmdd']]
    nan_worms.to_csv(METADATA_FILE.parent / 'my_nan_worms.csv', index=False)
    print('{} nan worms - my data'.format(nan_worms.shape[0]))
    feat = feat.drop(index=nan_worms.index)
    meta = meta.drop(index=nan_worms.index)

    #Rename genes to be consistent with the paper
    meta.worm_gene.replace({'N2':'WT',
                            'cua-1':'H828Q'},  
                           inplace=True)    
    # Remove unwanted treatment conditions (as defined above)
    meta = meta[~meta['treatment_condition'].isin(treatments_to_drop)]
    feat = feat.loc[meta.index] 

    # filter features
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
    
    #%% Add worm gene and treatment info together
    meta_df['analysis'] = meta_df['worm_gene'] + '_' + meta_df['treatment_condition']
    meta_df.analysis.replace({'WT_none':'WT',
                              'H828Q_none':'H828Q'}, inplace=True)
    
    # Find different the treatments in the data set
    treatments = [t for t in meta_df.analysis.unique() if not t in controls]
    treatments.sort()
    
    # Now make a colour map with wild-type/H828Q hardcoded
    CONTROL_DICT={TARGET_STRAIN:('salmon'),
                  CONTROL_STRAIN:('lightgrey')}    
    
    # Make a lut of individual drug treatments, with controls 
    drug_lut = make_colourmaps_two_controls(treatments, 
                                            CONTROL_STRAIN, 
                                            TARGET_STRAIN, 
                                            idx=[], 
                                            candidate_drug=None, 
                                            CONTROL_DICT=CONTROL_DICT)
    
    
    # %% Impute nans using Tierpsy Tools function
    feat_nonan = impute_nan_inf(feat_df)
    
    # Now build a list of bluelight only features
    featlist = list(feat_nonan.columns)
    featlist = [val for val in featlist if not val.endswith('stim')]

    # Concatenate featurematrix and analysis column from metadata
    featmat = pd.concat([feat_nonan,
                         meta_df.loc[:,'analysis']],
                         axis=1)
    
    # Calculate z-scores of data
    featmatZ = pd.DataFrame(data=stats.zscore(featmat[featlist],
                                              axis=0),
                            columns=featlist,
                            index=featmat.index)
    assert featmatZ.isna().sum().sum() == 0
    
    # %% Transform data with sklearn decomposition package
    pca = PCA()
    X2=pca.fit_transform(featmatZ.loc[:,featlist])
    
    # Explain PC variance using cumulative variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    thresh = cumvar <= 0.95 #set 95% variance threshold
    cut_off = int(np.argwhere(thresh)[-1])

    # Plt the explained variance as a figure
    plt.figure()
    plt.plot(range(0, len(cumvar)), cumvar*100)
    plt.plot([cut_off,cut_off], [0, 100], 'k')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Variance Explained')
    plt.tight_layout()
    plt.savefig(saveto / 'long_df_variance_explained.png', dpi =300)

    # %%
    # Now put the PC cutoffs into a dataframe, grouped by analysis column
    PCname = ['PC_%d' %(p+1) for p in range(0,cut_off+1)]
    PC_df = pd.DataFrame(data=X2[:,:cut_off+1],
                          columns=PCname,
                          index=featmatZ.index)

    PC_plotting = pd.concat([PC_df,
                              meta_df[['analysis']]],
                              axis=1)

    # groupby worm gene to see the trajectory through PC space
    PC_plotting_grouped = PC_plotting.groupby(
                                            ['analysis']).mean().reset_index()
    # PC_plotting_grouped['stimuli_order'] = PC_plotting_grouped['bluelight'].map(STIMULI_ORDER)
    PC_plotting_grouped.sort_values(by=['analysis'],
                                    ascending=True,
                                    inplace=True)

    mask = PC_plotting_grouped['analysis'].isin(controls)
    PC_plotting_grouped_no_controls = PC_plotting_grouped[~mask]
    PC_plotting_grouped_controls = PC_plotting_grouped[mask]
    
    # Calculate standard error of mean of PC matrix computed above
    PC_plotting_sd = PC_plotting.groupby(['analysis']).sem().reset_index()
    mask = PC_plotting_sd['analysis'].isin(controls)
    PC_plotting_sd_no_controls = PC_plotting_sd[~mask]
    PC_plotting_sd_controls = PC_plotting_sd[mask]  
    
    # Now I separate out all the data into individual DFs so I can plot nicely
    # 1) 4h wild-type
    exposed_WT = ['WT_exposed_25uM_CuCl2',
                'WT_exposed_50uM_CuCl2',
                'WT_exposed_100uM_CuCl2']
    mask = PC_plotting_grouped['analysis'].isin(exposed_WT)
    PC_4h_WT = PC_plotting_grouped_no_controls[mask]
    PC_sd_4h_WT = PC_plotting_sd_no_controls[mask]
    
    # 2) 4h mutant
    exposed_mutant = ['H828Q_exposed_25uM_CuCl2',
                      'H828Q_exposed_50uM_CuCl2',
                      'H828Q_exposed_100uM_CuCl2']
    mask = PC_plotting_grouped['analysis'].isin(exposed_mutant)
    PC_4h_mutant = PC_plotting_grouped_no_controls[mask]
    PC_sd_4h_mutant = PC_plotting_sd_no_controls[mask]
    
    # 3) F1 wild-type
    gen_1_WT = ['WT_reared_on_25uM_CuCl2',
                'WT_reared_on_50uM_CuCl2',
                'WT_reared_on_100uM_CuCl2']
    mask = PC_plotting_grouped['analysis'].isin(gen_1_WT)
    PC_gen2_WT = PC_plotting_grouped_no_controls[mask]
    PC_sd_gen1_WT = PC_plotting_sd_no_controls[mask]
    
    # 4) F1 mutant
    gen_1_mutant = ['H828Q_reared_on_25uM_CuCl2',
                    'H828Q_reared_on_50uM_CuCl2',
                    'H828Q_reared_on_100uM_CuCl2']
    mask = PC_plotting_grouped['analysis'].isin(gen_1_mutant)
    PC_gen1_mutant = PC_plotting_grouped_no_controls[mask]
    PC_sd_gen1_mutant = PC_plotting_sd_no_controls[mask]

    # 5) F2 wild-type
    gen_2_WT = ['WT_gen2_reared_on_25uM_CuCl2',
                'WT_gen2_reared_on_50uM_CuCl2',
                'WT_gen2_reared_on_100uM_CuCl2']
    mask = PC_plotting_grouped['analysis'].isin(gen_2_WT)
    PC_gen1_WT = PC_plotting_grouped_no_controls[mask]
    PC_sd_gen2_WT = PC_plotting_sd_no_controls[mask]
    
    # 6) F2 mutant
    gen_2_mutant = ['H828Q_gen2_reared_on_25uM_CuCl2',
                    'H828Q_gen2_reared_on_50uM_CuCl2',
                    'H828Q_gen2_reared_on_100uM_CuCl2']
    mask = PC_plotting_grouped['analysis'].isin(gen_2_mutant)
    PC_gen2_mutant = PC_plotting_grouped_no_controls[mask]
    PC_sd_gen2_mutant = PC_plotting_sd_no_controls[mask]
    
    # Separate mutant and wt non-treated controls
    wt_no_treat = ['WT']
    mask = PC_plotting_grouped_controls['analysis'].isin(wt_no_treat)
    PC_wt_no_treat = PC_plotting_grouped_controls[mask]

    mutant_no_treat = ['H828Q']
    mask = PC_plotting_grouped_controls['analysis'].isin(mutant_no_treat)
    PC_mutant_no_treat = PC_plotting_grouped_controls[mask]
    # %% Plot principal components and sd of points, here I control separate
    # style elements for each group made above- allows for fine control 

    ax1 = plt.figure(figsize = [15,15])
    marker_size = 800
    # Colour of treated mutant and wild-type plots
    colour_WT = ['darkgrey']
    colour_mutant = ['salmon']
    
    # Plot 4h exposure
    ax1 = plt.scatter(x='PC_1',
                    y='PC_2',
                    data=PC_4h_WT,
                    c=colour_WT,
                    s=marker_size,
                    marker='o',
                    label=''
                    )
    plt.scatter(x='PC_1',
                    y='PC_2',
                    data=PC_4h_mutant,
                    c=colour_mutant,
                    s=marker_size,
                    marker='o',
                    label=''
                    )
    
    # Plot F1s
    plt.scatter(x='PC_1',
                    y='PC_2',
                    data=PC_gen2_WT,
                    c=colour_WT,
                    s=marker_size,
                    marker='s',
                    label='WT'
                    )
    plt.scatter(x='PC_1',
                    y='PC_2',
                    data=PC_gen1_mutant,
                    c=colour_mutant,
                    s=marker_size,
                    marker='s',
                    label='H828Q'
                    )   
    
    # Plot F2s
    plt.scatter(x='PC_1',
                    y='PC_2',
                    data=PC_gen1_WT,
                    c=colour_WT,
                    s=marker_size,
                    marker='v',
                    label=''
                    )
    plt.scatter(x='PC_1',
                    y='PC_2',
                    data=PC_gen2_mutant,
                    c=colour_mutant,
                    s=marker_size,
                    marker='v',
                    label=''
                    )        
    
    # Error bars are all in the same dataframe, so just plot as one
    plt.errorbar(x=PC_plotting_grouped_no_controls['PC_1'],
                y=PC_plotting_grouped_no_controls['PC_2'],
                xerr=PC_plotting_sd_no_controls['PC_1'], 
                yerr=PC_plotting_sd_no_controls['PC_2'],
                fmt='.',
                c='grey',
                alpha=0.15,
                )
    
    # Plot controls with a seperate marker style and colour to distinguish
    control_mutant=['salmon']  #salmon=mutant
    control_wt = ['darkgrey']
    
    plt.scatter(x='PC_1',
                    y='PC_2',
                    data=PC_wt_no_treat,
                    c=control_wt,
                    linewidth=0,
                    s=1750,
                    marker='*',
                    label=''
                    )
                    
    plt.scatter(x='PC_1',
                    y='PC_2',
                    data=PC_mutant_no_treat,
                    c=control_mutant,
                    linewidth=0,
                    s=1750,
                    marker='*',
                    label='',
                    )
    plt.errorbar(
                x=PC_plotting_grouped_controls['PC_1'],
                y=PC_plotting_grouped_controls['PC_2'],
                xerr=PC_plotting_sd_controls['PC_1'], 
                yerr=PC_plotting_sd_controls['PC_2'],
                c='grey',
                fmt='*',
                alpha=0.15,
                )    
    
    # Set axes subplot as a figure (calling legend won't work otherwise)
    fig = ax1.get_figure()
    
    plt.autoscale(enable=True, axis='both')
    plt.axis('equal')
    plt.xlabel('PC 1 ({}%)'.format(np.round(pca.explained_variance_ratio_[0]*100,2)),
               fontsize=40)
    plt.ylabel('PC 2 ({}%)'.format(np.round(pca.explained_variance_ratio_[1]*100,2)),
               fontsize=40)               
    plt.xticks(fontsize=36)         
    plt.yticks(fontsize=36)  
    
    # Remove default legend, and then add in new formatted legend
    # ax1.get_legend().remove()
    WT = mpatches.Patch(color='darkgrey', label='WT')
    H828Q = mpatches.Patch(color='salmon', label='H828Q')
    plt.legend(handles=[WT, H828Q], loc='upper right',
                    fontsize=28)

    fig.tight_layout()
    fig.savefig(saveto / 'PC1_PC2_F1s_and_100uM_conc_only_bluelight', dpi=300)
    plt.show()    
    plt.close('all')
    