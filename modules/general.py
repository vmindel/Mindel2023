import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from importlib import reload
import pickle
from Bio import SeqIO
import scipy.io as sio
import scipy
sns.set_style('white')
from importlib import reload
from scipy import stats
import re 
from Bio import motifs

def get_data(tf, data_check, rna_nonlibs, thresh=3):
    '''
    Get name of tf and return subset of data needed for analysis
        Parameters:
            tf (str): Name of TF
            data_check (pd.DataFrame): DataFrame of sumProm
            rna_nonlibs (pd.DataFrame): DataFrame of RNA-seq of nonlibs strains
            thresh (int): z-score threshold for definig targets of binding

    '''
    binding = return_family(data_check.filter(regex='^'+tf))
    rna = return_family_rna(rna_nonlibs.filter(regex='^'+tf))
    full = binding.filter(regex='{}_lab_data'.format(tf))
    if tf == 'Gal4':
        full = binding.filter(regex='Full').filter(regex='erv')
    elif tf == 'Gcn4':
        full = binding.filter(regex='{}_lown_lab_data'.format(tf))    
    top50 = full.loc[(get_zscored(full)>thresh).values].squeeze().sort_values(ascending=False).index.values[:100]
    fam = rna.filter(regex='tef|erv')
    fam.columns = ['_'.join(name.split('_')[:-1]) for name in fam.columns.values]
    fam = fam.transpose().reset_index().groupby('index').median().transpose().rename_axis(None, axis=1).loc[:, fam.columns.drop_duplicates()]
    return binding, rna, top50, fam, full

def return_family(data):
    '''
    Function to arange given DataFrame of Chec-Seq strains in a specified order
    input:
        * data -> pd.DataFrame of strains that need to be ordered
    output:
        * new_order -> pd.DataFrame newly ordered
    '''
    df_list = []
    data_no_gal11 = data.drop(data.filter(regex='Gal11').columns, axis=1)
    versions = ['deletions_tef', 'nodeletion_tef', 'aro80', 'erv', 'gal11']
    families = ['Full', 'Gal4AD', 'Gcn4AD', 'DBDGal11']    
    df_list.append(data.filter(regex='lab'))

    for fam in families:
        for version in versions:
            df_list.append(data_no_gal11.filter(regex=fam).filter(regex=version))
    df_list.append(data.filter(regex='Gal11'))
    new_order = pd.concat(df_list, axis=1)
    return new_order

def return_family_rna(data):
    '''
    Function to arange given DataFrame of RAN strains in a specified order
    input:
        * data -> pd.DataFrame of strains that need to be ordered
    output:
        * new_order -> pd.DataFrame newly ordered
    '''   
    df_list = []
    versions = ['deletions', 'nodeletion','libs', 'erv', 'gal11']
    families = ['Full', 'Gal4AD', 'Gcn4AD']    
    for fam in families:
        for version in versions:
            df_list.append(data.filter(regex=fam).filter(regex=version))
    df_list.append(data.filter(regex='BYd'))
    new_order = pd.concat(df_list, axis=1)
    return new_order

def get_corr(plot_material, f_size=4):
    plt.rcParams.update({'font.size': f_size, 'text.color': 'black'})

    families = []
    for i in ['lab', 'Full', 'Gal4AD', 'Gcn4AD']:
        families.append(len([name for name in plot_material.columns.values if re.search(i, name)]))
    cumsum_families = np.cumsum(np.array(families))
    
    fig, ax = plt.subplots(1, dpi=160)
    sns.heatmap(plot_material.corr(), ax=ax, cmap='Blues', linecolor='gainsboro', linewidth=0.1)
    
    for cmsm in cumsum_families:
        ax.axhline(cmsm, c='k')
        ax.axvline(cmsm, c='k')
    ax.text(-2,-1,'LAB', size=8)
    ax.text(cumsum_families[0]+1,-1, 'FULL', size=8)
    ax.text(cumsum_families[1]+1,-1, 'DBD_GAL4AD', size=8)
    ax.text(cumsum_families[2]+1,-1, 'DBD_GCN4AD', size=8)

    

def zscore_andpvals_df(df, mean, std, side=2):
    '''
    get dataframe of rna zscores and pvalues claculated from given mean and stdev (your control)
    input:
        * df -> pd.DataFrame of given strains
        * mean -> pd.DataFrame of shape df.shape with mean expression of your preffered control strains
        * stdev -> pd.DataFrame of shape df.shape with stdev expression of your preffered control strains
        * side -> int, 2 if not specified,  for two-sided comparison- 2, for one-sided- 1
    
    output:
        * pd.DataFrame of zscores
        * pd.DataFrame of p-values
    '''
    zscores_rna = df.sub(mean.values, axis=0).divide(std.values, axis=0)
    pvalues = pd.DataFrame(scipy.stats.norm.sf(np.abs(zscores_rna)), columns=zscores_rna.columns, index=zscores_rna.index)*side
    # pvalues = pd.DataFrame(scipy.stats.t.sf(np.abs(zscores_rna), df=zscores_rna.shape[1]-1), columns=zscores_rna.columns, index=zscores_rna.index)*side

    
    return zscores_rna, pvalues

def fdr_and_genes_passed(pvals, q=0.025):
    '''
    get DataFrame of genes, strains that pass FDR thresholding
    
    input:
    * pvals -> pd.DataFrame of form strains x genes and their respective pvalues
    * q -> int Fraction of allowed false discoveries for FDR
    output:
    * (fdr_passed_df, genes_passed):
        * pd.DataFrame stacked of genes pvalues and qvalues that passed statistical test
        * pd.Index of single occured genes that passed test
    '''
    stacked_pval = pvals.stack().reset_index().sort_values(by=0)
    N = pvals.shape[0]*pvals.shape[1]
    i = np.arange(1, N+1) # the 1-based i index of the p values, as in p(i)
    q_values =  q * i / N

    stacked_pval.loc[:, 'qvalues'] = q_values
    stacked_pval.index = stacked_pval.level_1

    fdr_passed = stacked_pval.loc[:, 0] < stacked_pval.qvalues
    fdr_passed_df = stacked_pval.loc[fdr_passed.values]


    genes_passed = pd.Index(fdr_passed_df.loc[pvals.columns].name).drop_duplicates()
    return fdr_passed_df, genes_passed, fdr_passed

def gene_order_pvalues(genes_passed, fc_df):
    '''
    get new order of genes for heatmap
    input:
        * pd.Index of genes that passed the statistical tests before
        * pd.DataFrame of fold change between selected strains and the control
    '''
    new_gene_oder = []
    gene_index = genes_passed
    gene_nums = []
    strict = []
    for column in fc_df.columns.values:
        passed = gene_index[(np.abs(fc_df.loc[gene_index, column]) > 0.4).values]
        to_extend = list(fc_df.loc[passed, column].sort_values(ascending=False).index.values)
        if len(to_extend) < 20:
            strict.extend(to_extend)
        else:
            strict.extend(to_extend[:10])
            strict.extend(to_extend[-10:])
            
        new_gene_oder.extend(to_extend)
        gene_nums.append(len(passed))
        gene_index = gene_index.drop(passed)
    new_gene_oder.extend(list(gene_index))

    return new_gene_oder, gene_nums, strict
def get_zscored(to_plot):
    '''
    Get zscores on promoters of DataFrame
    input:
    to_plot ->    
    '''
    zscored = pd.DataFrame(stats.zscore(to_plot.fillna(0)), index=to_plot.index, columns=to_plot.columns)
    return zscored

def get_tops(df, zscore_thresh=2):
    '''
    Extract DataFrame Indexed by relevant genes only
    input:
    df ->
    zscore_thresh ->
    
    '''
    tops = []
    get_tops_df = (df>zscore_thresh)
    for strain in get_tops_df.columns.values:
        tops.append(set(get_tops_df.loc[(get_tops_df.loc[:, strain] == True)].index.values))

    genes = list(set.union(*tops))
    to_screen = df.loc[genes]
    return to_screen

def get_gene_order(to_screen, experiment, msn2=False, thresh=2):
    
    if experiment =='check':
        versions = ['deletions_tef', 'nodeletion_tef', 'aro80', 'erv', 'gal11']
        families = ['Full', 'Gal4AD', 'Gcn4AD', 'DBDGal11']
    elif experiment == 'rna':
        versions = ['deletions', 'nodeletion','libs', 'erv', 'gal11']
        families = ['Full', 'Gal4AD', 'Gcn4AD'] 
    else:
        print('NONOOONO')
        return
    
    gene_index = to_screen.index
    new_gene_oder = []
    strains = to_screen.filter(regex='lab_data').columns
    strains_screen = to_screen.loc[gene_index, strains]
    passed_thresh = list(strains_screen.loc[strains_screen.median(axis=1) > thresh].median(axis=1).sort_values(ascending=False).index)
    new_gene_oder.extend(passed_thresh)
    gene_index = gene_index.drop(passed_thresh)
    for fam in families:
        for version in versions:
            strains = to_screen.filter(regex=fam).filter(regex=version).columns
            strains_screen = to_screen.loc[gene_index, strains]
            passed_thresh = list(strains_screen.loc[strains_screen.median(axis=1) > thresh].median(axis=1).sort_values(ascending=False).index)
            new_gene_oder.extend(passed_thresh)
            gene_index = gene_index.drop(passed_thresh)
    new_gene_oder.extend(list(gene_index))
    return new_gene_oder


def get_individual_gene_order(to_screen, thresh=2, add_others=True):

    
    gene_index = to_screen.index
    new_gene_oder = []
    
    for strain in to_screen.columns.values:
        strains_screen = to_screen.loc[gene_index, strain]
        passed_thresh = list(strains_screen.loc[strains_screen > thresh].sort_values(ascending=False).index)
        new_gene_oder.extend(passed_thresh)
        gene_index = gene_index.drop(passed_thresh)
    if add_others:
        new_gene_oder.extend(list(gene_index))
    return new_gene_oder

def load_prom_signals(name, start_path):
    return pd.read_pickle(os.path.join(start_path,(name+'_700bp_signals.gz')))

def get_df_motifs(df, motifs, strands_df, flanks=20):
    sig_dict = dict()
    for prom_motifs in motifs.index.values:
        for i,mot in enumerate(motifs.loc[prom_motifs].dropna()):
            strand = strands_df.loc[prom_motifs].iloc[i]
            sig_dict[prom_motifs+'_'+str(i)] = get_mot_signal(df, int(mot), prom_motifs, flanks, strand)
    return pd.DataFrame(sig_dict).transpose()

def get_mot_signal(signal_df, indices, prom, flanks, strand):
    sp_arr = np.empty(flanks)
    sp_arr[:] = np.nan
    if strand:
        sig = signal_df.loc[prom].dropna().values[indices-22:indices+28]
        sp_arr[flanks-len(sig):] =  sig 
    else:
        sig = signal_df.loc[prom].dropna().values[indices-28:indices+22]
        sp_arr[flanks-len(sig):] =  sig[::-1]
    return sp_arr

def norm_for_plot(df):
    return pd.DataFrame(stats.zscore(df.sum(axis=1)), index=df.sum(axis=1).index)

def load_corrs(tf, path='data/csvs_wts/'):
    '''
    Function to load correlation dataset of library behaviours
    has a default path
    in:

    tf -> str, name of the tf, case agnostic
    path -> path where csv files sit
    '''
    
    file_name ='{}_corrs_wt.csv'.format(tf.lower())
    file_path = os.path.join(path, file_name)
    corrs = pd.read_csv(file_path, index_col=0)
    return corrs