import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
sns.set_style('white')
import general as gn
import matplotlib
import statsmodels.api as sm

    
def get_data(tf, data_check, rna_nonlibs, thresh=3):
    '''Get name of tf and return subset of data needed for analysis'''
    binding = gn.return_family(data_check.filter(regex='^'+tf))
    rna = gn.return_family_rna(rna_nonlibs.filter(regex='^'+tf))
    full = binding.filter(regex='{}_lab_data'.format(tf))
    if tf == 'Gal4':
        full = binding.filter(regex='Full').filter(regex='erv')
    elif tf == 'Gcn4':
        full = binding.filter(regex='{}_lown_lab_data'.format(tf))    
    top50 = full.loc[(gn.get_zscored(full)>thresh).values].squeeze().sort_values(ascending=False).index.values[:100]
    fam = rna.filter(regex='tef|erv')
    fam.columns = ['_'.join(name.split('_')[:-1]) for name in fam.columns.values]
    fam = fam.transpose().reset_index().groupby('index').median().transpose().rename_axis(None, axis=1).loc[:, fam.columns.drop_duplicates()]
    return binding, rna, top50, fam, full

def return_aro_binding(names,chec, aro_targets):
    '''Return sum of aro80 reads on given strains from given data'''
    return chec.loc[aro_targets, names].sum()

def plotlibs_indgene(tf,genes, rna_libs,):
    '''Plot individual genes expression as a function of abundance of given TF library of a given gene'''
    cmap = sns.color_palette('flare', n_colors=6)
    fig, ax = plt.subplots(3,2, figsize=(4,7), sharey=False, sharex=False, constrained_layout=True)
    maxdiff = 0
    for k,gene in enumerate(genes):
        for i, typ in enumerate(['Full', 'DBDGal4AD', 'DBDGcn4AD']):
            curr_libs = rna_libs.filter(regex='^'+tf).filter(regex=typ).sort_values(by='facs', axis=1)
            curr_gene = curr_libs.loc[gene]
            maxy = curr_gene[-3:].mean()
            miny = curr_gene[:3].mean()
            maxdiff = np.max((np.float64(maxdiff),(maxy-miny)))
            ax[i][k].scatter(curr_libs.loc['facs'], curr_gene, c=cmap[np.abs(int(np.ceil(maxy-miny))-1)], edgecolors='k', linewidths=1, s=60, zorder=10, alpha=0.8)
            
            ax[i][k].set_title(typ)
        ax[i][k].set_xlabel('FACS')
        sns.despine(fig)
    fig.suptitle(tf +' ' +' '.join(genes))
    
def plotlibs(tf,genes,rna_libs,linewid=5, deg=2):
    '''Plot mean library expression on given targets of a given library'''
    cmap = matplotlib.cm.get_cmap('PuBuGn')
    pale = sns.color_palette('Set2')
    pale = sns.color_palette('inferno', n_colors=3)
    fig, ax = plt.subplots(1, figsize=(5,3), sharex=True, sharey=True, constrained_layout=True, dpi=160)
    maxdiff=0
    for i, typ in enumerate(['Full', 'DBDGal4AD', 'DBDGcn4AD']):
        curr_libs = rna_libs.filter(regex='^'+tf).filter(regex=typ).sort_values(by='facs', axis=1)
        y = curr_libs.loc[genes].median().values
        y = y - y[:5].mean()
        x = curr_libs.loc['facs'].values
        p4 = np.poly1d(np.polyfit(x, y, deg))
        ax.scatter(x, y, edgecolors=pale[i], linewidths=2, s=30, zorder=10, alpha=0.3, c='k', label=typ)
        ax.scatter(x,y, edgecolors='k', linewidths=.5, s=30, zorder=15, facecolors='none')
        ax.plot(x, p4(x),  zorder=20, linewidth=linewid, alpha=1, c=pale[i])

        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 0.32))

    fig.suptitle('Median expression of top {} {}-bound gene-promoters'.format(len(genes), tf))
    sns.despine(fig)
    return fig

def corr_calc(data, genes):
    '''Calculate spearman rank corr on a given geens (Library only)'''
    corrs = []
    for gene in genes:
        corrs.append(scipy.stats.spearmanr(data.loc['facs'], data.loc[gene])[0])
    return corrs

def facs_pic(tf, rna_libs, arosum, tops_exp_bind, mean_libs_validated, sc_colors, size, alpha,edges):
    '''Plot median expression as a function of FACS and Absolut binding'''

    targ_direct = 'upregulated'

    if tf == 'Mig3':
        targ_direct = 'downregulated'
    fig, ax = plt.subplots(1,2, figsize=(7,3), sharey=True)
    for i, s_type in enumerate(['Full', 'DBDGal4AD', 'DBDGcn4AD']):
        curr_libs = rna_libs.filter(regex=tf+s_type)
        
        tef = curr_libs.filter(regex='TEF')
        curr_libs = curr_libs.drop(tef.columns.values, axis=1)
        curr_check = mean_libs_validated.filter(regex=tf+s_type).sort_values(by='facs', axis=1)
        x1 = curr_check.div(arosum.loc[curr_check.columns]).loc[tops_exp_bind[tf][targ_direct]].sum()
        x2 = curr_check.loc['facs']

        curr_check_notef = curr_check.drop(curr_check.filter(regex='TEF'), axis=1)
        curr_check_notef.columns = [name.replace('__', '_') for name in curr_check_notef.columns.values]
        names = [name.split('_')[2] for name in curr_check_notef.columns.values]

        y_ax = []
        for name in names:
            
            s_ind = np.where(rna_libs.filter(regex=tf+s_type).columns.str.contains(name))[0][0]
            curr_spread = rna_libs.filter(regex=tf+s_type).iloc[:, s_ind-1:s_ind+2].loc[tops_exp_bind[tf][targ_direct]].median().mean()
            if s_ind == 0:
                curr_spread = rna_libs.filter(regex=tf+s_type).iloc[:, s_ind:s_ind+3].loc[tops_exp_bind[tf][targ_direct]].median().mean()


            y_ax.append(curr_spread)
        y_ax.append(tef.loc[tops_exp_bind[tf][targ_direct]].median().mean())
        
        
        ax[0].scatter(x1, y_ax, c=sc_colors[i], s=size, edgecolors=edges, alpha=alpha, linewidths=1.5)
        ax[1].scatter(x2, y_ax, c=sc_colors[i], s=size, edgecolors=edges, alpha=alpha, linewidths=1.5)
        ax[0].set_xlabel('Binding strenth')
        ax[1].set_xlabel('FACS')
        ax[0].set_ylabel('Median exp of {} genes'.format(len(tops_exp_bind[tf][targ_direct])))  
        fig.suptitle(tf)
    
def panel_scatters(tf, data, top, arosum, size, alpha, edges, sc_colors):
    '''Plot panels of different things for factor'''
    if tf == 'Mig3':
        top = top['downregulated']
    else:
        top = top['upregulated']

    curr_mots = pd.read_csv('old_notebooks/libs_validation_check/mots_zscores/{}_mean_mots_signal.csv'.format(tf), index_col=0)
    curr_mots.columns = [name.replace('DBD_', 'DBD') for name in curr_mots.columns.values]
    
    fig, ax = plt.subplots(2,2, figsize=(9,8), constrained_layout=True, sharex=True)
    ax= ax.flatten()
    
    for i, typ in enumerate(['Full', 'Gal4AD', 'Gcn4AD']):
        curr_libs = data.filter(regex='^'+tf).filter(regex=typ).sort_values(by='facs', axis=1)
        norm_curr_libs = curr_libs.div(arosum.loc[curr_libs.columns], axis=1) 

        if tf == 'Msn2':
            curr_libs = curr_libs.drop(curr_libs.filter(regex='tef').columns, axis=1)
            norm_curr_libs = curr_libs.div(arosum.loc[curr_libs.columns], axis=1) 
        ax[0].scatter( curr_libs.loc['facs'], norm_curr_libs.loc[top].sum(), c=sc_colors[i],
                      alpha=alpha, edgecolors=edges, s=size, )
        ax[1].scatter( curr_libs.loc['facs'], norm_curr_libs.sum(), c=sc_colors[i], alpha=alpha, edgecolors=edges, s=size)
        ax[2].scatter( curr_libs.loc['facs'], curr_mots.loc[:, norm_curr_libs.columns].sum(), c=sc_colors[i], alpha=alpha, edgecolors=edges, s=size)
        ax[3].scatter( curr_libs.loc['facs'], (curr_mots.loc[:, norm_curr_libs.columns]>curr_mots.loc[:, norm_curr_libs.columns].max()*0.05).sum()
                     , c=sc_colors[i], alpha=alpha, edgecolors=edges, s=size, label=tf+typ)
        
        ax[0].set_title('Sum signal on\n target promoters\nnorm by aro')
        ax[1].set_title('Sum signal on\n all promoters\nnorm by aro')
        ax[2].set_title('Sum signal on\n motifs')
        ax[3].set_title('N motifs in range\nof 5% from max')
        
        for axi in ax:
            axi.set_xlabel('FACS')


    ax[3].legend(loc='upper right', bbox_to_anchor=(1.6, 0.35))
    sns.despine()
    fig.suptitle(tf)

def heatmaps_figure1(libs_list, curr_corrs, tops, wt):
    '''

    '''
    inters = set(tops).intersection(set(curr_corrs.loc[sm.stats.multipletests(curr_corrs.pval.values, method='fdr_bh')[0]].index))
    fig, ax = plt.subplots(1,5, figsize=(8,2), width_ratios=[0.2,0.2,1,1,1.2], sharey=True, dpi=300, constrained_layout=True)
    zscores = ax[0]
    pvals = ax[1]
    ax = ax[2:]
    cbar = [False, False, True]
    
    og = libs_list[0].loc[curr_corrs.loc[tops].sort_values(by='pval').index]
    remove_low = og.iloc[:, :3].min(axis=1)
    remove_top = og.iloc[:, -3:].max(axis=1)
    by_fc = (remove_top - remove_low).sort_values(ascending=False)
    passed_sorted = by_fc.loc[inters].sort_values(ascending=False).index
    notpassed_sorted = by_fc.drop(inters).sort_values(ascending=False).index
    
    sns.heatmap(og.sub(remove_low, axis=0).loc[np.concatenate([passed_sorted, notpassed_sorted])], cmap='RdGy_r',
                yticklabels=False, ax=ax[0], xticklabels=False, cbar=cbar[0], vmax=4, vmin=-2, center=0)
    sns.heatmap(pd.DataFrame(-np.log10(curr_corrs.loc[np.concatenate([passed_sorted, notpassed_sorted])].pval)), cmap='afmhot', ax=pvals, xticklabels=False)
    sns.heatmap(gn.get_zscored(wt).loc[np.concatenate([passed_sorted, notpassed_sorted])], ax=zscores, vmax=10, cmap='cool', xticklabels=False, yticklabels=False)
    for i, curr_libs in enumerate(libs_list[1:]):
        og = curr_libs.loc[curr_corrs.loc[tops].sort_values(by='pval').index]
        remove_low = og.iloc[:, :3].min(axis=1)
        sns.heatmap(og.sub(remove_low, axis=0).loc[np.concatenate([passed_sorted, notpassed_sorted])], cmap='RdGy_r',
                yticklabels=False, ax=ax[i+1], xticklabels=False, cbar=cbar[i+1], vmax=4, vmin=-2, center=0 )

    for axi in ax:
        axi.set_xlabel(None)
        axi.set_ylabel(None)
        axi.axhline(len(inters), c='lime', linewidth=3)
    ax[0].set_title('Full')
    ax[1].set_title('DBD-Gal4AD')
    ax[2].set_title('DBD-Gcn4AD')
    pvals.set_ylabel(None)
    pvals.set_xlabel('-log10\npval')
    pvals.axhline(len(inters), c='lime',linewidth=3)
    zscores.set_ylabel(None)
    zscores.set_xlabel('zscore\nwt\nbinding')
    zscores.axhline(len(inters), c='lime',linewidth=3)
    return fig, passed_sorted

# def get_bound_wt(tf, wts, thresh):
#     wt_proms = wts[tf].loc[(gn.get_zscored(wts[tf]) > thresh).values]
#     wt_proms = wt_proms.sort_values(by=wt_proms.columns.values[0], ascending=False)
#     return wt_proms

# def get_bound_tef(tf, data, s_type, thresh):
#     filtered_data = data.filter(regex='^{}{}'.format(tf, s_type))
#     tef_proms = filtered_data.loc[(gn.get_zscored(filtered_data) >= thresh).values]
#     tef_proms = tef_proms.sort_values(by=tef_proms.columns.values[0], ascending=False).index.values
#     return tef_proms, filtered_data

def return_libs_tf(tf, rna_libs):
    tfdat = rna_libs.filter(regex='^'+tf)
    full = tfdat.filter(regex='Full').sort_values(by='facs', axis=1)
    gal = tfdat.filter(regex='Gal4AD').sort_values(by='facs', axis=1)
    gcn = tfdat.filter(regex='Gcn4AD').sort_values(by='facs', axis=1)
    
    return full, gal, gcn

# def plot_scatter_polynom(ax, data, genes):
#     cmap = sns.color_palette('Greens', as_cmap=True)
#     x = data.loc['facs'].values
#     y = data.loc[genes].median()
#     c = np.abs(y[-3:].mean() - y[:3].mean())
#     p4 = np.poly1d(np.polyfit(x,y, 2))
#     ax.scatter(x, y, c=cmap(c), edgecolors='k', s=60)# type: ignore    
#     ax.plot(x, p4(x), c='r', linewidth=2)

# def get_facs_mean_between2(tf, s_type, data):
#     facs_dat = data.filter(regex='^'+tf).filter(regex=s_type).sort_values(by='facs', axis=1).drop('facs')
#     high = facs_dat.iloc[:, 2:].mean(axis=1)
#     low = facs_dat.iloc[:, :2].mean(axis=1)
#     df_return = pd.concat([high, low], axis=1)
#     df_return.columns = ['High facs {}'.format(tf+s_type), 'Low facs {}'.format(tf+s_type)]
#     return df_return

# def libs_median_patterns_fig3(opn, geneset, full, gal, gcn, cmfrm, trial_data, set_names, ax, fig,i, tf):
#     opn_set = opn.loc[geneset].query('opn_score>0').index
#     dpn_set = opn.loc[geneset].query('opn_score<=0').index
    
#     plot_scatter_polynom(ax[1], full, opn_set)
#     plot_scatter_polynom(ax[2], gal, opn_set)
#     plot_scatter_polynom(ax[3], gcn, opn_set)
    
#     # fig.suptitle()
#     ax[4].get_xaxis().set_visible(False)
#     ax[4].get_yaxis().set_visible(False)
#     ax[4].text(.3,.5,cmfrm[i], size=20)
#     ax[4].text(.3,.1,'OPN N={} ,DPN N={}'.format(len(opn_set), len(dpn_set)), size=20)


#     sns.despine(ax=ax[4], top=True, bottom=True, left=True)
    
#     plot_scatter_polynom(ax[5], full, dpn_set)
#     plot_scatter_polynom(ax[6], gal, dpn_set)
#     plot_scatter_polynom(ax[7], gcn, dpn_set)
    
#     ax[1].set_title('{}_Full'.format(tf))
#     ax[5].set_title('{}_Full'.format(tf))
#     ax[2].set_title('{}_Gal4AD'.format(tf))
#     ax[6].set_title('{}_Gal4AD'.format(tf))
#     ax[3].set_title('{}_Gcn4AD'.format(tf))
#     ax[7].set_title('{}_Gcn4AD'.format(tf))
#     fig.subplots_adjust(wspace=2)
    
#     boxplot_dat = trial_data.loc[(trial_data.genetype == set_names[i]).values].drop('genetype', axis=1)
#     sns.boxplot(boxplot_dat.loc[opn_set], palette='terrain', ax=ax[0], showfliers=False)
#     sns.boxplot(boxplot_dat.loc[dpn_set], palette='terrain', ax=ax[8], showfliers=False)
#     ax[0].set_xticklabels('')
#     ax[0].set_ylabel('zscore')
#     ax[8].set_xticklabels('')
#     ax[8].set_ylabel('zscore')
#     return opn_set, dpn_set

def get_nonlib_data(tf: str, rna: pd.DataFrame, check: pd.DataFrame, pattern: str):
    '''
    Get name of tf and return it's bidning and average expression data by given pattern
    '''
    bind = check.filter(regex='^'+tf).filter(regex=pattern)
    exp = rna.filter(regex='^'+tf).filter(regex=pattern)

    exp.columns = ['_'.join(name.split('_')[:-1]) for name in exp.columns.values]
    exp = exp.transpose().reset_index().groupby('index').mean().transpose().rename_axis(None, axis=1)
    return bind, exp

def get_targets_df(data, thresh, t_name, opn, sort_by='opn'):

    threshed = gn.get_zscored(data)>=thresh
    targets = pd.DataFrame(np.where(threshed.values)[0],
    np.where(threshed.values)[1]).reset_index().groupby('index').agg({0:lambda x: list(x)})
    targets.index = data.columns

    if sort_by == 'opn':
        numeric_to_name = lambda x: opn.iloc[targets.iloc[x].values[0]].loc[:, 'opn_score'].sort_values(ascending=False).index.values
    else:
        numeric_to_name = lambda x: data.iloc[targets.iloc[x].values[0]].sort_values(ascending=False, by=data.columns.values[x]).index.values

    targets.loc[:, t_name] = list(map(numeric_to_name, range(targets.index.shape[0])))
    targets = targets.drop(0, axis=1)
    return targets

# def return_long_zscores(tf, targets_df, prom_type, binding, rna, lab_data):
#     '''
#     Given targets of binding of a tf, return zscores of binding signal and foldchange of
#      rna on theese targets for all variants of the factor

#      tf: str, name of TF
#      targets_df: pd.DataFrame dataframe of the targets divided by opn/dpn and coming from wt, teffull, tefdbd
#      prom_type: str, index for targets_df to take the targets
#      binding: pd.DataFrame bnidnig of your factors in zscores
#      rna: pd.DataFrame rna of your strains already in foldchange from control
#      lab_data: pd.DataFrame, binding data of lab strains for binding comparison  
#     '''
#     bind_list = []
#     bind_rna = []
#     for stype in ['Full', 'Gal4AD', 'Gcn4AD']:
        
#         if type(targets_df.loc[tf, prom_type]) == float:
#             continue
        
#         b = lambda x :binding.filter(regex='^'+x).filter(regex=stype).loc[targets_df.loc[x, prom_type]]
#         r = lambda x :rna.filter(regex='^'+x).filter(regex=stype).loc[targets_df.loc[x, prom_type]]
#         datb = b(tf).transpose().values
#         datr = r(tf).transpose().values
#         if datb.shape[0] > 0:
#             bind_list.extend(list(zip([tf]*datb[0].shape[0],[stype]*datb[0].shape[0], datb[0], targets_df.loc[tf, prom_type])))
#         if datr.shape[0] > 0:
#             bind_rna.extend(list(zip([tf]*datr[0].shape[0],[stype]*datr[0].shape[0], datr[0], targets_df.loc[tf, prom_type])))
    
#     if type(targets_df.loc[tf, prom_type]) != float:
                
#         multplier = targets_df.loc[tf, prom_type].shape[0]
#         labscores = gn.get_zscored(lab_data).loc[targets_df.loc[tf, prom_type], tf].values
#         bind_list.extend(list(zip([tf]*multplier,['lab']*multplier, labscores, targets_df.loc[tf, prom_type])))

#     return bind_list, bind_rna

# def boxplots_zscores_fc(binding, rna, all_tfs, prom_type):
#     df_binding = pd.DataFrame(binding)
#     df_rna= pd.DataFrame(rna)
#     df_rna.columns, df_binding.columns = ['name', 'type', 'fc', 'gene'], ['name', 'type', 'zscore', 'gene']
#     df_rna.index = df_rna.name
#     df_binding.index = df_binding.name


#     fig, ax = plt.subplots(2,1, figsize=(20,7), dpi=300, constrained_layout=True)
#     ax= ax.flatten()

#     sns.boxplot(data=df_rna, hue='type', x='name', y='fc', ax=ax[0], showfliers=False, palette='cubehelix', order=all_tfs,  boxprops={"facecolor": 'none'},zorder=20)
#     sns.stripplot(data=df_rna, hue='type', x='name', y='fc', ax=ax[0], palette='cubehelix',
#     order=all_tfs, dodge=True, size=6, alpha=0.7, zorder=1, edgecolors='k', linewidth=1, jitter=True, legend=False)
#     ax[0].set_xlabel('FC of expression')
#     ax[0].legend(loc='upper right', bbox_to_anchor=(1.1,1.35))
#     ax[0].axhline(0, c='r', linestyle=':')

#     sns.boxplot(data=df_binding, hue='type', x='name', y='zscore', ax=ax[1], showfliers=False, palette='cubehelix', order=all_tfs, hue_order=['lab', 'Full', 'Gal4AD', 'Gcn4AD'], boxprops={"facecolor": 'none'},zorder=20)
#     sns.stripplot(data=df_binding, hue='type', x='name', y='zscore', ax=ax[1], palette='cubehelix',
#     order=all_tfs, dodge=True, size=6, alpha=0.7, zorder=1, edgecolors='k', linewidth=1, jitter=True, hue_order=['lab', 'Full', 'Gal4AD', 'Gcn4AD'], legend=False)
#     ax[1].axhline(3, c='r', linestyle=':')
#     ax[1].set_xlabel('Zscores of target bindings')
#     ax[1].legend(loc='upper right', bbox_to_anchor=(1.1,1.35))
#     fig.suptitle(prom_type)
#     return fig

