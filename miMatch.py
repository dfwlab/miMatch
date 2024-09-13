#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20221120

@author: dingfengwu

packages:
install.packages('MatchIt') # R package
pip install rpy2==3.2.2
pip install tzlocal

references:
http://www.360doc.com/content/17/1129/16/95144_708343053.shtml
https://zhuanlan.zhihu.com/p/145170602
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import random
import time
import copy
import shutil
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as multi
import statsmodels.api as sm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
import matplotlib.lines as mlines
import configparser
from sklearn.decomposition import PCA
import cloudpickle as pickle
from multiprocessing import Pool
### rpy2
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.pandas2ri import py2rpy, rpy2py
pandas2ri.activate()

class miMatch():
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.realpath('__file__'))
        self.config = self.load_config()
        self.output_dir_checked = 0
        # refresh global R environment
        r('rm(list = ls(all.names=TRUE))')
        r("as.data.frame(gc())")
        # import R packages
        importr('base')
        importr('MatchIt')
        importr('cobalt')
        importr('grDevices')
        
    def load_data(self, data):
        r.assign('data', data)
    
    def set_config(self, part, key, value):
        self.config[part][key] = value
    
    def get_config(self, part, key):
        return self.config[part][key]
        
    def load_config(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.base_dir,'config.ini'))
        return config
    
    def check_output_dir(self):
        if self.output_dir_checked == 0:
            self.outpath = os.path.abspath(os.path.join(self.base_dir, self.config['output']['output_dir']))
            if os.path.exists(self.outpath):
                shutil.rmtree(self.outpath)
            os.makedirs(self.outpath)
        self.output_dir_checked += 1
            
    
    def create_test_data(self):
        N = int(self.config['data']['random_sample_size_pre_group'])
        features = eval(self.config['data']['random_features'])
        random_features_us = eval(self.config['data']['random_features_u'])
        random_features_sigmas = eval(self.config['data']['random_features_sigma'])
        data = pd.DataFrame(np.zeros((N*2, len(features))), columns=features, index=['S_'+str(i) for i in range(N*2)])
        for fi in range(len(features)):
            data.iloc[:N, fi] = np.random.normal(loc=random_features_us[fi][0], scale=random_features_sigmas[fi][0], size=N)
            data.iloc[N:, fi] = np.random.normal(loc=random_features_us[fi][1], scale=random_features_sigmas[fi][1], size=N)
        data['Group'] = 0
        data.iloc[N:, -1] = 1
        return data
    
    def output_tables(self, target):
        ### Output
        with open(os.path.abspath(os.path.join(self.outpath, self.config['output']['summary_report_filename'])), 'w') as f:f.write(str(r('summary(psm)')))
        sum_all = r('as.data.frame(summary(psm)$sum.all)')
        sum_all.to_csv(os.path.abspath(os.path.join(self.outpath, self.config['output']['balance_summary_all_filename'])), sep='\t')
        sample_size = r('as.data.frame(summary(psm)$nn)')
        sample_size.to_csv(os.path.abspath(os.path.join(self.outpath, self.config['output']['sample_size_filename'])), sep='\t')
        sum_matched = r('as.data.frame(summary(psm)$sum.matched)')
        sum_matched.to_csv(os.path.abspath(os.path.join(self.outpath, self.config['output']['balance_summary_matched_filename'])), sep='\t')
        r('as.data.frame(summary(psm)$reduction)').to_csv(os.path.abspath(os.path.join(self.outpath, self.config['output']['percent_balance_summary_filename'])), sep='\t')
        match_drop_unmatched = r("match.data(psm, drop.unmatched=TRUE)")
        match = r("match.data(psm, drop.unmatched=FALSE)")
        pairs = r("as.data.frame(psm$match.matrix)")
        match.to_csv(os.path.abspath(os.path.join(self.outpath, self.config['output']['matched_table_filename'])), sep='\t')
        pairs.to_csv(os.path.abspath(os.path.join(self.outpath, self.config['output']['matched_pairs_filename'])), sep='\t')
        # balance report
        with open(os.path.abspath(os.path.join(self.outpath, self.config['output']['balance_report_by_mean_difference_filename'])), 'w') as f:f.write(str(r('bal.tab(psm, m.threshold=%s)'%self.config['output']['balance_report_mean_difference_threshold'])))
        with open(os.path.abspath(os.path.join(self.outpath, self.config['output']['balance_report_by_variance_ratio_filename'])), 'w') as f:f.write(str(r('bal.tab(psm, v.threshold=%s)'%self.config['output']['balance_report_variance_ratio_threshold'])))
        balance_stats = self.covariate_balance_test(match, pairs, target, list(sum_all.index))
        return sample_size, match_drop_unmatched, match, pairs, sum_all, sum_matched, balance_stats
    
    def get_matched_pair_set(self, pairs):
        pair_set = []
        for i in pairs.index:
            for j in pairs.loc[i, :]:
                if (str(i) == 'NA_character_' or str(i).upper() == 'NAN') or (str(j) == 'NA_character_' or str(j).upper() == 'NAN'):
                    continue
                pair_set.append([i, j])
        pair_set = pd.DataFrame(pair_set, columns=['Case', 'Control'])
        pair_set = pair_set.dropna(how='any')
        return pair_set
    
    def percentile(self, x, y):
        if len(x)==len(y):
            x_pct = sorted(x)
            y_pct = sorted(y)
        else:
            x_pct = np.asarray([stats.percentileofscore(y, i) for i in y])
            y_pct = np.asarray([stats.percentileofscore(x, i) for i in y])
        return x_pct, y_pct
    
    def qqplot(self, x_pct, y_pct, range_min, range_max):
        plt.scatter(x=x_pct, y=y_pct, color='blue', s=20, alpha=0.6)
        plt.xlim((range_min, range_max))
        plt.ylim((range_min, range_max))
        plt.plot([-1000, 1000], [-1000, 1000], linewidth=1, color='k')
        plt.plot([range_min, range_max], [range_min+(range_max-range_min)*0.1, range_max+(range_max-range_min)*0.1], linewidth=1, color='k', ls='--')
        plt.plot([range_min, range_max], [range_min-(range_max-range_min)*0.1, range_max-(range_max-range_min)*0.1], linewidth=1, color='k', ls='--')
        plt.ticklabel_format(style ='scientific', axis ='y', scilimits=(0, 0), useMathText=True)

    def eQQ_plot(self, match, pairs, target, features):
        #r("png(file='%s', width=%s, height=%s, units='%s', res=%s);plot(psm);dev.off()"%(os.path.abspath(os.path.join(self.outpath, self.config['plot']['qq_plot_name'])), self.config['plot']['qq_plot_width'], self.config['plot']['qq_plot_height'], self.config['plot']['qq_plot_unit'], self.config['plot']['qq_plot_dpi']))
        # https://www.itl.nist.gov/div898/handbook/eda/section3/eda33o.htm
        # https://www.cnblogs.com/arkenstone/p/5763069.html
        pair_set = self.get_matched_pair_set(pairs)
        ### Plot
        fig = plt.figure(0, (int(self.config['plot']['qq_plot_width']), int(self.config['plot']['qq_plot_height'])), dpi=int(self.config['plot']['qq_plot_dpi']))
        gs = gridspec.GridSpec(len(features), 2)
        gs.update(left=0.08, right=0.92, top=0.92, bottom=0.08, wspace=0.05, hspace=0.1)
        for f in features:
            x_pct, y_pct = self.percentile(match.loc[match[target]==0, f], match.loc[match[target]==1, f])
            x_matched_pct, y_matched_pct = self.percentile(match.loc[pair_set['Control'], f], match.loc[pair_set['Case'], f])
            range_min = min(min(x_pct), min(y_pct), min(x_matched_pct), min(y_matched_pct))
            range_max = max(max(x_pct), max(y_pct), max(x_matched_pct), max(y_matched_pct))
            range_min = range_min-(range_max-range_min)*0.05
            range_max = range_max+(range_max-range_min)*0.05
            ### left
            ax = plt.subplot(gs[features.index(f), 0])
            self.qqplot(x_pct, y_pct, range_min, range_max)
            plt.xticks([])
            if features.index(f) == 0:
                plt.title('All', fontsize='large')
            plt.ylabel(f, fontsize='large')
            ### right
            ax = plt.subplot(gs[features.index(f), 1])
            self.qqplot(x_matched_pct, y_matched_pct, range_min, range_max)
            plt.xticks([]);plt.yticks([])
            if features.index(f) == 0:
                plt.title('Matched', fontsize='large')
        fig.text(x=0.43, y=0.05, s='Control Units', weight='semibold', fontsize='x-large')
        fig.text(x=0.95, y=0.47, s='Case Units', rotation=90, weight='semibold', fontsize='x-large')
        fig.suptitle('eQQ plot', weight='semibold', fontsize='xx-large')
        plt.savefig(os.path.abspath(os.path.join(self.outpath, self.config['plot']['qq_plot_name'])))
        plt.close()
    
    def histplot(self, ax, x, title):
        axes = sns.histplot(data=x, stat='density', binrange=(0, 1), binwidth=float(self.config['plot']['hist_plot_binwidth']), 
                            color = self.config['plot']['hist_plot_color'])
        plt.ylabel('Propotion', fontsize='x-large')
        plt.xlabel('Propensity score', fontsize='x-large')
        plt.title(title, weight='semibold', fontsize='x-large')
        plt.yticks(axes.get_yticks(), axes.get_yticks()/10.)
        plt.xlim(0, 1)
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)
    
    def Hist_plot(self, match, pairs, target):
        #r("png(file='%s', width=%s, height=%s, units='%s', res=%s);plot(psm, type='hist');dev.off()"%(os.path.abspath(os.path.join(self.outpath, self.config['plot']['hist_plot_name'])).replace('.png', '_2.png'), '300', '200', 'mm', '300'))
        pair_set = self.get_matched_pair_set(pairs)
        ### Plot
        fig = plt.figure(0, (int(self.config['plot']['hist_plot_width']), int(self.config['plot']['hist_plot_height'])), dpi=int(self.config['plot']['hist_plot_dpi']))
        gs = gridspec.GridSpec(2, 2)
        gs.update(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.25)
        ax = plt.subplot(gs[0, 0])
        self.histplot(ax, match.loc[match[target]==0, 'distance'], 'Raw Control')
        ax = plt.subplot(gs[0, 1])
        self.histplot(ax, match.loc[pair_set['Control'], 'distance'], 'Matched Control')
        ax = plt.subplot(gs[1, 0])
        self.histplot(ax, match.loc[match[target]==1, 'distance'], 'Raw Case')
        ax = plt.subplot(gs[1, 1])
        self.histplot(ax, match.loc[pair_set['Case'], 'distance'], 'Matched Case')
        plt.savefig(os.path.abspath(os.path.join(self.outpath, self.config['plot']['hist_plot_name'])))
        plt.close()
    
    def Jitter_plot(self, match, pairs, target):
        #r("png(file='%s', width=%s, height=%s, units='%s', res=%s);plot(psm, type='jitter', interactive=FALSE);dev.off()"%(os.path.abspath(os.path.join(self.outpath, self.config['plot']['jitter_plot_name'])), self.config['plot']['jitter_plot_width'], self.config['plot']['jitter_plot_height'], self.config['plot']['jitter_plot_unit'], self.config['plot']['jitter_plot_dpi']))
        pair_set = self.get_matched_pair_set(pairs)
        ### Plot
        fig = plt.figure(0, (int(self.config['plot']['jitter_plot_width']), int(self.config['plot']['jitter_plot_height'])), dpi=int(self.config['plot']['jitter_plot_dpi']))
        x_matched = pd.DataFrame({'ID':pair_set['Control'], 'distance':match.loc[pair_set['Control'], 'distance'].values})
        size = x_matched.groupby('ID').size()
        x_matched = x_matched.drop_duplicates(subset='ID')
        x = pd.DataFrame({'Propensity score':match.loc[match[target]==0, 'distance'], 'y':0, 'Size':1})
        x = x.loc[[i for i in x.index if i not in list(x_matched['ID'])], :]
        x_matched = pd.DataFrame({'Propensity score':x_matched['distance'], 'y':1, 'Size':size.loc[x_matched['ID']].values})
        y_matched = pd.DataFrame({'ID':pair_set['Case'], 'distance':match.loc[pair_set['Case'], 'distance'].values})
        size = y_matched.groupby('ID').size()
        y_matched = y_matched.drop_duplicates(subset='ID')
        y = pd.DataFrame({'Propensity score':match.loc[match[target]==1, 'distance'], 'y':3, 'Size':1})
        y = y.loc[[i for i in y.index if i not in list(y_matched['ID'])], :]
        y_matched = pd.DataFrame({'Propensity score':y_matched['distance'], 'y':2, 'Size':size.loc[y_matched['ID']].values})
        plt.scatter(x['Propensity score'], x['y']+np.random.normal(0, 0.03, x.shape[0]), 
                    alpha=0.6, s=x['Size']*10, color=self.config['plot']['jitter_plot_color'])
        plt.annotate('Unmatched Control Units', xy=(0.36, 0.5), fontsize='medium')
        plt.scatter(y['Propensity score'], y['y']+np.random.normal(0, 0.03, y.shape[0]), 
                    alpha=0.6, s=y['Size']*10, color=self.config['plot']['jitter_plot_color'])
        plt.annotate('Unmatched Case Units', xy=(0.365, 3.5), fontsize='medium')
        plt.scatter(x_matched['Propensity score'], x_matched['y']+np.random.normal(0, 0.03, x_matched.shape[0]), 
                    alpha=0.6, s=x_matched['Size']*10, color=self.config['plot']['jitter_plot_color'])
        plt.annotate('Matched Control Units', xy=(0.37, 1.5), fontsize='medium')
        plt.scatter(y_matched['Propensity score'], y_matched['y']+np.random.normal(0, 0.03, y_matched.shape[0]), 
                    alpha=0.6, s=y_matched['Size']*10, color=self.config['plot']['jitter_plot_color'])
        plt.annotate('Matched Case Units', xy=(0.375, 2.5), fontsize='medium')
        plt.xlim((-0.05, 1.0))
        plt.ylim(-0.5, 3.8)
        plt.xticks(np.round(np.arange(0, 1, 0.1), 1), np.round(np.arange(0, 1, 0.1), 1))
        plt.yticks([])
        plt.xlabel('Propensity score', fontsize='large')
        plt.title('Distribution of Propensity score', weight='semibold', fontsize='x-large')
        plt.savefig(os.path.abspath(os.path.join(self.outpath, self.config['plot']['jitter_plot_name'])))
        plt.close()
    
    def covariate_balance_test(self, match, pairs, target, features):
        #r("png(file='%s', width=%s, height=%s, units='%s', res=%s);print(love.plot(bal.tab(psm, m.threshold=%s), stat ='mean.diffs', abs = F));dev.off()"%(os.path.abspath(os.path.join(self.outpath, self.config['plot']['summary_plot_name'])), self.config['plot']['summary_plot_width'], self.config['plot']['summary_plot_height'], self.config['plot']['summary_plot_unit'], self.config['plot']['summary_plot_dpi'], self.config['plot']['m_threshold']))
        pair_set = self.get_matched_pair_set(pairs)
        result = []
        k = 0
        for f in features:
            ### unmatched
            control = match.loc[match[target]==0, f]
            case = match.loc[match[target]==1, f]
            t, p = stats.ttest_ind(control, case)
            result.append(['unmatched', 't-test', f, k, p])
            w, p = stats.mannwhitneyu(control, case)
            result.append(['unmatched', 'wilcoxon test', f, k, p])
            ### matched
            #print(pair_set['Control'])
            control = match.loc[pair_set['Control'], f]
            case = match.loc[pair_set['Case'], f]
            t, p = stats.ttest_ind(control, case)
            result.append(['matched', 't-test', f, k, p])
            w, p = stats.mannwhitneyu(control, case)
            result.append(['matched', 'wilcoxon test', f, k, p])
            k += 1
        result = pd.DataFrame(result, columns=['isMatch', 'test', 'feature', 'feature_index', 'pvalue'])
        return result

    def balance_test_plot(self, test, target, features):
        _ = plt.figure(0, (int(self.config['plot']['statstic_balance_plot_width']), int(self.config['plot']['statstic_balance_plot_height'])), dpi=int(self.config['plot']['statstic_balance_plot_dpi']))
        ax = plt.subplot()
        temp = test.loc[(test['isMatch']=='unmatched')&((test['test']=='t-test')), :]
        plt.scatter(temp['pvalue'], temp['feature_index'], s=100, alpha=0.5, color='blue', marker='o', label='t-test p-values before matching')
        temp = test.loc[(test['isMatch']=='unmatched')&((test['test']=='wilcoxon test')), :]
        plt.scatter(temp['pvalue'], temp['feature_index'], s=100, alpha=0.5, color='blue', marker='^', label='wilcoxon test p-values before matching')
        temp = test.loc[(test['isMatch']=='matched')&((test['test']=='t-test')), :]
        plt.scatter(temp['pvalue'], temp['feature_index'], s=100, alpha=0.5, color='red', marker='o', label='t-test p-values after matching')
        temp = test.loc[(test['isMatch']=='matched')&((test['test']=='wilcoxon test')), :]
        plt.scatter(temp['pvalue'], temp['feature_index'], s=100, alpha=0.5, color='red', marker='^', label='wilcoxon test p-values after matching')
        N = len(features)
        plt.vlines(x=0.05, ymin=-0.5, ymax=N-0.5, ls='--', alpha=0.5)
        plt.ylim(-0.5, N-0.5)
        plt.yticks(range(N), features)
        plt.xlim(-0.03, 1.05)
        plt.xticks([0, 0.05, 0.1, 0.5, 1], [0, 0.05, 0.1, 0.5, 1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(bbox_to_anchor=(0.54, -0.05), loc='upper center', borderaxespad=0, ncol=2, edgecolor='white')
        plt.savefig(os.path.abspath(os.path.join(self.outpath, self.config['plot']['statstic_balance_plot_name'])))
        plt.close()
    
    def feature_distribution(self, features):
        for f in features:
            r("png(file='%s%s.png', width=%s, height=%s, units='%s', res=%s);print(bal.plot(psm, var.name='%s', which = 'both'));dev.off()"%(os.path.abspath(os.path.join(self.outpath, self.config['plot']['feature_distribution_prefix'])), f, self.config['plot']['feature_distribution_width'], self.config['plot']['feature_distribution_height'], self.config['plot']['feature_distribution_unit'], self.config['plot']['feature_distribution_dpi'], f))
    
    def Covariate_Balance_plot(self, sum_all, sum_matched, balance_stats, target, features):
        ### mean_difference_balance
        fig = plt.figure(0, (int(self.config['plot']['mean_difference_balance_plot_width']), int(self.config['plot']['mean_difference_balance_plot_height'])), dpi=int(self.config['plot']['mean_difference_balance_plot_dpi']))
        plt.subplot(position=[0.1, 0.1, 0.75, 0.8])
        plt.scatter(sum_all['Std. Mean Diff.'], range(sum_all.shape[0])[::-1], alpha=0.7, label='Unmatched')
        plt.scatter(sum_matched['Std. Mean Diff.'], range(sum_matched.shape[0])[::-1], alpha=0.7, label='Matched')
        plt.vlines(x=0, ymin=-1000, ymax=1000, lw=1)
        plt.vlines(x=float(self.config['plot']['mean_difference_balance_threshold']), 
                   ymin=-1000, ymax=1000, lw=1, ls='--')
        plt.vlines(x=-float(self.config['plot']['mean_difference_balance_threshold']), 
                   ymin=-1000, ymax=1000, lw=1, ls='--')
        plt.hlines(y=sum_all.shape[0]-1.5, xmin=-1000, xmax=1000, lw=1)
        plt.ylim(-0.5, sum_all.shape[0]-0.5)
        Min, Max = [min(sum_all['Std. Mean Diff.'].min(), sum_matched['Std. Mean Diff.'].min()), 
                    max(sum_all['Std. Mean Diff.'].max(), sum_matched['Std. Mean Diff.'].max())]
        plt.xlim(Min-(Max-Min)*0.1, Max+(Max-Min)*0.1)
        plt.legend(bbox_to_anchor=(1.18, 0.55), edgecolor='white', framealpha=0)
        plt.yticks(range(sum_all.shape[0])[::-1], sum_all.index)
        plt.xlabel('Standardized Mean Difference', fontsize='large')
        plt.savefig(os.path.abspath(os.path.join(self.outpath, self.config['plot']['mean_difference_balance_plot_name'])))
        plt.close()
        ### statistic balance
        self.balance_test_plot(balance_stats, target, list(sum_all.index))
        ### Feature balance
        #self.feature_distribution(features)
    
    def output_figures(self, match, pairs, target, features, sum_all, sum_matched, balance_stats):
        # QQ plot
        self.eQQ_plot(match, pairs, target, features)
        # Hist plot
        self.Hist_plot(match, pairs, target)
        # Jitter plot
        self.Jitter_plot(match, pairs, target)
        # Summary plot
        self.Covariate_Balance_plot(sum_all, sum_matched, balance_stats, target, features)

    def propensity_score_match(self, data, target, features):
        ### data
        self.load_data(data)
        self.check_output_dir()
        psm = r("psm <- matchit(formula=%s~%s, data=data, method='%s', distance='%s', link='%s', caliper=%s, ratio=%s, replace=%s)"%(target, '+'.join(features), self.config['psm']['method'], self.config['psm']['distance'], self.config['psm']['link'], self.config['psm']['caliper'], self.config['psm']['ratio'], self.config['psm']['replace']))
        sample_size, match_drop_unmatched, match, pairs, sum_all, sum_matched, balance_stats = self.output_tables(target)
        self.output_figures(match, pairs, target, features, sum_all, sum_matched, balance_stats)
        return sample_size, match_drop_unmatched, match, pairs, sum_matched, balance_stats
    
    def simu_match(self,data, target, features,caliper,ratio):
        self.load_data(data)
        self.check_output_dir()
        psm = r("psm <- matchit(formula=%s~%s, data=data, method='nearest', distance='glm', link='logit', caliper=%s, ratio=%s, replace=TRUE)"%(target, '+'.join(features),caliper,ratio))
        sample_size, match_drop_unmatched, match, pairs, sum_all, sum_matched, balance_stats = self.output_tables(target)
        self.output_figures(match, pairs, target, features, sum_all, sum_matched, balance_stats)
        return sample_size, match_drop_unmatched, match, pairs, sum_matched, balance_stats
    
    def variance_pca(self, data, max_pc=np.nan, explained_variance_threshold=np.nan, min_pc=np.nan):
        self.check_output_dir()
        if pd.isnull(max_pc):
            max_pc = int(self.config['pca']['pca_max_pc_number'])
        if pd.isnull(min_pc):
            min_pc = int(self.config['pca']['pca_min_pc_number'])
        if pd.isnull(explained_variance_threshold):
            explained_variance_threshold = float(self.config['pca']['pca_explained_variance_threshold'])
        pca = PCA(n_components=min(max_pc, data.shape[0], data.shape[1]))
        X = pca.fit_transform(data)
        cumulative_explained_variance = 0
        for k in range(len(pca.explained_variance_ratio_)):
            cumulative_explained_variance += pca.explained_variance_ratio_[k]
            #print(k, cumulative_explained_variance)
            if cumulative_explained_variance >= explained_variance_threshold:
                break
        pc_data = pd.DataFrame(X[:, :max(k, min_pc)], index=data.index, columns=['PC'+str(i) for i in range(max(k, min_pc))])
        pickle.dump([pca, data, pc_data], open(os.path.abspath(os.path.join(self.outpath, self.config['pca']['pca_model_save_filename'])), 'wb'))
        return cumulative_explained_variance, pc_data

def run_miMatch(data, target, params, is_pca):
    psm = miMatch()
    for p1, p2, p3 in params:
        psm.set_config(p1, p2, str(p3))
    features = [i for i in list(data.columns) if i != target]
    if is_pca:
        cumulative_explained_variance, data_pca = psm.variance_pca(data.loc[:, features])
        features = list(data_pca.columns)
        data_pca[target] = data[target]
        data = data_pca
    sample_size, match_drop_unmatched, match, pairs, sum_matched, balance_stats = psm.propensity_score_match(data, target=target, features=features)
    return sample_size, match_drop_unmatched, match, pairs, sum_matched, balance_stats

def run_simu_match(data,target,params, features,caliper,ratio):
    psm = miMatch()
    for p1, p2, p3 in params:
        psm.set_config(p1, p2, str(p3))
    sample_size, match_drop_unmatched, match, pairs, sum_matched, balance_stats = psm.simu_match(data, target=target, features=features,caliper=caliper,ratio=ratio)
    return sample_size, match_drop_unmatched, match, pairs, sum_matched, balance_stats

def run_miMatch_subprocess(data, target, params, is_pca=True):
    pool = Pool(processes=1)
    res = pool.apply_async(run_psm, args=(data, target, params, is_pca))
    pool.close()
    pool.join()
    return res.get()

#correct if the population S.D. is expected to be equal for the two groups.
#https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
#https://www.cnblogs.com/HuZihu/p/12009535.html
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return np.abs(np.mean(x)-np.mean(y))/np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2+(ny-1)*np.std(y, ddof=1) ** 2)/dof)

# Wilcoxon-signed-rank-test主要用于成对样品(stats.wilcoxon), Mann-Whitney-U-test(Wilcoxon rank-sum)用于两组样品(stats.mannwhitneyu)
# http://blog.sciencenet.cn/blog-306699-984510.html
# https://blog.csdn.net/chang349276/article/details/76344979
def wilcoxon_signed_rank_test(x, y):
    # x, y 需为配对样本
    res = stats.wilcoxon(x, y)
    return res

def wilcoxon_rank_sum_test(x, y):
    res = stats.mannwhitneyu(x ,y)
    return res

def diff_by_rank_sum(data, target, features):
    result = []
    for f in features:
        x = data.loc[data[target]==0, f]
        y = data.loc[data[target]==1, f]
        try:
            stat, p = wilcoxon_rank_sum_test(x, y)
        except:
            stat, p = [None, 1.0]
       
        result.append([len(x), x.mean(), len(y), y.mean(), cohen_d(x, y), y.mean()/x.mean() if x.mean()!=0 else np.inf, p])
    result = pd.DataFrame(result, columns=['N Control', 'Mean Control', 'N Case', 'Mean Case', "cohen's d", 'Fold change', 'p-value'], index=features)
    result['fdr'] = multi.multipletests(result['p-value'], method = 'fdr_bh')[1]
    return result

def get_matched_pair_set(pairs):
    pair_set = []
    for i in pairs.index:
        for j in pairs.loc[i, :]:
            if (str(i) == 'NA_character_' or str(i).upper() == 'NAN') or (str(j) == 'NA_character_' or str(j).upper() == 'NAN'):
                    continue
            pair_set.append([i, j])
    pair_set = pd.DataFrame(pair_set, columns=['Case', 'Control'])
    pair_set = pair_set.dropna(how='any')
    return pair_set

def diff_by_signed_rank(data, pairs, features):
    pair_set = get_matched_pair_set(pairs)
    result = []
    for f in features:
  
        y = data.loc[pair_set['Case'], f]
        x = data.loc[pair_set['Control'], f]
        try:
            stat, p = wilcoxon_signed_rank_test(x, y)
        except:
            stat, p = [None, 1.0]
        result.append([len(x), x.mean(), len(y), y.mean(), cohen_d(x, y), y.mean()/x.mean() if x.mean()!=0 else np.inf, p])
    result = pd.DataFrame(result, columns=['N Control', 'Mean Control', 'N Case', 'Mean Case', "cohen's d", 'Fold change', 'p-value'], index=features)
    result['fdr'] = multi.multipletests(result['p-value'], method = 'fdr_bh')[1]
    return result

def test():
    psm = miMatch()
    data = psm.create_test_data()
    sample_size, match_drop_unmatched, match, pairs, sum_matched, balance_stats = psm.propensity_score_match(data, target='Group', features=['F1', 'F2', 'F3'])
    print(diff_by_rank_sum(data, target='Group', features=['F1', 'F2', 'F3']))
    print(diff_by_signed_rank(data, pairs,features=['F1', 'F2', 'F3']))

if __name__ == '__main__':
    pass
    #test()

