# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:----------------------------
    CTG_features = CTG_features.apply(pd.to_numeric,errors='coerce')
    c_ctg={}
    for feature in CTG_features.columns:
        c_ctg[feature] = CTG_features[feature].dropna()
    del c_ctg[extra_feature]
    # --------------------------------------------------------------------------
    
    return(c_ctg)


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg=rm_ext_and_nan(CTG_features, extra_feature)
    CTG_features = CTG_features.apply(pd.to_numeric,errors='coerce')
    cols=['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DR', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
    for feature in cols:
        for i in range(1,len(CTG_features[feature])+1):
            if np.isnan(CTG_features.at[i,feature]):
                CTG_features.at[i,feature]=np.random.choice(c_ctg[feature])
    for feature in CTG_features.columns:
        c_cdf[feature]=CTG_features[feature]
    del c_cdf[extra_feature]
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary={}
    for feature in c_feat.columns:
        d_summary[feature]={'min':c_feat[feature].min(),'Q1':c_feat[feature].quantile(0.25),'median':c_feat[feature].quantile(0.5),'Q3':c_feat[feature].quantile(0.75),'max':c_feat[feature].max()}
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_feat_copy = c_feat.copy()
    for feature in c_feat_copy.columns:
        outlier_big=d_summary[feature]['Q3']+(1.5*(d_summary[feature]['Q3']-d_summary[feature]['Q1']))
        outlier_small=d_summary[feature]['Q1']-(1.5*(d_summary[feature]['Q3']-d_summary[feature]['Q1']))
        for i in range(1, len(c_feat_copy[feature]) + 1):
            if (c_feat_copy.at[i,feature] > outlier_big) or (c_feat_copy.at[i,feature] < outlier_small):
                c_feat_copy.at[i,feature]=np.nan
    for feature in c_feat_copy.columns:
        c_no_outlier[feature] = c_feat_copy[feature]
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature=c_cdf.copy()
    feat_col=filt_feature[feature]
    for i in range(1,len(feat_col)+1):
        if feat_col[i] > thresh:
            feat_col[i]=np.nan
    filt_feature.replace(filt_feature[feature],feat_col)
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nsd_res = {}
    for feature in CTG_features.columns:
        mean_feat = sum(CTG_features[feature])/len(CTG_features[feature])
        min_feat= CTG_features[feature].min()
        max_feat = CTG_features[feature].max()
        std_feat = np.std(CTG_features[feature])
        if mode =='standard':
            nsd_res[feature]=(CTG_features[feature]-mean_feat)/std_feat
        elif mode =='MinMax':
            nsd_res[feature]=(CTG_features[feature]-min_feat)/(max_feat-min_feat)
        elif mode =='mean':
            nsd_res[feature]=(CTG_features[feature]-mean_feat)/(max_feat-min_feat)
        else:
            nsd_res[feature]=CTG_features[feature]
    if flag == True:
        h1 = nsd_res[x].hist(bins=40)
        h1.set_xlabel(f"mode = {mode}")
        h1.set_ylabel ("count")
        h1.set_title(f"{x}")
        plt.show()

        h2 = nsd_res[y].hist(bins=40)
        h2.set_xlabel(f"mode = {mode}")
        h2.set_ylabel("count")
        h2.set_title(f"{y}")
        plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
