import pandas as pd
from scipy.stats import chi2_contingency, ranksums


def X_cat_syn_eval(X_pos_cat, X_pos_syn_cat):
    """evaluate raw categorical data with synthetic categorical data"""
    high_sim_count = 0
    for c in X_pos_cat.columns:
        p = chi2_contingency(pd.crosstab(X_pos_cat[c], X_pos_syn_cat[c]))[1]
        if p < 0.05:
            print(c + ":", p)
            high_sim_count = high_sim_count +1
    return high_sim_count


def X_num_syn_eval(X_pos_num, X_pos_syn_num):
    high_sim_count = 0
    for c in X_pos_num.columns:
        p = ranksums(X_pos_num[c].dropna(), X_pos_syn_num[c].dropna()).pvalue
        if p > 0.05:
            print(c + ":", p)
            high_sim_count = high_sim_count +1
    return high_sim_count
