
import os
import itertools
import sys

import pandas as pd
import numpy as np
import scipy.stats as spstats

# ###############################################################
# STATS
# ###############################################################
   
# Stats
def do_mannwhitney(mdf, metric='I_rs', multi_comp_test='holm'):
    '''
    bonferroni : one-step correction

    sidak : one-step correction

    holm-sidak : step down method using Sidak adjustments

    holm : step-down method using Bonferroni adjustments

    simes-hochberg : step-up method (independent)

    hommel : closed method based on Simes tests (non-negative)

    fdr_bh : Benjamini/Hochberg (non-negative)

    fdr_by : Benjamini/Yekutieli (negative)

    fdr_tsbh : two stage fdr correction (non-negative)

    fdr_tsbky : two stage fdr correction (non-negative)
    '''
    
    #import statsmodels.api as sm

    visual_areas = ['V1', 'Lm', 'Li']
    mpairs = list(itertools.combinations(visual_areas, 2))

    pvalues = []
    stats = []
    nsamples = []
    for mp in mpairs:
        d1 = mdf[mdf['visual_area']==mp[0]][metric]
        d2 = mdf[mdf['visual_area']==mp[1]][metric]

        # compare samples
        stat, p = spstats.mannwhitneyu(d1, d2)
        n1=len(d1)
        n2=len(d2)

        # interpret
        alpha = 0.05
        if p > alpha:
            interp_str = '... Same distribution (fail to reject H0)'
        else:
            interp_str = '... Different distribution (reject H0)'
        # print('[%s] Statistics=%.3f, p=%.3f, %s' % (str(mp), stat, p, interp_str))

        pvalues.append(p)
        stats.append(stat)
        nsamples.append((n1, n2))

    reject, pvals_corrected, _, _ = sm.stats.multitest.multipletests(pvalues,
                                                                     alpha=0.05,
                                                                     method=multi_comp_test)
#    r_=[]
#    for mp, rej, pv, st, ns in zip(mpairs, reject, pvals_corrected, stats, nsamples):
#        print('[%s] p=%.3f (%s), reject H0=%s' % (str(mp), pv, multi_comp_test, rej))
#        r_.append(pd.Series({'d1': mp[0], 'd2': mp[1], 'n1': ns[0], 'n2': ns[1],
#                             'reject': rej, 'p_val': pv, 'U_val': st}))
#    results = pd.concat(r_, axis=1).T.reset_index(drop=True)
    results = pd.DataFrame({'d1': [mp[0] for mp in mpairs],
                            'd2': [mp[1] for mp in mpairs],
                            'reject': reject,
                            'p_val': pvals_corrected,
                            'U_val': stats,
                            'n1': [ns[0] for ns in nsamples],
                            'n2': [ns[1] for ns in nsamples]})
    print(results)

    return results


def paired_ttest_from_df(plotdf, metric='avg_size', c1='rfs', c2='rfs10', compare_var='experiment',
                            round_to=None, return_vals=False, ttest=True):
    
    a_vals = plotdf[plotdf[compare_var]==c1].sort_values(by='datakey')[metric].values
    b_vals = plotdf[plotdf[compare_var]==c2].sort_values(by='datakey')[metric].values
    #print(a_vals, b_vals)
    if ttest:
        tstat, pval = spstats.ttest_rel(np.array(a_vals), np.array(b_vals))
    else:
        tstat, pval = spstats.wilcoxon(np.array(a_vals), np.array(b_vals))

    #print('%s: %.2f (p=%.2f)' % (visual_area, tstat, pval))
    if round_to is not None:
        tstat = round(tstat, round_to)
        pval = round(pval, round_to)

    pdict = {'t_stat': tstat, 'p_val': pval}

    if return_vals:
        return pdict, a_vals, b_vals
    else:
        return pdict


def paired_ttests(comdf, metric='avg_size',  round_to=None,
                c1='rfs', c2='rfs10', compare_var='experiment', ttest=True,
                visual_areas=['V1', 'Lm', 'Li']):
    r_=[]
    for ai, visual_area in enumerate(visual_areas):

        plotdf = comdf[comdf['visual_area']==visual_area].copy()

        pdict = paired_ttest_from_df(plotdf, c1=c1, c2=c2, metric=metric,
                        compare_var=compare_var, round_to=round_to, return_vals=False, ttest=ttest)
        pdict.update({'visual_area': visual_area})
        res = pd.DataFrame(pdict, index=[ai])
        r_.append(res)

    statdf = pd.concat(r_, axis=0)

    return statdf



