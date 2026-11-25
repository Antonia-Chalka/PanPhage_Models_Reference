# %% Load libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Single-phage models performance

single_path = '../data/single-phage-models/'

# Iterate over the Monte Carlo CV runs

iterations = [f for f in os.listdir(single_path)
              if not f.startswith('.')]

singlephage_metrics = {
    'f1': {},
    'rmse': {},
    'phi': {},
    'f1_phage': {},
    'rmse_phage': {},
    'phi_phage': {}
}

for i in iterations:
    predictions_path = single_path + i
    prediction_files = [f for f in os.listdir(predictions_path)
                        if f.endswith('_predictions.csv')
                        and not f.startswith('.')]

    singlephage_metrics['f1_phage'][i] = {}
    singlephage_metrics['rmse_phage'][i] = {}
    singlephage_metrics['phi_phage'][i] = {}

    cumulative_tp, cumulative_tn, cumulative_fp, cumulative_fn = 0, 0, 0, 0
    cumulative_squared_error, cumulative_n_test = 0.0, 0

    for csv in prediction_files:
        df = pd.read_csv(predictions_path + '/' + csv)
        phage = csv.split('_')[1]

        df['class'] = np.zeros(len(df), dtype='<U2')
        df['binary-observed'] = df['observed'] < 60
        df['binary-predicted'] = df['predicted'] < 60
        tp_mask = df['binary-observed'] & df['binary-predicted']
        df.loc[tp_mask, 'class'] = 'TP'
        tn_mask = ~df['binary-observed'] & ~df['binary-predicted']
        df.loc[tn_mask, 'class'] = 'TN'
        fp_mask = ~df['binary-observed'] & df['binary-predicted']
        df.loc[fp_mask, 'class'] = 'FP'
        fn_mask = df['binary-observed'] & ~df['binary-predicted']
        df.loc[fn_mask, 'class'] = 'FN'

        df['square-error'] = (df['predicted'] - df['observed'])**2

        test_mask = df['dataset'] == 'test'

        test_squared_errors = df.loc[test_mask, 'square-error']
        cumulative_squared_error += test_squared_errors.sum()
        cumulative_n_test += test_squared_errors.count()

        rmse = test_squared_errors.mean()**.5
        singlephage_metrics['rmse_phage'][i][phage] = rmse

        tp = (df[test_mask]['class'] == 'TP').sum()
        tn = (df[test_mask]['class'] == 'TN').sum()
        fp = (df[test_mask]['class'] == 'FP').sum()
        fn = (df[test_mask]['class'] == 'FN').sum()

        singlephage_metrics['f1_phage'][i][phage] = 2 * tp / (2 * tp + fp + fn)

        phi = ((tp * tn - fp * fn) /
               ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**.5)
        singlephage_metrics['phi_phage'][i][phage] = phi

        cumulative_tp += tp
        cumulative_tn += tn
        cumulative_fp += fp
        cumulative_fn += fn

    rmse = (cumulative_squared_error / cumulative_n_test)**.5
    singlephage_metrics['rmse'][i] = rmse

    f1 = 2 * cumulative_tp / (2 * cumulative_tp
                              + cumulative_fp + cumulative_fn)
    singlephage_metrics['f1'][i] = f1

    phi = ((cumulative_tp * cumulative_tn - cumulative_fp * cumulative_fn) /
           ((cumulative_tp + cumulative_fp)
            * (cumulative_tp + cumulative_fn)
            * (cumulative_tn + cumulative_fp)
            * (cumulative_tn + cumulative_fn))**.5)
    singlephage_metrics['phi'][i] = phi

# %% PanPhage models performance

panphage_path = '../data/panphage-models/'

# Iterate over the Monte Carlo CV runs

iterations = [f for f in os.listdir(panphage_path)
              if not f.startswith('.')]

panphage_metrics = {
    'f1': {},
    'rmse': {},
    'phi': {},
    'f1_phage': {},
    'rmse_phage': {},
    'phi_phage': {}
}

for i in iterations:
    predictions_path = panphage_path + i
    df = pd.read_csv(predictions_path + '/predictions.csv', index_col=0)

    df['class'] = np.zeros(len(df), dtype='<U2')
    df['binary-observed'] = df['observed'] < 60
    df['binary-predicted'] = df['predicted'] < 60
    tp_mask = df['binary-observed'] & df['binary-predicted']
    df.loc[tp_mask, 'class'] = 'TP'
    tn_mask = ~df['binary-observed'] & ~df['binary-predicted']
    df.loc[tn_mask, 'class'] = 'TN'
    fp_mask = ~df['binary-observed'] & df['binary-predicted']
    df.loc[fp_mask, 'class'] = 'FP'
    fn_mask = df['binary-observed'] & ~df['binary-predicted']
    df.loc[fn_mask, 'class'] = 'FN'

    df['square-error'] = (df['predicted'] - df['observed'])**2

    test_mask = df['dataset'] == 'test'
    rmse = df[test_mask]['square-error'].mean()**.5
    panphage_metrics['rmse'][i] = rmse

    tp = (df[test_mask]['class'] == 'TP').sum()
    tn = (df[test_mask]['class'] == 'TN').sum()
    fp = (df[test_mask]['class'] == 'FP').sum()
    fn = (df[test_mask]['class'] == 'FN').sum()

    panphage_metrics['f1'][i] = 2 * tp / (2 * tp + fp + fn)

    phi = ((tp * tn - fp * fn) /
           ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**.5)
    panphage_metrics['phi'][i] = phi

    # Performance by phage
    phages = np.unique([interaction[1]
                        for interaction in df.index.str.split('_')])

    panphage_metrics['rmse_phage'][i] = {}
    panphage_metrics['f1_phage'][i] = {}
    panphage_metrics['phi_phage'][i] = {}

    for phage in phages:
        phage_mask = df.index.str.endswith('_' + phage)

        rmse = df[phage_mask & test_mask]['square-error'].mean()**.5
        panphage_metrics['rmse_phage'][i][phage] = rmse

        tp = (df[phage_mask & test_mask]['class'] == 'TP').sum()
        tn = (df[phage_mask & test_mask]['class'] == 'TN').sum()
        fp = (df[phage_mask & test_mask]['class'] == 'FP').sum()
        fn = (df[phage_mask & test_mask]['class'] == 'FN').sum()

        panphage_metrics['f1_phage'][i][phage] = 2 * tp / (2 * tp + fp + fn)

        phi = ((tp * tn - fp * fn) /
               ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**.5)
        panphage_metrics['phi_phage'][i][phage] = phi

# %% Check test/train splits are the same for single and panphage models

for i in iterations:
    df_pan = pd.read_csv(panphage_path + i + '/predictions.csv', index_col=0)
    for phage in phages:
        df_sin = pd.read_csv(single_path + i + '/phage_' + phage
                             + '_predictions.csv', index_col=0)
        phage_mask = df_pan.index.str.endswith('_' + phage)
        test_mask_pan = phage_mask & (df_pan['dataset'] == 'test')
        test_mask_sin = df_sin['dataset'] == 'test'
        test_pan = [interaction[0] for interaction
                    in df_pan[test_mask_pan].index.str.split('_')]
        test_sin = [interaction[0] for interaction
                    in df_sin[test_mask_sin].index.str.split('_')]
        coincidencias = np.array([bug in test_sin for bug in test_pan])
        if coincidencias.sum() != len(test_pan):
            raise NameError
        elif len(test_pan) != len(test_sin):
            raise NameError
print('Equal train-test splits for equal random seed :)')

# %% Null models performance

null_metrics = {
    'rmse': {},
    'f1': {},
    'phi': {}
}

rng = np.random.default_rng(seed=22266627227280132390902731719189379021)

for i in iterations:
    df = pd.read_csv(panphage_path + i + '/predictions.csv', index_col=0)
    train_mask = df['dataset'] == 'train'

    # Continious null model
    df['null-prediction'] = df[train_mask]['observed'].mean()
    df['squared-error'] = (df['observed']
                           - df['null-prediction'])**2

    null_metrics['rmse'][i] = df[~train_mask]['squared-error'].mean()**.5

    # Binary null model
    df['binary-observed'] = df['observed'] < 60
    p = df[train_mask]['binary-observed'].sum() / train_mask.sum()
    df['binary-null'] = rng.binomial(1, p, size=len(df))

    tp_mask = df['binary-observed'] & df['binary-null']
    df.loc[tp_mask, 'class'] = 'TP'
    tn_mask = ~df['binary-observed'] & ~df['binary-null']
    df.loc[tn_mask, 'class'] = 'TN'
    fp_mask = ~df['binary-observed'] & df['binary-null']
    df.loc[fp_mask, 'class'] = 'FP'
    fn_mask = df['binary-observed'] & ~df['binary-null']
    df.loc[fn_mask, 'class'] = 'FN'

    tp = (df[~train_mask]['class'] == 'TP').sum()
    tn = (df[~train_mask]['class'] == 'TN').sum()
    fp = (df[~train_mask]['class'] == 'FP').sum()
    fn = (df[~train_mask]['class'] == 'FN').sum()

    null_metrics['f1'][i] = 2 * tp / (2 * tp + fp + fn)

    phi = ((tp * tn - fp * fn) /
           ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**.5)
    null_metrics['phi'][i] = phi

# %% Plot overal metrics:

fig, ax = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

ax[1].set_title('128 repeats (single-phage, PanPhage, & null)')

ax[0].set_xlabel('RMSE')
ax[1].set_xlabel('F1')
ax[2].set_xlabel(r'$\phi$')

ax[0].set_ylabel('counts')

# Single-phage models
ax[0].hist([singlephage_metrics['rmse'][i]
            for i in singlephage_metrics['rmse'].keys()], alpha=.6)
ax[1].hist([singlephage_metrics['f1'][i]
            for i in singlephage_metrics['f1'].keys()], alpha=.6)
ax[2].hist([singlephage_metrics['phi'][i]
            for i in singlephage_metrics['phi'].keys()],
           label='single-phage', alpha=.6)

# PanPhage models
ax[0].hist([panphage_metrics['rmse'][i]
            for i in panphage_metrics['rmse'].keys()], alpha=.6)
ax[1].hist([panphage_metrics['f1'][i]
            for i in panphage_metrics['f1'].keys()], alpha=.6)
ax[2].hist([panphage_metrics['phi'][i]
            for i in panphage_metrics['phi'].keys()],
           label='PanPhage', alpha=.6)

# Null models
ax[0].hist([null_metrics['rmse'][i]
            for i in null_metrics['rmse'].keys()], alpha=.6)
ax[1].hist([null_metrics['f1'][i]
            for i in null_metrics['f1'].keys()], alpha=.6)
ax[2].hist([null_metrics['phi'][i]
            for i in null_metrics['phi'].keys()],
           label='null', alpha=.6)

for a in ax:
    lims = a.get_ylim()
    a.set_ylim(lims)

# First training
ax[0].vlines(singlephage_metrics['rmse']['100'], 0, lims[1], 'C0', '--')
ax[1].vlines(singlephage_metrics['f1']['100'], 0, lims[1], 'C0', '--')
ax[2].vlines(singlephage_metrics['phi']['100'], 0, lims[1], 'C0', '--')

# PanPhage
ax[0].vlines(14.98, 0, lims[1], 'C1', '--')
ax[1].vlines(.7, 0, lims[1], 'C1', '--')
ax[2].vlines(0.66, 0, lims[1], 'C1', '--')

# Null model
ax[0].vlines(22, 0, lims[1], 'C2', '--')
ax[1].vlines(0.14769652573630593, 0, lims[1], 'C2', '--')
ax[2].vlines(0, 0, lims[1], 'C2', '--')

# Theoretical values
# std of observed data
ax[0].vlines(22.02755617639293, 0, lims[1], 'k', (0, (1, 3)))
# phi = 0 for bernoulli distributed labels
ax[2].vlines(0, 0, lims[1], 'k', (0, (1, 3)), label='theretical values')

ax[2].legend(loc='upper left', fontsize=9)
fig.set_tight_layout(True)
# plt.savefig('prelim-MCCV-performances.png')

# %%
