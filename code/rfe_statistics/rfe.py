#!/usr/bin/env python3
"""
Recursive feature elimination

This script reads some training data and then run a recursive feature 
elimination process. The features selected are exported togheter with the 
test/train split used.

"""

# Load Libraries
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn import set_config
from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict

def main() -> None:
    print('=============================================================================', flush=True)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(f'Initialising {__file__}', flush=True)

    # Seed
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int, help='Random seed to use')
    args = parser.parse_args()
    seed = args.seed
    if not -1 < seed < 2**32:
        print('Seed value must be in the range [0, 2**32 - 1]', file=sys.stderr, flush=True)
        sys.exit(1)

    # Input data
    dtypes = defaultdict(lambda: bool)
    dtypes['Score'] = float
    dtypes[0] = str

    print('Loading interaction data...', flush=True)
    interaction_data = pd.read_csv('../../data/training_data.tsv', delimiter='\t', dtype=dtypes) #TODO CHANGE & rerun with different sets
    print('Data loaded', flush=True)

    # Parameters
    random_state = seed  # for reproducibility
    ratio_test = 0.25  # Train/Test ratios
    # REF
    perc_step = 0.1 
    n_features_first=2000

    set_config(transform_output="pandas")

    # Construct full output paths
    output_dir='4.outputs/panaroo_default/stat_sel_features' #TODO CHANGE PATH AS NEEDED
    os.makedirs(os.path.join(output_dir, str(seed)), exist_ok=True)
    test_train_split_path = os.path.join(output_dir, str(seed), 'test_train_split.csv')
    rfe_features_path = os.path.join(output_dir, str(seed), 'rfe_features.txt')

    # RFE and RFECV
    gradient_boosting = GradientBoostingRegressor(n_estimators=1000, random_state=seed)

    rfe_init = RFE(estimator=gradient_boosting,
                    n_features_to_select=n_features_first,
                    step=perc_step,
                    verbose=100)

    x = interaction_data.drop(columns=['Score'])
    y = interaction_data['Score']

    # Stratify based on whether score is above or below 60
    y_strat = (y >= 60).astype(int)
    # ALD1 only has 1 positive interaction. We must label it as negative or train_test_split will raise exception
    y_strat['CAN98_ALD1'] = 1
    # For Panphage, we can further stratify by phage ID to ensure representation
    indexes = interaction_data.index
    phages = [idx.split('_')[1] for idx in indexes]
    combined_strat = [f"{phage}_{y_strat}" for phage, y_strat in zip(phages, y_strat)]

    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size=ratio_test, 
                                                        random_state=seed,
                                                        stratify=combined_strat)

    # ---- 1. First RFE ----
    print('Starting RFE', flush=True)
    rfe_stage1 = rfe_init.fit(x_train, y_train)
    #X_train_s1 = rfe_stage1.transform(x_train)
    X_test_s1  = rfe_stage1.transform(x_test)

    # Save selected features:
    with open(rfe_features_path, 'w') as txt:
        for feature in X_test_s1.columns.values:
            txt.write(feature + '\n')

    # Export test/train split:
    y_test = pd.DataFrame(y_test)
    y_test['dataset'] = 'test'
    y_train = pd.DataFrame(y_train)
    y_train['dataset'] = 'train'
    split = pd.concat([y_train, y_test])
    split.drop(columns='Score').to_csv(test_train_split_path)

    print('RFE done! :D', flush=True)
    print(datetime.now().strftime("%H:%M:%S"), flush=True)
    print('=============================================================================', flush=True)

if __name__ == '__main__':
    main()