#!/usr/bin/env python3
"""
Recursive feature elimination with cross-validation

This script reads the training data and the selected features from rfe.py. 
Then it runs a recursive feature elimination process with cross validation. 
The final features selected are exported.

"""

# Load Libraries
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn import set_config
from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict

def main() -> None:
    print('=================================================================', 
          flush=True)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(f'Initialising {__file__}', flush=True)

    # Load interaction data
    dtypes = defaultdict(lambda: bool)
    dtypes['Score'] = float
    dtypes[0] = str

    print('Loading interaction data...', flush=True)
    interaction_data = pd.read_csv('3.trainingdata_defaults.tsv', 
                                   delimiter='\t', dtype=dtypes)
    print('Data loaded', flush=True)
    
    # Read bash script arguments:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('rfe_seed', type=int, 
                        help='Random seed used by rfe.py')
    parser.add_argument('nCPUs', type=int, help='# of CPUs to use per task')
    args = parser.parse_args()
    seed = args.rfe_seed
    cores = args.nCPUs

    if not -1 < seed < 2**32:
        print('Seed value must be in the range [0, 2**32 - 1]', 
              file=sys.stderr, flush=True)
        sys.exit(1)
    print(f'Number of cores to be used {cores}', flush=True)
    
    # RFE output paths + construct path for RFECV output
    output_dir = '4.outputs/panaroo_default/stat_sel_features/' # TODO: remove testing/
    output_dir += str(seed)
    test_train_split_path = os.path.join(output_dir, 'test_train_split.csv')
    rfe_features_path = os.path.join(output_dir, 'rfe_features.txt')
    rfecv_features_path = os.path.join(output_dir, 'rfecv_features.txt')

    # Model parameters
    random_state = 100  # for reproducibility
    # RepeatedKFold parameters
    n_splits = 10   # TODO: Change to 10 after testing
    n_repeats = 3   # TODO: Change to 3 after testing
    # RFECV parameters
    perc_step = 0.1 #TODO: Change to 0.1 after testing
    n_features_final = 45

    set_config(transform_output="pandas")

    # Initialise scikit objects
    gradient_boosting = GradientBoostingRegressor(n_estimators=1000, 
                                                  random_state=random_state)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, 
                       random_state=random_state)
    rfecv = RFECV(gradient_boosting,
                step=perc_step,
                verbose=100,
                min_features_to_select=n_features_final,
                cv=cv,
                n_jobs=cores,
                scoring='neg_root_mean_squared_error')

    # Make test/train split
    ratio_test = 0.25  # Train/Test ratios

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

    # Load selected features in RFE:
    rfe_features = pd.read_csv(rfe_features_path, header=None, names=['feature'])
    X_train_s1 = x_train[rfe_features['feature'].values]
    X_test_s1  = x_test[rfe_features['feature'].values]

    # ---- 2. Second RFE (RFECV) ----
    rfe_stage2 = rfecv.fit(X_train_s1, y_train)
    #X_train_s2 = rfe_stage2.transform(X_train_s1)
    X_test_s2  = rfe_stage2.transform(X_test_s1)

    # Save selected features:
    with open(rfecv_features_path, 'w') as txt:
        for feature in X_test_s2.columns.values:
            txt.write(feature + '\n')

    # Check test/train split:   # TODO: comment this section
    # rfe_split = pd.read_csv(test_train_split_path, index_col=0)
    # y_test = pd.DataFrame(y_test)
    # y_test['dataset'] = 'test'
    # y_train = pd.DataFrame(y_train)
    # y_train['dataset'] = 'train'
    # split = pd.concat([y_train, y_test])
    # if not all((split.drop(columns='Score') == rfe_split).values):
    #     print(f'WARNING: split for seed #{seed} is different to the one saved by rfe.py', flush=True)
    #     split_path = os.path.join(output_dir, 'rfecv_split.csv')
    #     split.drop(columns='Score').to_csv(split_path)
    #     print('The current split was saved as rfecv_split.csv', flush=True)

    print('All RFECVs done! :D')
    print(datetime.now().strftime("%H:%M:%S"))
    print('=============================================================================', flush=True)

if __name__ == '__main__':
    main()