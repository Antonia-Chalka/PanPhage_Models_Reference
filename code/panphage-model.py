#!/usr/bin/env python3
"""
Panphage model training

This script receives a parsed random seed used for the test-train splitting.
"""

# Load Libraries
import pandas as pd
import numpy as np
import pickle
import os
import sys
from io import TextIOWrapper
from typing import cast
from sklearn.model_selection import (train_test_split, RepeatedKFold,
                                     RandomizedSearchCV)
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import set_config
from datetime import datetime
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from time import perf_counter


def main() -> None:
    # Set flush=True by default for print function
    cast(TextIOWrapper, sys.stdout).reconfigure(line_buffering=True)
    print('==================================================================')
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f'Initialising {__file__}')

    # Read bash script arguments:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('seed', type=int, help='Random seed to use')
    parser.add_argument('nCPUs', type=int, help='# of CPUs to use per task')
    args = parser.parse_args()

    seed = args.seed
    if not -1 < seed < 2**32:
        print('Seed value must be in the range [0, 2**32 - 1]',
              file=sys.stderr)
        sys.exit(1)
    print(f'Random seed: {seed}')

    cores = args.nCPUs
    print(f'Number of cores to be used {cores}')

    # Parameters
    random_state = seed

    ratio_test = 0.25   # Train/Test ratios

    # RepeatedKFold parameters
    n_splits = 10
    n_repeats = 3

    # RFE parameters
    perc_step = 0.1
    # Final number of features to select after first round of RFE
    n_features_first = 2000
    # Minimum number of features to select after RFECV
    n_features_final = 45

    # Load interaction data and panphage matrix
    dtypes: defaultdict[str | int, type] = defaultdict(lambda: bool)
    dtypes['Score'] = float
    dtypes[0] = str

    data_path = (Path(__file__).resolve().parent.parent
                 / 'data/training_data.tsv')
    print(f'Python {sys.version.split()[0]}, pandas {pd.__version__}')
    print(f'Loading interaction data from {data_path} '
          f'({data_path.stat().st_size / 1024**2:.1f} MiB)...')
    started = perf_counter()
    chunks = []
    rows_loaded = 0
    # For this very wide matrix, pandas' default low-memory parsing creates
    # many tiny internal chunks. Bound memory with explicit row chunks instead.
    # The TSV has an unnamed leading index field, inferred by pandas; retain
    # the dtype mapping above to keep IDs as strings and Score as float.
    with pd.read_csv(data_path, delimiter='\t', dtype=dtypes,
                     chunksize=512, low_memory=False) as reader:
        for chunk in reader:
            chunks.append(chunk)
            rows_loaded += len(chunk)
            print(f'  Loaded {rows_loaded:,} rows '
                  f'in {perf_counter() - started:.1f} s')
    interaction_data = pd.concat(chunks)
    del chunks, chunk
    memory_mib = (
        interaction_data.memory_usage(index=True, deep=True).sum() / 1024**2
    )
    print(f'Data loaded: {interaction_data.shape[0]:,} rows x '
          f'{interaction_data.shape[1]:,} columns '
          f'in {perf_counter() - started:.1f} s; '
          f'DataFrame memory: {memory_mib:.1f} MiB (excludes parser overhead)')

    # Set output directory
    output_dir = f'../output/panphage/{random_state}/'
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    # Construct full output paths
    predictions_path = os.path.join(output_dir, 'predictions.csv')
    feature_importance_path = os.path.join(output_dir,
                                           'feature_importances.csv')
    model_path = os.path.join(output_dir, 'best_xgb_model.pkl')
    bacteria_features_path = os.path.join(output_dir, 'feature_bacteria.txt')
    phage_features_path = os.path.join(output_dir, 'feature_phage.txt')

    # Set pipeline configuration
    set_config(transform_output="pandas")
    gradient_boosting = GradientBoostingRegressor(n_estimators=1000,
                                                  random_state=random_state)
    cv = RepeatedKFold(n_splits=n_splits,
                       n_repeats=n_repeats,
                       random_state=random_state)
    rfecv = RFECV(
        gradient_boosting,
        step=perc_step,
        verbose=0,
        min_features_to_select=n_features_final,
        cv=cv,
        n_jobs=cores,
        scoring='neg_root_mean_squared_error'
    )
    rfe_init = RFE(
        estimator=gradient_boosting,
        n_features_to_select=n_features_first,
        step=perc_step,
        verbose=0
    )
    param_grid = {
        # Number of trees in the boosting ensemble
        'n_estimators': [500, 750, 1000, 1250],
        # Depth of the individual trees
        'max_depth': [3, 5, 10, 20],
        # Learning rate for boosting
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        # Fraction of samples used for fitting each tree
        'subsample': [0.8, 0.9, 1.0],
        # Minimum number of samples required to split a node
        'min_samples_split': [2, 5, 10],
        # Minimum number of samples required in each leaf
        'min_samples_leaf': [1, 2, 4]
    }
    tuning = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=random_state),
        param_distributions=param_grid,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        verbose=0,
        n_jobs=cores,
        n_iter=25,
        refit=True,
        random_state=random_state
    )

    # Test-train split

    x = interaction_data.drop(columns=['Score'])
    y = interaction_data['Score']

    # Stratify based on whether score is above or below 60
    y_strat = (y >= 60).astype(int)
    # ALD1 only has 1 positive interaction. We must label it as negative or
    # train_test_split will raise exception. In this case, it will be random if
    # this interaction is in the train or test set
    y_strat['CAN98_ALD1'] = 1

    # We need to further stratify by phage so each phage has the same
    # train-test split proportion
    indexes = interaction_data.index
    phages = [idx.split('_')[1] for idx in indexes]
    combined_strat = [f"{phage}_{y_strat}" for phage,
                      y_strat in zip(phages, y_strat)]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=ratio_test,
        random_state=random_state,
        stratify=combined_strat)

    # Train and test models

    # ---- 1. First RFE ----
    rfe_stage1 = rfe_init.fit(x_train, y_train)
    X_train_s1 = rfe_stage1.transform(x_train)
    X_test_s1 = rfe_stage1.transform(x_test)

    # ---- 2. Second RFE (RFECV) ----
    rfe_stage2 = rfecv.fit(X_train_s1, y_train)
    X_train_s2 = rfe_stage2.transform(X_train_s1)
    X_test_s2 = rfe_stage2.transform(X_test_s1)

    # ---- 3. Model tuning ----
    tuning.fit(X_train_s2, y_train)
    best_model = cast(GradientBoostingRegressor,
                      tuning.best_estimator_)

    # Save the best model
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # ---- 4. Predict on test data ----
    y_pred = best_model.predict(X_test_s2)
    y_pred_train = best_model.predict(X_train_s2)

    train_results = pd.DataFrame({
        'observed': y_train,
        'predicted': y_pred_train,
        'dataset': "train"
    })
    test_results = pd.DataFrame({
        'observed': y_test,
        'predicted': y_pred,
        'dataset': "test"
    })

    combined_results = pd.concat([train_results, test_results])
    combined_results.to_csv(predictions_path, index=True)

    importance_df = pd.DataFrame({
        'Feature': best_model.feature_names_in_,
        'Importance': best_model.feature_importances_,
    })
    importance_df.to_csv(feature_importance_path, index=False)

    # Filter features for bacteria and phage
    features_bacteria = [feature.replace('_bacteria', '')
                         for feature in best_model.feature_names_in_
                         if feature.endswith('_bacteria')]
    features_phage = [feature.replace('_phage', '')
                      for feature in best_model.feature_names_in_
                      if feature.endswith('_phage')]
    print((f"Found {len(features_bacteria)} bacteria features and "
           f"{len(features_phage)} phage features"))

    with open(bacteria_features_path, 'w') as file:
        for feature in features_bacteria:
            file.write(feature + '\n')
    with open(phage_features_path, 'w') as file:
        for feature in features_phage:
            file.write(feature + '\n')


if __name__ == '__main__':
    main()
