# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import logging
from sklearn.model_selection import train_test_split, RepeatedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Machine Learning Pipeline for Feature Selection and Model Training')
    
    # Input arguments
    parser.add_argument('--input', '-i', 
                    default='./data/1.training/3.trainingdata.tsv',
                    help='Input training data file (default: ./data/1.training/3.trainingdata.tsv)')
    # Output arguments
    parser.add_argument('--output-dir', '-o',
                    default='.',
                    help='Output directory for results (default: current directory)')
    
    return parser.parse_args()

#interaction_data = pd.read_csv('./data/1.training/3.trainingdata.tsv', delimiter='\t')

def main():

    # Setup logging
    logger.info("Starting Phage Training Pipeline...")

    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    
    # Construct full output paths
    feature_plot_path = os.path.join(args.output_dir, 'feature_reduction.png')
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    feature_importance_path = os.path.join(args.output_dir, 'feature_importances.csv')
    model_path = os.path.join(args.output_dir, 'best_xgb_model.pkl')
    bacteria_features_path = os.path.join(args.output_dir, 'feature_bacteria.txt')
    phage_features_path = os.path.join(args.output_dir, 'feature_phage.txt')

    
    # Load data
    logger.info(f"Loading data from: {args.input}")
    try:
        interaction_data = pd.read_csv(args.input, delimiter='\t')
        logger.info(f"Data loaded successfully. Shape: {interaction_data.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Model parameters
    random_state = 100  # for reproducibility
    cores = -1  # -1 uses all cores
    ratio_test = 0.25  # Train/Test ratios

    # RepeatedKFold parameters
    n_splits = 5
    n_repeats = 2

    # REF
    perc_step = 1000 
    n_features_final = 300

    logger.info("Initializing model components...")
    logger.info(f"Model parameters: random_state={random_state}, cores={cores}, test_ratio={ratio_test}")
    logger.info(f"Cross-validation: {n_splits} splits, {n_repeats} repeats")
    logger.info(f"Feature selection: step={perc_step}, final_features={n_features_final}")

    gradient_boosting = GradientBoostingRegressor(n_estimators=1000, random_state=random_state)

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    # Initialize RFECV with GradientBoostingRegressor
    # This will perform recursive feature elimination with cross-validation
    # The step parameter controls how many features to remove at each iteration
    # The min_features_to_select parameter ensures that at least n_features_final features are selected
    logger.info("Initializing RFECV...")
    rfecv = RFECV(gradient_boosting,
                step=perc_step, 
                verbose=100,
                min_features_to_select=n_features_final, 
                cv=cv,
                n_jobs=cores, 
                scoring='neg_root_mean_squared_error')

    # Define the parameter grid for RandomizedSearchCV
    # This grid is based on the hyperparameters of the GradientBoostingRegressor
    # Adjust the values based on your specific needs and computational resources
    param_grid = {
        'n_estimators': [500, 750, 1000, 1250],  # Number of trees in the boosting ensemble
        'max_depth': [3, 5, 10, 20],  # Depth of the individual trees
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rate for boosting
        'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for fitting each tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required in each leaf
    }

    # Initialize RandomizedSearchCV with GradientBoostingRegressor
    # This will perform hyperparameter tuning using cross-validation
    # The scoring is set to 'neg_root_mean_squared_error' to minimize RMSE
    # n_iter specifies the number of different combinations to try (remember that the parameters are selected randomly)
    # refit=True means the best model will be refitted on the entire dataset
    logger.info("Initializing RandomizedSearchCV...")
    logger.info(f"Parameter grid size: {len(param_grid)} parameters")
    tuning = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=random_state), 
                                param_distributions=param_grid, 
                                cv=cv, 
                                scoring='neg_root_mean_squared_error',
                                verbose=3,
                                n_jobs=cores,
                                n_iter=5,
                                refit=True,
                                random_state=random_state)
    
    
    # Split the data into features (X) and target variable (y)
    X = interaction_data.drop(columns=['Score'])
    y = interaction_data['Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio_test, random_state=random_state)

    # Perform feature selection using RFECV and filter the training and test feature sets
    logger.info("Starting feature selection with RFECV...")
    selector = rfecv.fit(X_train, y_train)
    X_train_final = selector.transform(X_train)
    X_test_final = selector.transform(X_test)
    logger.info(f"Selected features: {X_train_final.shape[1]} out of {X_train.shape[1]}")


    # Create feature selection plot
    logger.info(f"Creating feature reduction plot: {feature_plot_path}")
    plt.figure(figsize=(10, 6))
    plt.plot(selector.cv_results_['n_features'], selector.cv_results_['mean_test_score'], marker='o', linestyle='-', color='b')
    plt.title('Model Performance vs Number of Features')
    plt.xlabel('Number of Features')
    plt.ylabel('Cross-validation Score (Neg RMSE)')
    plt.grid(True)
    plt.savefig(feature_plot_path)
    plt.close()
    logger.info("Feature reduction plot saved successfully")

    # Perform hyperparameter tuning using RandomizedSearchCV
    logger.info("Starting hyperparameter tuning...")
    tuning.fit(X_train_final, y_train)
    logger.info(f"Best parameters: {tuning.best_params_}")
    logger.info(f"Best cross-validation score: {tuning.best_score_}")

    # Get the best model from the tuning process
    best_xgb_model = tuning.best_estimator_

    # Attach feature names to the model
    feature_names = selector.feature_names_in_[selector.support_]
    best_xgb_model.feature_names_ = feature_names.tolist()

    # Predict on the test set
    logger.info("Making predictions...")
    y_pred = best_xgb_model.predict(X_test_final)
    y_pred_train = best_xgb_model.predict(X_train_final)

    # Create DataFrames for train and test results and save to CSV
    logger.info(f"Saving predictions to: {predictions_path}")
    train_results = pd.DataFrame({
        'observed': y_train, 
        'predicted': y_pred_train, 
        'dataset':"train"
    })  
    test_results = pd.DataFrame({
        'observed': y_test, 
        'predicted': y_pred , 
        'dataset':"test"
    }) 
    combined_results = pd.concat([train_results, test_results])
    combined_results.to_csv(predictions_path, index=True)
    logger.info("Predictions saved successfully")

    # Calculate RMSE (CURRENTLY NOT OUTPUTTED)
    rmse = np.sqrt(mean_squared_error(np.concatenate([y_pred, y_pred_train]), np.concatenate([y_test, y_train])))
    logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Feature importance extraction
    feature_importances = best_xgb_model.feature_importances_
    feature_names = selector.feature_names_in_[selector.support_]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })  
    importance_df.to_csv(feature_importance_path, index=False)
    logger.info("Feature importances saved successfully")

    # Save the best model to a file
    with open(model_path, 'wb') as f:
        pickle.dump(best_xgb_model, f)
    logger.info("Model saved successfully")

    # Filter features for bacteria and phage
    features_bacteria = [feature.replace('_bacteria', '') for feature in feature_names if feature.endswith('_bacteria')]
    features_phage = [feature.replace('_phage', '') for feature in feature_names if feature.endswith('_phage')]
    logger.info(f"Found {len(features_bacteria)} bacteria features and {len(features_phage)} phage features")


    with open(bacteria_features_path, 'w') as file: 
        for feature in features_bacteria:
            file.write(feature + '\n')
    with open(phage_features_path, 'w') as file: 
        for feature in features_phage:
            file.write(feature + '\n')

    logger.info("Pipeline completed successfully!")
    logger.info("=" * 50)
    logger.info("SUMMARY:")
    logger.info(f"  Input file: {args.input}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Selected features: {X_train_final.shape[1]}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  Best CV score: {tuning.best_score_:.4f}")
    logger.info("=" * 50)

if __name__ == "__main__":
    main()