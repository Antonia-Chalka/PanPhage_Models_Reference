# Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split,RepeatedKFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

interaction_data = pd.read_csv('./data/1.training/3.trainingdata.tsv', delimiter='\t')


random_state = 100  # for reproducibility
cores = -1  # -1 uses all cores
ratio_test = 0.25  # Train/Test ratios

# RepeatedKFold parameters
n_splits = 5
n_repeats = 2

# REF
perc_step = 1000 
n_features_first=2500
n_features_final = 300


gradient_boosting = GradientBoostingRegressor(n_estimators=1000, random_state=random_state)

cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

# Initialize RFECV with GradientBoostingRegressor
# This will perform recursive feature elimination with cross-validation
# The step parameter controls how many features to remove at each iteration
# The min_features_to_select parameter ensures that at least n_features_final features are selected
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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform feature selection using RFECV and filter the training and test feature sets
selector = rfecv.fit(X_train, y_train)
X_train_final = selector.transform(X_train)
X_test_final = selector.transform(X_test)

# Create feature selection plot
plt.figure(figsize=(10, 6))
plt.plot(selector.cv_results_['n_features'], selector.cv_results_['mean_test_score'], marker='o', linestyle='-', color='b')
plt.title('Model Performance vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Cross-validation Score (Neg RMSE)')
plt.grid(True)
plt.savefig('feature_reduction.png')

# Perform hyperparameter tuning using RandomizedSearchCV
tuning.fit(X_train_final, y_train)

# Get the best model from the tuning process
best_xgb_model = tuning.best_estimator_

# Attach feature names to the model
feature_names = selector.feature_names_in_[selector.support_]
best_xgb_model.feature_names_ = feature_names.tolist()

# Predict on the test set
y_pred = best_xgb_model.predict(X_test_final)
y_pred_train = best_xgb_model.predict(X_train_final)

# Create DataFrames for train and test results and save to CSV
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
combined_results.to_csv('predictions.csv', index=True)

# Calculate RMSE (CURRENTLY NOT OUTPUTTED)
rmse = np.sqrt(mean_squared_error(np.concatenate([y_pred, y_pred_train]), np.concatenate([y_test, y_train])))
# print(f"Root Mean Squared Error (RMSE): {rmse}")

# Feature importance extraction
feature_importances = best_xgb_model.feature_importances_
feature_names = selector.feature_names_in_[selector.support_]
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})  
importance_df.to_csv('feature_importances.csv', index=False)

# Save the best model to a file
with open('best_xgb_model.pkl', 'wb') as f:
    pickle.dump(best_xgb_model, f)

# Filter features for bacteria and phage
features_bacteria = [feature.replace('_bacteria', '') for feature in feature_names if feature.endswith('_bacteria')]
features_phage = [feature.replace('_phage', '') for feature in feature_names if feature.endswith('_phage')]

with open('feature_bacteria.txt', 'w') as file: 
    for feature in features_bacteria:
        file.write(feature + '\n')
with open('feature_phage.txt', 'w') as file: 
    for feature in features_phage:
        file.write(feature + '\n')