from glob import glob
import pandas as pd
import pickle
import os



def merge_gene_presence_absence(directory):
    # Find all matching CSV files in subdirectories
    file_paths = glob(os.path.join(directory, "**", "*_gene_presence_absence_matched.csv"), recursive=True)
    
    if not file_paths:
        raise FileNotFoundError("No matching files found.")
    
    # Read the first file to get the correct order of genes
    base_df = pd.read_csv(file_paths[0]).sort_values(by="Gene").reset_index(drop=True)
    all_data = [base_df]
    
    for file in file_paths[1:]:
        df = pd.read_csv(file)
        df = df.sort_values(by="Gene").reset_index(drop=True)  # Sort to match the order
        
        if not df["Gene"].equals(base_df["Gene"]):
            raise ValueError(f"Gene column mismatch in {file}")
        
        all_data.append(df.drop(columns=["Gene"]))
    
    # Concatenate all data horizontally
    merged_df = pd.concat(all_data, axis=1)
    return merged_df

# === Step 1: Merge gene presence/absence matrices from bacterial isolates ===
bacteria_dir = "./data/5.testing/<test_folder>/2.pangenome/2.panaroo_match"
merged_df = merge_gene_presence_absence(bacteria_dir)


# Convert gene presence to binary (1 if present, 0 if NaN)
merged_df.iloc[:, 1:] = merged_df.iloc[:, 1:].notna().astype(int)


# Transform from wide to long format, then pivot to isolate-gene matrix
bacteria_long = merged_df.melt(id_vars=["Gene"], var_name="Isolate", value_name="Value")
bacteria_matrix = bacteria_long.pivot(index="Isolate", columns="Gene", values="Value")
bacteria_matrix.columns = [f"{col}_bacteria" for col in bacteria_matrix.columns]



# === Step 2: Load phage presence/absence matrix and reshape ===

phage_path = "./data/2.phage_pangenome/gene_presence_absence.Rtab"
phage_df = pd.read_table(phage_path)

phage_long = phage_df.melt(id_vars=["Gene"], var_name="Phage", value_name="Value")
phage_matrix = phage_long.pivot(index="Phage", columns="Gene", values="Value")
phage_matrix.columns = [f"{col}_phage" for col in phage_matrix.columns]

# === Step 3: Create Cartesian product (every Bacteria x Phage pair) ===

# Reset index and add dummy key for cross join
bacteria_reset = bacteria_matrix.reset_index()
phage_reset = phage_matrix.reset_index()
bacteria_reset['key'] = 1
phage_reset['key'] = 1

# Cross join using dummy key
combined = pd.merge(bacteria_reset, phage_reset, on='key').drop(columns=['key'])

# Combine isolate and phage names for row index
combined['Bacteria_Phage'] = combined['Isolate'] + "_" + combined['Phage']
combined = combined.drop(columns=['Isolate', 'Phage']).set_index('Bacteria_Phage')

# Save the combined pangenome dataset
combined.to_csv("./data/5.testing/<test_folder>/4.phage_bacteria_pangenome.csv")

# === Step 4: Load trained model and make predictions ===
# Load model from pickle
with open('1.out/phage/best_xgb_model.pkl', 'rb') as f:
    best_xgb_model = pickle.load(f)

    # Load and align data to model features
    new_data = pd.read_csv("./data/5.testing/<test_folder>/4.phage_bacteria_pangenome.csv", index_col="Bacteria_Phage")
    new_data = new_data[best_xgb_model.feature_names_]

    # Predict using model
    predictions = best_xgb_model.predict(new_data)

    # Format predictions into a DataFrame
    predictions_df = pd.DataFrame({
        'Bacteria_Phage': new_data.index,
        'Predicted_Score': predictions
    })

# Extract isolate and phage names back from combined index
predictions_df[['Isolate', 'Phage']] = predictions_df['Bacteria_Phage'].str.extract(r"(.+)_([^_]+)$")
predictions_df = predictions_df.drop(columns=['Bacteria_Phage'])

# Pivot to Isolate x Phage matrix of predicted scores
predictions_matrix = predictions_df.pivot(index="Isolate", columns="Phage", values="Predicted_Score")

# Save predictions
predictions_matrix.to_csv('./data/5.testing/<test_folder>/5.predictions.csv', index=True)
