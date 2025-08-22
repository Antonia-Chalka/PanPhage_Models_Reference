from glob import glob
import pandas as pd
import pickle
import os
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Machine Learning Pipeline for Testing New Bacterial Isolates for Phage Interaction')
    
    # Input arguments
    parser.add_argument('--bacterial-matched-pangenome', '-b', 
                    default='./data/5.testing/<test_folder>/2.pangenome/2.panaroo_match',
                    help='Directory of matched bacterial pangenome files (default: ./data/5.testing/<test_folder>/2.pangenome/2.panaroo_match)')

    parser.add_argument('--phage-pangenome', '-p', 
                    default='./data/2.phage_pangenome/gene_presence_absence.Rtab',
                    help='Phage pangenome file (default: ./data/2.phage_pangenome/gene_presence_absence.Rtab)')
    
    parser.add_argument('--model', '-m', 
                    default='1.out/phage/best_xgb_model.pkl',
                    help='Model file (default: 1.out/phage/best_xgb_model.pkl)')
    
    # Output arguments
    parser.add_argument('--output-pangenome', '-op',
                    default='./data/5.testing/<test_folder>/4.phage_bacteria_pangenome.csv',
                    help='Output directory for phage-bacteria pangenome used for testing (default: ./data/5.testing/<test_folder>/4.phage_bacteria_pangenome.csv)')
    
    parser.add_argument('--output-predictions', '-o',
                    default='./data/5.testing/<test_folder>/5.predictions.csv',
                    help='Output directory for predictions (default: ./data/5.testing/<test_folder>/5.predictions.csv)')
    return parser.parse_args()



def merge_gene_presence_absence(directory):
    """Merge gene presence/absence matrices from bacterial isolates"""
    logging.info(f"Searching for bacterial pangenome files in: {directory}")
    
    # Find all matching CSV files in subdirectories
    file_paths = glob(os.path.join(directory, "**", "*_gene_presence_absence_matched.csv"), recursive=True)
    
    if not file_paths:
        logging.error(f"No matching files found in {directory}")
        raise FileNotFoundError("No matching files found.")
    
    logging.info(f"Found {len(file_paths)} bacterial pangenome files")
    
    # Read the first file to get the correct order of genes
    logging.debug(f"Reading base file: {file_paths[0]}")
    base_df = pd.read_csv(file_paths[0]).sort_values(by="Gene").reset_index(drop=True)
    all_data = [base_df]
    
    for file in file_paths[1:]:
        logging.debug(f"Processing file: {file}")
        df = pd.read_csv(file)
        df = df.sort_values(by="Gene").reset_index(drop=True)  # Sort to match the order
        
        if not df["Gene"].equals(base_df["Gene"]):
            logging.error(f"Gene column mismatch in {file}")
            raise ValueError(f"Gene column mismatch in {file}")
        
        all_data.append(df.drop(columns=["Gene"]))
    
    # Concatenate all data horizontally
    merged_df = pd.concat(all_data, axis=1)
    logging.info(f"Successfully merged bacterial pangenome data: {merged_df.shape}")
    
    return merged_df

def load_phage_pangenome(phage_path):
    """Load and process phage pangenome data"""
    logging.info(f"Loading phage pangenome from: {phage_path}")
    
    if not os.path.exists(phage_path):
        logging.error(f"Phage pangenome file not found: {phage_path}")
        raise FileNotFoundError(f"Phage pangenome file not found: {phage_path}")
    
    phage_df = pd.read_table(phage_path)
    logging.info(f"Loaded phage pangenome data: {phage_df.shape}")
    
    return phage_df

def create_combined_matrix(bacteria_matrix, phage_matrix):
    """Create Cartesian product of bacteria and phage matrices"""
    logging.info("Creating combined bacteria-phage matrix...")
    
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
    
    logging.info(f"Combined matrix created: {combined.shape}")
    return combined


def make_predictions(model_path, combined_data):
    """Load model and make predictions"""
    logging.info(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        best_xgb_model = pickle.load(f)
    
    logging.info("Model loaded successfully")
    
    # Load and align data to model features
    
    # Check if all required features are present
    missing_features = set(best_xgb_model.feature_names_) - set(combined_data.columns)
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
    
    combined_data = combined_data[best_xgb_model.feature_names_]
    logging.info(f"Data aligned to model features: {combined_data.shape}")
    
    # Predict using model
    logging.info("Making predictions...")
    predictions = best_xgb_model.predict(combined_data)
    
    # Format predictions into a DataFrame
    predictions_df = pd.DataFrame({
        'Bacteria_Phage': combined_data.index,
        'Predicted_Score': predictions
    })
    
    logging.info(f"Predictions completed: {len(predictions)} predictions made")
    return predictions_df

def main():
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(args.output_pangenome), exist_ok=True)
        os.makedirs(os.path.dirname(args.output_predictions), exist_ok=True)
        
        # Step 1: Merge gene presence/absence matrices from bacterial isolates
        logging.info("Step 1: Processing bacterial pangenome data")
        merged_df = merge_gene_presence_absence(args.bacterial_matched_pangenome)
        
        # Convert gene presence to binary (1 if present, 0 if NaN)
        merged_df.iloc[:, 1:] = merged_df.iloc[:, 1:].notna().astype(int)
        
        # Transform from wide to long format, then pivot to isolate-gene matrix
        bacteria_long = merged_df.melt(id_vars=["Gene"], var_name="Isolate", value_name="Value")
        bacteria_matrix = bacteria_long.pivot(index="Isolate", columns="Gene", values="Value")
        bacteria_matrix.columns = [f"{col}_bacteria" for col in bacteria_matrix.columns]
        
        logging.info(f"Bacterial matrix created: {bacteria_matrix.shape}")
        
        # Step 2: Load phage presence/absence matrix and reshape
        logging.info("Step 2: Processing phage pangenome data")
        phage_df = load_phage_pangenome(args.phage_pangenome, logging)
        
        phage_long = phage_df.melt(id_vars=["Gene"], var_name="Phage", value_name="Value")
        phage_matrix = phage_long.pivot(index="Phage", columns="Gene", values="Value")
        phage_matrix.columns = [f"{col}_phage" for col in phage_matrix.columns]
        
        logging.info(f"Phage matrix created: {phage_matrix.shape}")
        
        # Step 3: Create Cartesian product (every Bacteria x Phage pair)
        logging.info("Step 3: Creating combined bacteria-phage matrix")
        combined = create_combined_matrix(bacteria_matrix, phage_matrix)
        
        # Save the combined pangenome dataset
        logging.info(f"Saving combined pangenome to: {args.output_pangenome}")
        combined.to_csv(args.output_pangenome)
        
        # Step 4: Load trained model and make predictions
        logging.info("Step 4: Making predictions")
        predictions_df = make_predictions(args.model, combined)
        
        # Extract isolate and phage names back from combined index
        predictions_df[['Isolate', 'Phage']] = predictions_df['Bacteria_Phage'].str.extract(r"(.+)_([^_]+)$")
        predictions_df = predictions_df.drop(columns=['Bacteria_Phage'])
        
        # Pivot to Isolate x Phage matrix of predicted scores
        predictions_matrix = predictions_df.pivot(index="Isolate", columns="Phage", values="Predicted_Score")
        
        # Save predictions
        logging.info(f"Saving predictions to: {args.output_predictions}")
        predictions_matrix.to_csv(args.output_predictions, index=True)
        
        logging.info("Pipeline completed successfully!")
    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
