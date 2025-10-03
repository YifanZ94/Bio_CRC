import pandas as pd
import numpy as np

def merge_tissue_with_dtcru():
    """
    Merge 'tissue' column from 'tcr_obs_all' with 'DTCRU_extracted_features_96' 
    based on matching CDR3_Beta (DTCRU) and VDJ_1_cdr3_aa (tcr_obs_all)
    """
    
    print("Loading data files...")
    
    # Load the data files
    tcr_obs = pd.read_csv('tcr_obs_all.csv')
    dtcru_features = pd.read_csv('DTCRU_extracted_features_96.csv')
    
    print(f"tcr_obs_all shape: {tcr_obs.shape}")
    print(f"DTCRU_extracted_features_96 shape: {dtcru_features.shape}")
    
    # Display column names for verification
    print("\nColumns in tcr_obs_all:")
    print(tcr_obs.columns.tolist())
    print("\nColumns in DTCRU_extracted_features_96:")
    print(dtcru_features.columns.tolist())
    
    # Check for missing values in key columns
    print(f"\nMissing values in VDJ_1_cdr3_aa: {tcr_obs['VDJ_1_cdr3_aa'].isna().sum()}")
    print(f"Missing values in CDR3_Beta: {dtcru_features['CDR3_Beta'].isna().sum()}")
    print(f"Missing values in tissue: {tcr_obs['tissue'].isna().sum()}")
    
    # Remove rows with missing CDR3 sequences
    tcr_obs_clean = tcr_obs.dropna(subset=['VDJ_1_cdr3_aa'])
    dtcru_clean = dtcru_features.dropna(subset=['CDR3_Beta'])
    
    print(f"\nAfter removing missing CDR3 sequences:")
    print(f"tcr_obs_clean shape: {tcr_obs_clean.shape}")
    print(f"dtcru_clean shape: {dtcru_clean.shape}")
    
    # Perform the merge on CDR3 sequences
    print("\nPerforming merge...")
    merged_df = pd.merge(
        dtcru_clean, 
        tcr_obs_clean[['VDJ_1_cdr3_aa', 'tissue']], 
        left_on='CDR3_Beta', 
        right_on='VDJ_1_cdr3_aa', 
        how='left'
    )
    
    print(f"Merged dataframe shape: {merged_df.shape}")
    print(f"Rows with tissue information: {merged_df['tissue'].notna().sum()}")
    print(f"Rows without tissue information: {merged_df['tissue'].isna().sum()}")
    
    # Display tissue distribution
    if merged_df['tissue'].notna().any():
        print("\nTissue distribution in merged data:")
        print(merged_df['tissue'].value_counts())
    
    # Remove the duplicate CDR3 column (VDJ_1_cdr3_aa) since we already have CDR3_Beta
    merged_df = merged_df.drop('VDJ_1_cdr3_aa', axis=1)
    
    # Save the merged dataframe
    output_filename = 'DTCRU_features_with_tissue.csv'
    merged_df.to_csv(output_filename, index=False)
    print(f"\nMerged data saved to: {output_filename}")
    
    # Display summary statistics
    print(f"\nSummary:")
    print(f"Original DTCRU features: {dtcru_features.shape[0]} rows")
    print(f"Successfully merged with tissue: {merged_df['tissue'].notna().sum()} rows")
    print(f"Merge success rate: {merged_df['tissue'].notna().sum() / dtcru_features.shape[0] * 100:.2f}%")
    
    return merged_df

if __name__ == "__main__":
    merged_data = merge_tissue_with_dtcru()

