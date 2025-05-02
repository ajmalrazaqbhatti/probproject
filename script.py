import pandas as pd
import os
import numpy as np

def preprocess_punjab_data(input_file='APY.csv', output_file='x.csv'):
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            return False
        
        # Read the CSV file
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Check if required columns exist
        required_columns = ['Crop','Crop_Year','Area','Production', 'Yield']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: The following required columns are missing: {', '.join(missing_columns)}")
            return False
        
        # Filter to keep only Wheat and Rice crops in Punjab
        print("Filtering data to keep only Wheat and Rice crops in Punjab...")
        
        # Check if State column exists
        if 'State' in df.columns:
            df = df[(df['Crop'].isin(['Wheat', 'Rice'])) & (df['State'] == 'Punjab')].copy()
        else:
            print("Warning: 'State' column not found. Filtering only by crop type.")
            df = df[df['Crop'].isin(['Wheat', 'Rice'])].copy()
        
        # Check if any data remains after filtering
        if df.empty:
            print("No data found for Wheat, Rice, or Sugarcane crops.")
            return False
        
        # Save filtered data directly without aggregation or recalculation
        print(f"Saving filtered data to {output_file}...")
        df.to_csv(output_file, index=False)
        
        print(f"Successfully saved {len(df)} rows of filtered crop data to {output_file}")
        return True
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    preprocess_punjab_data()
