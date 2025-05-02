import pandas as pd
import os
import numpy as np

def preprocess_punjab_data(input_file='croppunjab.csv', output_file='x.csv'):
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
        
        # Filter to keep only Wheat, Rice, and Sugarcane crops
        print("Filtering data to keep only Wheat, Rice, and Sugarcane crops...")
        df = df[df['Crop'].isin(['Wheat', 'Rice', 'Sugarcane'])].copy()
        
        # Check if any data remains after crop filtering
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
