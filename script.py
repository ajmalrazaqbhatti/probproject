import pandas as pd
import os

def filter_rice_data(input_file='cropdata.csv', output_file='crop_punjab.csv'):
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            return False
        
        # Read the CSV file
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Check if 'Crop' column exists
        if 'State' not in df.columns:
            print("Error: 'Crop' column not found in the CSV file.")
            return False
        
        # Filter rows where Crop is "Rice"
        print("Filtering data for Rice crops...")
        rice_df = df[df['State'] == 'Punjab']
        
        # Check if any rice data was found
        if rice_df.empty:
            print("No rows found with Crop as 'Rice'.")
            return False
        
        # Save filtered data to a new CSV file
        print(f"Saving filtered data to {output_file}...")
        rice_df.to_csv(output_file, index=False)
        
        print(f"Successfully saved {len(rice_df)} rows of Rice crop data to {output_file}")
        return True
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    filter_rice_data()
