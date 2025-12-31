import pandas as pd
import numpy as np

def clean_house_data(file_path):
    print(f"--- Cleaning Data: {file_path} ---")
    df = pd.read_csv(file_path)
    initial_count = len(df)

    # 1. Check for duplicates
    df = df.drop_duplicates()
    if len(df) < initial_count:
        print(f"Removed {initial_count - len(df)} duplicate rows.")

    # 2. Outlier Removal using IQR (Interquartile Range)
    # We focus on 'price' and 'area' as they impact regression the most
    for column in ['price', 'area']:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filter the data
        before_count = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        print(f"Removed {before_count - len(df)} outliers from '{column}' (Range: {lower_bound:,.2f} - {upper_bound:,.2f})")

    # 3. Final stats
    print(f"Final dataset size: {len(df)} (Reduced from {initial_count})")
    
    # Save cleaned data
    cleaned_file = 'house_prices_cleaned.csv'
    df.to_csv(cleaned_file, index=False)
    print(f"Cleaned data saved to {cleaned_file}")
    return cleaned_file

if __name__ == "__main__":
    clean_house_data('house_prices.csv')
