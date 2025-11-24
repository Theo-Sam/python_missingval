import pandas as pd
import numpy as np
import os

def main():
    print("Starting verification script...")

    # 1. Load the dataset
    file_path = 'sensor_log.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
    print(f"Shape: {df.shape}")
    
    # Display missing values
    print("\nMissing values per column:")
    print(df.isna().sum())

    # Create cleaned_data directory
    output_dir = 'cleaned_data'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")

    # 2. Implement cleaning methods
    
    # Method 1: Backfill
    print("\n--- Method 1: Backfill ---")
    df_backfill = df.bfill()
    print("Missing values after backfill:")
    print(df_backfill.isna().sum())
    save_path = os.path.join(output_dir, 'sensor_log_backfill.csv')
    df_backfill.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
    print("Summary Statistics:")
    print(df_backfill.describe())

    # Method 2: Forward fill
    print("\n--- Method 2: Forward fill ---")
    df_ffill = df.ffill()
    print("Missing values after forward fill:")
    print(df_ffill.isna().sum())
    save_path = os.path.join(output_dir, 'sensor_log_ffill.csv')
    df_ffill.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
    print("Summary Statistics:")
    print(df_ffill.describe())

    # Method 3: Interpolation
    print("\n--- Method 3: Interpolation ---")
    # Interpolation requires numeric types mostly, or time index. 
    # The dataset has a timestamp column, let's try to use it if possible, 
    # but standard interpolate works on index too.
    # Let's ensure we are interpolating numeric columns.
    df_interpolate = df.copy()
    # We can set timestamp as index for better time-based interpolation if needed, 
    # but for this exercise linear interpolation on default index is likely expected unless specified.
    # However, the tutorial mentioned "Interpolation: smoothly estimate missing values between known points."
    # and "method='time'" in the tutorial. Let's try to follow the tutorial's lead if possible,
    # but for simplicity and robustness with the raw CSV, linear is a good start.
    # Let's check if we can convert timestamp.
    try:
        df_interpolate['timestamp'] = pd.to_datetime(df_interpolate['timestamp'])
        df_interpolate = df_interpolate.set_index('timestamp')
        df_interpolate = df_interpolate.interpolate(method='time')
        df_interpolate = df_interpolate.reset_index() # Reset index to save properly
    except Exception as e:
        print(f"Time interpolation failed ({e}), falling back to linear.")
        df_interpolate = df.interpolate()

    print("Missing values after interpolation:")
    print(df_interpolate.isna().sum())
    save_path = os.path.join(output_dir, 'sensor_log_interpolate.csv')
    df_interpolate.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
    print("Summary Statistics:")
    print(df_interpolate.describe())

    # Method 4: Mean Imputation
    print("\n--- Method 4: Mean Imputation ---")
    df_mean = df.copy()
    # Select numeric columns
    numeric_cols = df_mean.select_dtypes(include='number').columns
    for col in numeric_cols:
        df_mean[col] = df_mean[col].fillna(df_mean[col].mean())
    
    print("Missing values after mean imputation:")
    print(df_mean.isna().sum())
    save_path = os.path.join(output_dir, 'sensor_log_mean.csv')
    df_mean.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")
    print("Summary Statistics:")
    print(df_mean.describe())

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
