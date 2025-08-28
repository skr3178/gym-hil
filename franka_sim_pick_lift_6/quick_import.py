#!/usr/bin/env python3
"""
Quick script to import parquet files from franka_sim_pick_lift_6 dataset.
Simple and direct approach for immediate data access.
"""
#%%
import pandas as pd
import glob
import os
from pathlib import Path
#%%

 df = pd.read_parquet("/home/skr3178/gym-hil/franka_sim_pick_lift_6/data/chunk-000/episode_000000.parquet")
#%%
def quick_import_parquet():
    """Quickly import all parquet files and return combined dataset."""
    
    # Find all parquet files
    data_dir = "data/chunk-000"
    parquet_files = sorted(glob.glob(f"{data_dir}/*.parquet"))
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Load and combine all files
    all_data = []
    
    for file_path in parquet_files:
        print(f"Loading {file_path}...")
        df = pd.read_parquet(file_path)
        
        # Add episode number from filename
        episode_num = int(Path(file_path).stem.split('_')[-1])
        df['episode_number'] = episode_num
        df['episode_file'] = Path(file_path).name
        
        all_data.append(df)
        print(f"  Shape: {df.shape}")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Total episodes: {combined_df['episode_number'].nunique()}")
    print(f"Columns: {combined_df.columns.tolist()}")
    
    return combined_df

def show_sample_data(df, num_episodes=3):
    """Show sample data from the first few episodes."""
    print(f"\nSample data from first {num_episodes} episodes:")
    
    for episode in range(num_episodes):
        episode_data = df[df['episode_number'] == episode]
        if not episode_data.empty:
            print(f"\nEpisode {episode}:")
            print(f"  Timesteps: {len(episode_data)}")
            print(f"  Total reward: {episode_data['next.reward'].sum():.3f}")
            print(f"  Task index: {episode_data['task_index'].iloc[0]}")
            print(f"  State dim: {len(episode_data['observation.state'].iloc[0])}")
            print(f"  Action dim: {len(episode_data['action'].iloc[0])}")

if __name__ == "__main__":
    # Import the data
    data = quick_import_parquet()
    
    # Show sample information
    show_sample_data(data)
    
    # Save combined data
    output_file = "combined_franka_data.parquet"
    data.to_parquet(output_file)
    print(f"\nCombined data saved to {output_file}")
    
    print("\nData import completed!")
    print(f"You can now use 'data' variable to access the combined dataset")
