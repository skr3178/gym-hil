#!/usr/bin/env python3
"""
Script to import and analyze parquet files from franka_sim_pick_lift_6 dataset.
This script loads all episode data and provides various analysis functions.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import json

class FrankaSimDataImporter:
    def __init__(self, data_dir="data/chunk-000"):
        """
        Initialize the importer with the data directory path.
        
        Args:
            data_dir (str): Path to the directory containing parquet files
        """
        self.data_dir = Path(data_dir)
        self.episode_files = []
        self.episodes_data = []
        self.combined_data = None
        
    def discover_episode_files(self):
        """Find all parquet files in the data directory."""
        pattern = str(self.data_dir / "*.parquet")
        self.episode_files = sorted(glob.glob(pattern))
        print(f"Found {len(self.episode_files)} episode files")
        return self.episode_files
    
    def load_single_episode(self, file_path):
        """Load a single episode parquet file."""
        try:
            df = pd.read_parquet(file_path)
            episode_num = int(Path(file_path).stem.split('_')[-1])
            df['episode_file'] = Path(file_path).name
            df['episode_number'] = episode_num
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def load_all_episodes(self):
        """Load all episode files and combine them."""
        if not self.episode_files:
            self.discover_episode_files()
        
        print("Loading episodes...")
        for file_path in self.episode_files:
            df = self.load_single_episode(file_path)
            if df is not None:
                self.episodes_data.append(df)
                print(f"Loaded {file_path} - Shape: {df.shape}")
        
        if self.episodes_data:
            self.combined_data = pd.concat(self.episodes_data, ignore_index=True)
            print(f"\nCombined dataset shape: {self.combined_data.shape}")
            print(f"Total episodes: {self.combined_data['episode_number'].nunique()}")
        
        return self.combined_data
    
    def get_dataset_info(self):
        """Get basic information about the dataset."""
        if self.combined_data is None:
            print("No data loaded. Call load_all_episodes() first.")
            return None
        
        info = {
            'total_episodes': self.combined_data['episode_number'].nunique(),
            'total_timesteps': len(self.combined_data),
            'columns': self.combined_data.columns.tolist(),
            'episode_lengths': self.combined_data.groupby('episode_number').size().describe(),
            'task_distribution': self.combined_data['task_index'].value_counts().to_dict(),
            'reward_stats': self.combined_data['next.reward'].describe().to_dict(),
            'done_distribution': self.combined_data['next.done'].value_counts().to_dict()
        }
        
        return info
    
    def analyze_episode(self, episode_number):
        """Analyze a specific episode."""
        if self.combined_data is None:
            print("No data loaded. Call load_all_episodes() first.")
            return None
        
        episode_data = self.combined_data[self.combined_data['episode_number'] == episode_number]
        if episode_data.empty:
            print(f"Episode {episode_number} not found.")
            return None
        
        analysis = {
            'episode_number': episode_number,
            'timesteps': len(episode_data),
            'total_reward': episode_data['next.reward'].sum(),
            'final_reward': episode_data['next.reward'].iloc[-1],
            'done': episode_data['next.done'].iloc[-1],
            'task_index': episode_data['task_index'].iloc[0],
            'state_dimension': len(episode_data['observation.state'].iloc[0]) if episode_data['observation.state'].iloc[0] else 0,
            'action_dimension': len(episode_data['action'].iloc[0]) if episode_data['action'].iloc[0] else 0
        }
        
        return analysis
    
    def get_episode_data(self, episode_number):
        """Get the raw data for a specific episode."""
        if self.combined_data is None:
            print("No data loaded. Call load_all_episodes() first.")
            return None
        
        return self.combined_data[self.combined_data['episode_number'] == episode_number]
    
    def save_combined_data(self, output_path="combined_franka_data.parquet"):
        """Save the combined dataset to a single parquet file."""
        if self.combined_data is None:
            print("No data loaded. Call load_all_episodes() first.")
            return False
        
        try:
            self.combined_data.to_parquet(output_path)
            print(f"Combined data saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def export_to_csv(self, output_dir="exported_csv"):
        """Export each episode to a separate CSV file."""
        if self.combined_data is None:
            print("No data loaded. Call load_all_episodes() first.")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        for episode_num in self.combined_data['episode_number'].unique():
            episode_data = self.get_episode_data(episode_num)
            if episode_data is not None:
                csv_path = os.path.join(output_dir, f"episode_{episode_num:06d}.csv")
                episode_data.to_csv(csv_path, index=False)
                print(f"Exported episode {episode_num} to {csv_path}")
        
        return True

def main():
    """Main function to demonstrate the importer."""
    print("Franka Sim Pick Lift 6 Dataset Importer")
    print("=" * 50)
    
    # Initialize importer
    importer = FrankaSimDataImporter()
    
    # Load all episodes
    combined_data = importer.load_all_episodes()
    
    if combined_data is not None:
        # Get dataset information
        info = importer.get_dataset_info()
        print("\nDataset Information:")
        print(f"Total Episodes: {info['total_episodes']}")
        print(f"Total Timesteps: {info['total_timesteps']}")
        print(f"Columns: {info['columns']}")
        
        # Analyze first few episodes
        print("\nEpisode Analysis:")
        for i in range(min(5, info['total_episodes'])):
            analysis = importer.analyze_episode(i)
            if analysis:
                print(f"Episode {i}: {analysis['timesteps']} timesteps, "
                      f"Total reward: {analysis['total_reward']:.3f}, "
                      f"Task: {analysis['task_index']}")
        
        # Save combined data
        importer.save_combined_data()
        
        print("\nData import completed successfully!")
        
        return combined_data
    
    return None

if __name__ == "__main__":
    data = main()
