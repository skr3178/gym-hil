#!/usr/bin/env python3
"""
Interactive script to work with the imported franka_sim_pick_lift_6 dataset.
This script loads the combined data and provides examples of data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_explore_data():
    """Load the combined data and explore its structure."""
    
    # Load the combined data
    print("Loading combined franka dataset...")
    data = pd.read_parquet("combined_franka_data.parquet")
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Total episodes: {data['episode_number'].nunique()}")
    print(f"Total timesteps: {len(data)}")
    
    return data

def analyze_dataset(data):
    """Perform basic analysis on the dataset."""
    
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    
    # Episode length statistics
    episode_lengths = data.groupby('episode_number').size()
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean: {episode_lengths.mean():.1f} timesteps")
    print(f"  Min: {episode_lengths.min()} timesteps")
    print(f"  Max: {episode_lengths.max()} timesteps")
    print(f"  Std: {episode_lengths.std():.1f} timesteps")
    
    # Reward analysis
    print(f"\nReward Statistics:")
    print(f"  Mean reward per timestep: {data['next.reward'].mean():.3f}")
    print(f"  Total reward across all episodes: {data['next.reward'].sum():.3f}")
    print(f"  Reward per episode:")
    episode_rewards = data.groupby('episode_number')['next.reward'].sum()
    print(f"    Mean: {episode_rewards.mean():.3f}")
    print(f"    Min: {episode_rewards.min():.3f}")
    print(f"    Max: {episode_rewards.max():.3f}")
    
    # Task distribution
    print(f"\nTask Distribution:")
    task_counts = data['task_index'].value_counts()
    for task, count in task_counts.items():
        print(f"  Task {task}: {count} timesteps")
    
    # State and action dimensions
    sample_state = data['observation.state'].iloc[0]
    sample_action = data['action'].iloc[0]
    print(f"\nData Dimensions:")
    print(f"  State dimension: {len(sample_state)}")
    print(f"  Action dimension: {len(sample_action)}")
    
    return episode_lengths, episode_rewards, task_counts

def examine_episode(data, episode_num):
    """Examine a specific episode in detail."""
    
    episode_data = data[data['episode_number'] == episode_num]
    if episode_data.empty:
        print(f"Episode {episode_num} not found.")
        return None
    
    print(f"\n" + "="*50)
    print(f"EPISODE {episode_num} DETAILS")
    print("="*50)
    
    print(f"Timesteps: {len(episode_data)}")
    print(f"Task index: {episode_data['task_index'].iloc[0]}")
    print(f"Total reward: {episode_data['next.reward'].sum():.3f}")
    print(f"Final done status: {episode_data['next.done'].iloc[-1]}")
    
    # Show first few timesteps
    print(f"\nFirst 5 timesteps:")
    for i in range(min(5, len(episode_data))):
        timestep = episode_data.iloc[i]
        print(f"  Step {i}: Reward={timestep['next.reward']:.3f}, "
              f"Done={timestep['next.done']}, "
              f"Frame={timestep['frame_index']}")
    
    return episode_data

def plot_episode_rewards(data, num_episodes=10):
    """Plot rewards for the first few episodes."""
    
    try:
        plt.figure(figsize=(12, 6))
        
        for episode in range(min(num_episodes, data['episode_number'].nunique())):
            episode_data = data[data['episode_number'] == episode]
            if not episode_data.empty:
                rewards = episode_data['next.reward'].cumsum()
                plt.plot(rewards, label=f'Episode {episode}', alpha=0.7)
        
        plt.xlabel('Timestep')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards Over Time for First 10 Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('episode_rewards.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as 'episode_rewards.png'")
        
        plt.show()
        
    except Exception as e:
        print(f"Could not create plot: {e}")
        print("Make sure matplotlib is available for plotting")

def main():
    """Main function to demonstrate data analysis."""
    
    print("Franka Sim Pick Lift 6 Dataset Explorer")
    print("="*50)
    
    # Load data
    data = load_and_explore_data()
    
    # Analyze dataset
    episode_lengths, episode_rewards, task_counts = analyze_dataset(data)
    
    # Examine specific episodes
    for episode_num in [0, 1, 2]:
        examine_episode(data, episode_num)
    
    # Create visualization
    plot_episode_rewards(data)
    
    print(f"\n" + "="*50)
    print("DATA EXPLORATION COMPLETED")
    print("="*50)
    print(f"You can now work with the 'data' DataFrame")
    print(f"Use data.head() to see the first few rows")
    print(f"Use data.columns to see all available columns")
    print(f"Use data.groupby('episode_number') to analyze by episode")
    
    return data

if __name__ == "__main__":
    # This will run the analysis and return the data
    data = main()
