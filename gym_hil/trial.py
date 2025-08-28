import time
import gymnasium as gym
import numpy as np

import gym_hil
from gym_hil import PassiveViewerWrapper

def main():
    """Run the gym_hil environment with clear observation display"""
    
    print("🚀 Starting gym_hil Panda Pick Cube Environment...")
    
    # Create the base environment (without render_mode="human" as it's not supported)
    base_env = gym.make("gym_hil/PandaPickCubeBase-v0", 
                        render_mode="rgb_array",  # Use rgb_array as base
                        image_obs=True)           # Enable image observations
    
    # Wrap it with the PassiveViewerWrapper to show it on screen
    env = PassiveViewerWrapper(base_env)
    
    print(f"✅ Environment created successfully!")
    print(f"📊 Action space: {env.action_space}")
    print(f"👁️  Observation space: {env.observation_space}")
    
    # Reset the environment to start
    obs, info = env.reset()
    print(f"\n🔄 Environment reset. Initial observation keys: {list(obs.keys())}")
    
    # Display initial observation
    display_observation(obs, "Initial", info)
    
    # Main loop - run for 1000 steps
    for step in range(1000):
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Display observations every 20 steps (to avoid spam)
        if step % 20 == 0:
            display_observation(observation, f"Step {step}", info)
            print(f"💰 Reward: {reward:.3f}")
            print(f"🏁 Episode status: Terminated={terminated}, Truncated={truncated}")
        
        # Check for human intervention
        if "is_intervention" in info and info["is_intervention"]:
            print(f"\n🚨 HUMAN INTERVENTION at step {step}!")
            print(f"   Action taken: {info.get('action_intervention', 'Unknown')}")
        
        # Reset if episode is done
        if terminated or truncated:
            print(f"\n🔄 Episode ended at step {step}. Resetting...")
            observation, info = env.reset()
            time.sleep(1)  # Pause to see the reset
        
        # Small delay to make it easier to follow
        time.sleep(0.1)
    
    print("\n✅ Training completed!")
    env.close()

def display_observation(obs, step_name, info):
    """Display the current observation in a readable format"""
    print(f"\n{'='*50}")
    print(f"📋 {step_name} Observation")
    print(f"{'='*50}")
    
    # Display state information
    if "state" in obs:
        state = obs['state']
        print(f"🤖 Robot State (first 5 values): {state[:5]}")
        print(f"   Full state shape: {state.shape}")
    
    if "environment_state" in obs:
        env_state = obs['environment_state']
        print(f"🌍 Environment State: {env_state}")
    
    # Display image information if available
    if "pixels" in obs:
        images = obs["pixels"]
        print(f"📷 Camera Views Available:")
        for view_name, img in images.items():
            print(f"   • {view_name}: {img.shape} (dtype: {img.dtype})")
    
    # Display any other observation keys
    other_keys = [k for k in obs.keys() if k not in ['state', 'environment_state', 'pixels']]
    if other_keys:
        print(f"🔍 Other observation keys: {other_keys}")
    
    # Display info
    if info:
        print(f"ℹ️  Info keys: {list(info.keys())}")

if __name__ == "__main__":
    main()
