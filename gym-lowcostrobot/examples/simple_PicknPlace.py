import gymnasium as gym
import gym_lowcostrobot # Import the low-cost robot environments

# Create the environment
env = gym.make("PickPlaceCube-v0", render_mode="human")

# Reset the environment
observation, info = env.reset()

for _ in range(1000):
    # Sample random action
    action = env.action_space.sample()

    # Step the environment
    observation, reward, terminted, truncated, info = env.step(action)

    # Reset the environment if it's done
    if terminted or truncated:
        observation, info = env.reset()

# Close the environment
env.close()