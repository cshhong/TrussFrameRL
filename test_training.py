'''
DONE integrate trussframeASAP
TODO adjust observation space and action space to match PPO solver
TODO edit hong_ppo main to save h5 files 
'''
import sys
sys.path.append("/Users/chong/Dropbox/2024Fall/TrussframeASAP-RL/")  # ensure Python can locate the TrussFrameASAP module 
import gymnasium as gym
import TrussFrameASAP.gymenv  # Explicitly import gymenv to execute __init__.py and register gym environments; only have to do once!
import os

from cleanrl.cleanrl.hong_ppo_cantilever import Args, main
# from cleanrl.cleanrl.hong_ppo import Args, main
from gymnasium.vector import SyncVectorEnv

if __name__ == "__main__":

    # test cantilever env 
    # name = 'test_dependency_refactor'
    # render_mode = None
    # if render_mode == "rgb_list":
    #     render_dir = os.path.join("./videos/", name) # render_mode == "rgb_list"
    # elif render_mode == "rgb_end":
    #     render_dir = os.path.join("render/",name) # render_mode != "rgb_end"
    # else:
    #     render_dir = os.path.join("render/",name) # not used but required
    # env = gym.make("Cantilever-v0", render_mode=render_mode, render_dir=render_dir)

    # env.reset()

    # env.close()

    args = Args()
    # Modify args as needed, for example:
    args.env_id = "Cantilever-v0"
    args.total_timesteps = 5e6  # Shorter run for testing
    args.track = True
    args.num_envs = 1
    args.wandb_project_name = "cleanrl-test-cantilever-ppo_dec6"
    args.ent_coef = 0.1
    args.exp_name = "noinvalidpenalty_disp_inventoryleft_medonly_entcoef0.1"
    
    main(args) # TODO follow logging hdf5 and logic of term_eps_idx of random_rollout but trained!

    # Test single environment
    # Import the environment
    # render_mode = None
    # render_dir = "render"
    # env = gym.make("Cantilever-v0", render_mode=render_mode, render_dir=render_dir)

    # # Reset the environment
    # obs, info = env.reset()
    # print(f"Initial observation: {obs}")
    # print(f"Initial info: {info}")

    # # Simulate multiple steps
    # max_steps = 10  # Maximum number of steps to simulate
    # for step in range(max_steps):
    #     # Sample a random action
    #     action = env.action_space.sample()
        
    #     # Step the environment
    #     obs, reward, terminated, truncated, info = env.step(action)
        
    #     # Print results for the current step
    #     print(f"Step {step + 1}:")
    #     print(f"  Action: {action}")
    #     print(f"  Observation: {obs}")
    #     print(f"  Reward: {reward}")
    #     print(f"  Terminated: {terminated}")
    #     print(f"  Truncated: {truncated}")
    #     print(f"  Info: {info}")
        
    #     # Check if the episode has ended
    #     if terminated or truncated:
    #         print("Episode ended.")
    #         break

    # # Close the environment
    # env.close()

    # Test parallel environments
    # Create the environment factories for vectorization
    # render_mode = None
    # render_dir = "render"
    # env_fns = [
    #     lambda: gym.make("Cantilever-v0", render_mode=render_mode, render_dir=render_dir)
    #     for _ in range(4)
    # ]

    # # Create the vectorized environment
    # envs = SyncVectorEnv(env_fns)

    # obs, info = envs.reset()
    
    # # Simulate multiple steps
    # max_steps = 10  # Maximum number of steps to simulate
    # for step in range(max_steps):
    #     # Sample random actions for all environments
    #     actions = [envs.action_space.sample() for _ in range(envs.num_envs)]
    #     obs, rewards, terminateds, truncateds, infos = envs.step(actions)

    #     # Print results for the current step
    #     print(f"Step {step + 1}:")
    #     print(f"  Observations: {obs}")
    #     print(f"  Rewards: {rewards}")
    #     print(f"  Terminateds: {terminateds}")
    #     print(f"  Truncateds: {truncateds}")
    #     print(f"  Infos: {infos}")

    #     # Check if all environments are done
    #     if all(terminateds) or all(truncateds):
    #         print("All environments terminated or truncated.")
    #         break

    # # Close the environments
    # envs.close()