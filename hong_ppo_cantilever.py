'''
Modify ppo.py to 
- call from other script (create def main)
'''
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass, field
from typing import List

import gymnasium as gym # version 0.28.1 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from gymnasium.utils.save_video import save_video

from libs.TrussFrameASAP.PerformanceMap.h5_utils import *

from tqdm import tqdm

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")

from gymnasium.wrappers import FrameStack

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track_wb: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    # capture_video: bool = False
    # """whether to capture videos of the agent performances (check out `videos` folder)"""
    render_mode: str = None #  [None, "debug_all", "debug_valid", "rgb_list", "debug_end", "rgb_end", "rgb_end_interval"]
    render_dir: str = "render"
    # Additional arguments (Hong)
    """save h5 file used for plotting all terminated episodes"""
    save_h5: bool = True
    render_interval: int = 1000 # interval (num eps) to save render for render mode "rgb_end_interval" and "rgb_list"
    render_interval_count : int = 10 # number of consecutive renders to save for render mode "rgb_list" at each interval 
    # Algorithm specific arguments
    env_id: str = "Cantilever-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps_rollout: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2 
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.1 # default 0.01 For sparse reward environments, higher values (e.g., 0.05–0.1) are beneficial to encourage exploration.
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Additional algorithm specific arguments (Hong)
    """number of random actions to take at start of env that are not registered as trajectory"""
    rand_init_steps: int = 0
    rand_init_seed: int = None

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0 # rounds of training the policy
    """the number of iterations (computed in runtime)""" 

    # train mode
    train_mode = "train" # "train" or "inference"

    # load model to resume training / inference
    load_checkpoint: bool = False
    load_checkpoint_path: str = None

    # save model during training
    save_checkpoint: bool = True
    checkpoint_interval_steps: int = 0
    """interval in saving model (computed in runtime)""" 

    # epsilon greedy action selection
    epsilon_greedy: float = 1e-3

    # observation mode
    obs_mode: str = 'frame_grid' # 'frame_grid_singleint'

    num_stacked_obs: int = 3 # for frame_grid obs mode

    # boundary conditions
    bc_height_options: list = field(default_factory=list)# List[int] = [1, 2]
    bc_length_options: list = field(default_factory=list)#List[int] = [3, 4, 5]
    bc_loadmag_options: list = field(default_factory=list)#List[int] = [300, 400, 500]
    bc_inventory_options: list = field(default_factory=list) #List[tuple] = [(10,10), (10,5), (5,5), (8,3)]

    # Grid size
    frame_grid_size_x: int = 10
    frame_grid_size_y: int = 5
    frame_size: int = 2

    collect_complete: bool = False # collect complete trajectories in buffer with epsilon greedy
    collect_complete_epsilon: float = 0.1 # epsilon greedy for complete trajectories

    num_target_loads: int = 1 # number of target loads to place in the frame

    condition_dim: int = 0 # dimension of condition vector for actor, critic

    bc_fixed = None # fixed boundary condition (optional)

    elem_sections: list = field(default_factory=list) # List[tuple] = [(0.1, 0.1), (0.1, 0.2)]

    high_util_percentage : int = 25 # percentage of high utilization criteria

def layer_init(layer, std=np.sqrt(1.0), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def get_grad_norm(agent):
    total_norm = 0
    for param in agent.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(2).item()
            total_norm += param_norm ** 2
            # print(f"Layer: {param.shape} | Grad Norm: {param_norm:.4f}")

    total_norm = total_norm ** 0.5
    # print(f"Total Gradient Norm: {total_norm:.4f}") 

    return total_norm


class Agent(nn.Module):
    def __init__(self, envs):
        '''
        envs : Gymnasium.vector.VectorEnv object (https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv)
        '''
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, 
                             x, 
                             fixed_action=None, 
                             action_mask=None, 
                             epsilon_greedy=1e-2,
                             ):
        '''
        get next action based on actor network output
        x : next_obs from previous step (batch_size, obs_space)
        action_mask : np.ndarray of shape (n,) and dtype np.int8 where 1 represents valid actions and 0 invalid / infeasible

        * actions are selected with action mask, but log probabilities are calculated without action mask
        return : action, log_prob, entropy, value
        # TODO what are the probs of unused cells? how does it affect overall logit distribution and approxmiate kl divergence?
        '''
        logits = self.actor(x) # (action_space, )
        org_probs = Categorical(logits=logits)

        # Action selection 
        if fixed_action == None:
            # Epsilon greedy action selection
            if np.random.rand() < epsilon_greedy:
                if action_mask is not None:
                    # print(f'    epsilon_greedy random action with action mask')
                    valid_actions = np.where(action_mask == 1)[0]
                    action = torch.tensor(np.random.choice(valid_actions))
                else:
                    # print(f'    action mask is None (epsilon_greedy random action)')
                    action = torch.tensor(np.random.randint(len(org_probs.probs)))
            else:
                if action_mask is not None: # mask invalid actions to get next action
                    masked_logits = logits.clone()  # Clone logits to avoid modifying the original tensor
                    # Ensure action_mask has the correct shape
                    if len(action_mask.shape) == 1:
                        action_mask = torch.tensor(action_mask).unsqueeze(0)  # Add batch dimension to action_mask if needed
                    masked_logits[action_mask == 0] = -float('inf')  # Set logits of undesirable actions to -inf
                    masked_probs = Categorical(logits=masked_logits) # masked probs

                    action = masked_probs.sample()
                    
                else: # sample action without mask
                    print(f'action mask is None, taking action according to policy') 
                    action = org_probs.sample()
        else:
            action = fixed_action

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)

        return action, org_probs.log_prob(action), org_probs.entropy(), self.critic(x)
    


def normalize_frame_grid(frame_grid):
    """
    Standardizes a frame grid so that the mean is 0 and the standard deviation is 1.
    
    - load frame (-1) → replaced with 6
    - Standardization: (X - mean) / std
    
    Args:
        frame_grid (np.ndarray): The input grid with values in the range [-1, 5].

    Returns:
        torch.Tensor: The standardized grid with mean 0 and std 1.
    """
    # Replace -1 (load frame) with 6
    frame_grid = np.where(frame_grid == -1, 6, frame_grid)

    # Convert to tensor
    frame_grid = torch.tensor(frame_grid, dtype=torch.float32)

    # Compute mean and std
    mean = frame_grid.mean()
    std = frame_grid.std()

    # Apply standardization
    standardized_grid = (frame_grid - mean) / (std + 1e-8)  # Small epsilon to avoid division by zero

    return standardized_grid

def conv2d_output_shape(h_in, w_in, kernel_size, stride=1, padding=0):
    # integer division for PyTorch
    h_out = (h_in + 2 * padding - kernel_size) // stride + 1
    w_out = (w_in + 2 * padding - kernel_size) // stride + 1
    return h_out, w_out


class Agent_CNN(nn.Module):
    def __init__(self, envs, num_stacked_obs, condition_dim=0):
        super().__init__()
        H_in, W_in = envs.single_observation_space.shape # (15,7)
        h1, w1 = conv2d_output_shape(H_in, W_in, kernel_size=3, stride=1, padding=1) # 1) First conv
        h2, w2 = conv2d_output_shape(h1, w1, kernel_size=3, stride=2, padding=1)  # 2) Second conv
        h3, w3 = conv2d_output_shape(h2, w2, kernel_size=3, stride=1, padding=1) # 3) Third conv shape

        self.condition_dim = condition_dim

        self.network = nn.Sequential(
            # nn.Conv2d(num_stacked_obs, 32, kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm([32, h1, w1]),
            nn.Conv2d(num_stacked_obs, 64, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([64, h1, w1]),
            nn.ReLU(),

            # nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # nn.LayerNorm([64, h2, w2]),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([128, h2, w2]),
            nn.ReLU(),

            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.LayerNorm([64, h3, w3]),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([128, h3, w3]),
            nn.ReLU(),

            nn.Flatten(),
            # nn.Linear(64 * h3 * w3, 512),
            nn.Linear(128 * h3 * w3, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

        )
        print(f'Agent_CNN summary : {self.network}')

        if self.condition_dim == 0:
            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)
        else: # actor, critic conditioned on target (length, height, loadmag) and inventory (light, medium)
            self.actor = layer_init(nn.Linear(512 + condition_dim, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512 + condition_dim, 1), std=1)

        print(f'Actor : {self.actor}')
        print(f'Critic : {self.critic}')


        # # Network 2 bigger network
        # self.network = nn.Sequential(
        #     layer_init(nn.Conv2d(num_stacked_obs, 32, kernel_size=3, stride=1, padding=1)), # (3, 10, 7) -> (32, 10, 7),
        #     # nn.LayerNorm([32, 10, 7]),
        #     nn.LayerNorm([32, 10, 7]),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),  # (32, 10, 7) -> (64, 5, 4)
        #     nn.LayerNorm([64, 5, 4]),
        #     nn.ReLU(),
        #     layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),  # (64, 5, 4) -> (64, 5, 4)
        #     nn.LayerNorm([64, 5, 4]), 
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     layer_init(nn.Linear(64 * 5 * 4, 512)), # Final flattened size is 64 * 5 * 4
        #     nn.LayerNorm(512),
        #     nn.ReLU(),
        # )

        # self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        # self.critic = layer_init(nn.Linear(512, 1), std=1)
    
    def get_value(self, x, envs=None):
        # check if there is a batch layer, and if not, add one (batch size 1)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        normalized_x = normalize_frame_grid(x)
        hidden = self.network(normalized_x)
        # print(f'get value hidden : mean {torch.mean(hidden)} std {torch.std(hidden)}')
        if self.condition_dim == 0:
            value = self.critic(hidden)
        else:
            condition = torch.tensor(envs.bc_condition)
            condition = condition.expand(hidden.shape[0], -1)
            value = self.critic(torch.cat([condition, hidden], dim=1))
        return value
    
    def get_action_and_value(self, x, fixed_action=None, action_mask=None, epsilon_greedy=1e-2, envs=None):
        '''
        Output
        action : torch.tensor of shape (batch_size, ) with action selected
        log_prob : torch.tensor of shape (batch_size, ) with log probability of selected action
        entropy : torch.tensor of shape (batch_size, ) with entropy of action distribution
        value : torch.tensor of shape (batch_size, ) with value of state
        '''

        # check if there is a batch layer, and if not, add one (batch size 1)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        normalized_x = normalize_frame_grid(x)
        hidden = self.network(normalized_x)
        # print(f'get hidden : {hidden}')
        # print(f'get action hidden : mean {torch.mean(hidden)} std {torch.std(hidden)}')
        if self.condition_dim == 0:
            logits = self.actor(hidden) # these become too large and small causing nan!
        else:
            condition = torch.tensor(envs.bc_condition) # target (length, height, loadmag) and inventory (light, medium)
            # broadcast condition to hidden batch size\
            condition = condition.expand(hidden.shape[0], -1)
            logits = self.actor(torch.cat([condition, hidden], dim=1))
        # print(f'logits from actor : \n{logits}')
        org_probs = Categorical(logits=logits)

        # Action selection 
        if fixed_action == None:
            # Epsilon greedy action selection
            if np.random.rand() < epsilon_greedy:
                if action_mask is not None:
                    # print(f'    epsilon_greedy random action with action mask')
                    valid_actions = np.where(action_mask == 1)[0]
                    action = torch.tensor(np.random.choice(valid_actions))
                else: 
                    print(f'    action mask is None (epsilon_greedy random action)')
                    action = torch.tensor(np.random.randint(len(org_probs.probs)))
            else:
                if action_mask is not None: # mask invalid actions to get next action
                    # Assuming logits and action_mask are tensors
                    masked_logits = logits.clone()  # Clone logits to avoid modifying the original tensor
                    # Ensure action_mask has the correct shape
                    if len(action_mask.shape) == 1:
                        action_mask = torch.tensor(action_mask).unsqueeze(0)  # Add batch dimension to action_mask if needed
                    masked_logits[action_mask == 0] = -float('inf')  # Set logits of undesirable actions to -inf
                    masked_probs = Categorical(logits=masked_logits) # masked probs
                    
                    action = masked_probs.sample()

                else: # sample action without mask 
                    print(f'action mask is None, taking action according to policy') 
                    action = org_probs.sample()
        else:
            action = fixed_action
        
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        if len(action.shape) == 0:
            action = action.unsqueeze(0)
        
        # critic value
        if self.condition_dim == 0:
            value = self.critic(hidden)
        else:
            value = self.critic(torch.cat([condition, hidden], dim=1))
        return action, org_probs.log_prob(action), org_probs.entropy(), value

def video_save_trigger(episode_index):
    '''
    function that takes episode index and returns True if video should be saved used in save_video
    used as trigger to save video using gymnasium.utils.save_video
    '''
    if episode_index % args.render_interval < args.render_interval_count: # save args.render_interval_count consecutive videos at render intervals 
        print(f"Saving render at terminated episode count {episode_index}!")
        return True 
    else:
        return False


def run(args_param):
    '''
    Train the agent using PPO algorithm
    args : Args object containing hyperparameters
    '''
    global args # to use same args in video_save_trigger
    args = args_param
    print(f'Training Mode : {args.train_mode}')

    # Set batch size for training policy 
    if args.train_mode == 'train':
        # args = tyro.cli(Args)
        args.batch_size = int(args.num_envs * args.num_steps_rollout)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = int(args.total_timesteps // args.batch_size)
        # args.checkpoint_interval_steps = int(args.total_timesteps // 10) # save model every 10% of total timesteps
    elif args.train_mode == 'inference':
        args.num_iterations = 5

    # Convert the timestamp to a human-readable format
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # run_name = f"{args.env_id}_seed{args.seed}_{current_time}_{name}"
    run_name = f"{args.train_mode}_{args.exp_name}"

    if args.track_wb:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True, # automatric syncing of W&B logs from TensorBoard(writer)
            config=vars(args),
            name=run_name,
            # monitor_gym=True,
            save_code=True, # saves the code used for the run to WandB servers at start
            resume="allow",
            id=run_name, # TODO resume training what should this be? 
        )
    
    # Save data in event file that can be opened with TensorBoard
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup (parallel environment)
    # SyncVectorEnv is a wrapper to vectorize environments to run in parallel
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    # )
    
    # Use single environment
    envs = gym.make(
                    id=args.env_id,
                    frame_grid_size_x = args.frame_grid_size_x,
                    frame_grid_size_y = args.frame_grid_size_y,
                    frame_size = args.frame_size,
                    render_mode=args.render_mode, 
                    render_interval_eps=args.render_interval,
                    render_interval_consecutive=args.render_interval_count,
                    render_dir = args.render_dir,
                    max_episode_length = 400,
                    obs_mode=args.obs_mode,
                    rand_init_seed = args.rand_init_seed,
                    bc_height_options=args.bc_height_options,
                    bc_length_options=args.bc_length_options,
                    bc_loadmag_options=args.bc_loadmag_options,
                    bc_inventory_options=args.bc_inventory_options,
                    num_target_loads = args.num_target_loads,
                    bc_fixed = args.bc_fixed,
                    elem_sections = args.elem_sections,
                    high_util_percentage = args.high_util_percentage,
                    ) 
    
    # envs.print_framegrid() # DEBUG target 
    
    # Stack observations
    envs = FrameStack(envs, args.num_stacked_obs)
    
    print(f'Action Space : {envs.action_space}')
    if isinstance(envs, gym.Env): # single env
        assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    if isinstance(envs, gym.vector.SyncVectorEnv): # for parallel envs
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported" 

    # ALGO Logic: Storage setup (takes into consideration multiple environments)
    if args.train_mode == 'train':
        if args.obs_mode == 'frame_grid_singleint':
            obs = torch.zeros((args.num_steps_rollout, args.num_envs) + envs.single_observation_space.shape).to(device)
            actions = torch.zeros((args.num_steps_rollout, args.num_envs) + envs.single_action_space.shape).to(device)
            logprobs = torch.zeros((args.num_steps_rollout, args.num_envs)).to(device)
            values = torch.zeros((args.num_steps_rollout, args.num_envs)).to(device)
            rewards = torch.zeros((args.num_steps_rollout, args.num_envs)).to(device)
            dones = torch.zeros((args.num_steps_rollout, args.num_envs)).to(device)
            
        elif args.obs_mode == 'frame_grid':
            # obs_shape = (args.num_steps_rollout, args.num_stacked_obs) + envs.single_observation_space.shape
            # action_shape = (args.num_steps_rollout, args.num_envs) + envs.single_action_space.shape
            # logprobs_shape = (args.num_steps_rollout, args.num_envs)
            # values_shape = (args.num_steps_rollout, args.num_envs)
            obs_shape = (args.num_stacked_obs,) + envs.single_observation_space.shape
            action_shape = (args.num_envs,) + envs.single_action_space.shape
            logprobs_shape = (args.num_envs,)
            values_shape = (args.num_envs,)
            reward_shape = (args.num_envs,)
            dones_shape = (args.num_envs,)

            if args.collect_complete: # collect complete trajectories
                obs = np.empty((0,) + obs_shape)
                actions = np.empty((0,)+ action_shape)
                logprobs = np.empty((0,)+ logprobs_shape)
                values = np.empty((0,)+ values_shape)
                rewards = np.empty((0,)+ reward_shape)
                dones = np.empty((0,)+ dones_shape)
                # print(f'buffer shapes | obs : {obs.shape}, actions : {actions.shape}, logprobs :{logprobs.shape}, values : {values.shape}, rewards : {rewards.shape}, dones : {dones.shape}')

            else: # collect all trajectories
                obs = torch.zeros((args.num_steps_rollout,) + obs_shape).to(device)
                actions = torch.zeros((args.num_steps_rollout,) + action_shape).to(device)
                logprobs = torch.zeros((args.num_steps_rollout,) + logprobs_shape).to(device)
                values = torch.zeros((args.num_steps_rollout,) + values_shape).to(device)
                rewards = torch.zeros((args.num_steps_rollout,) + reward_shape).to(device)
                dones = torch.zeros((args.num_steps_rollout,) + dones_shape).to(device)
    elif args.train_mode == 'inference':
        if args.obs_mode == 'frame_grid_singleint':
            rewards = torch.zeros((args.num_steps_rollout, args.num_envs)).to(device)
            dones = torch.zeros((args.num_steps_rollout, args.num_envs)).to(device)
            
        elif args.obs_mode == 'frame_grid':
            reward_shape = (args.num_envs,)
            dones_shape = (args.num_envs,)

            if args.collect_complete: # collect complete trajectories
                rewards = np.empty((0,)+ reward_shape)
                dones = np.empty((0,)+ dones_shape)

            else: # collect all trajectories
                rewards = torch.zeros((args.num_steps_rollout,) + reward_shape).to(device)
                dones = torch.zeros((args.num_steps_rollout,) + dones_shape).to(device)



    # if train_mode is inference, do not store trajectory, only store rewards, dones
    # rewards = torch.zeros((args.num_steps_rollout, args.num_envs)).to(device)
    # dones = torch.zeros((args.num_steps_rollout, args.num_envs)).to(device)
    
    # Set Agent Network
    if envs.obs_mode == 'frame_grid_singleint':
        agent = Agent(envs).to(device)
    elif envs.obs_mode == 'frame_grid':
        agent = Agent_CNN(envs, 
                          num_stacked_obs=args.num_stacked_obs,
                          condition_dim=args.condition_dim,
                          ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # If loading from checkpoint, load the model and optimizer state dictionaries
    if args.train_mode == 'inference':
        assert args.load_checkpoint, "args.load_checkpoint should be True for inference"
        global_step = 0
        start_step = 0
    if args.load_checkpoint:
        assert args.load_checkpoint_path is not None, "Please provide a checkpoint path for loading the model."
        checkpoint = torch.load(args.load_checkpoint_path)

        agent.load_state_dict(checkpoint['model_state_dict'])
        # Load the optimizer state dictionary
        if args.train_mode == 'train':
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global_step = checkpoint['global_step']
            try: 
                start_iteration = checkpoint['iteration']
            except:
                start_iteration = 0
            print(f"Model loaded from checkpoint {args.load_checkpoint_path} at global step {global_step} iteration {start_iteration}!")
        else:
            start_iteration = 0
            print(f"Model loaded from checkpoint {args.load_checkpoint_path} for inference!")
    else:
        global_step = 0
        start_iteration = 0
    
    # TRY NOT TO MODIFY: start the game
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(np.array(next_obs)).to(device) 
    next_done = torch.zeros(args.num_envs).to(device)

    # Episode count 
    term_eps_idx = 0 # count episodes where designs were completed (terminated)

    # create h5py file
    if args.save_h5:
        # Open the HDF5 file at the beginning of rollout
        h5dir = 'train_h5'
        save_hdf5_filename = os.path.join(h5dir, f'{run_name}.h5')
        h5f = h5py.File(save_hdf5_filename, 'a', track_order=True)  # Use 'w' to overwrite or 'a' to append

        with h5py.File(save_hdf5_filename, 'a', track_order=True) as f:
            save_env_render_properties(f, envs)

    # Start iterations
    for iteration in tqdm(range(start_iteration, args.num_iterations + 1), desc="Iterations"):
        # Annealing the rate if instructed to do so.
        if args.train_mode == 'train' and args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout
        # set start step for loading checkpoint
        if args.load_checkpoint and iteration == start_iteration and args.train_mode == 'train':
            start_step = global_step % args.num_steps_rollout
        else:
            start_step = 0

        if args.collect_complete: # rollout loop to collect complete rollouts with epsilon greedy
            # step = 0
            # collect transition data for each episode, only use those that are completed
            eps_obs = []
            eps_actions = []
            eps_logprobs = []
            eps_values = []
            eps_dones = []
            eps_rewards = []
            # print(f'buffer \n obs : {obs} \n actions : {actions} \n logprobs : {logprobs} \n values : {values} \n dones : {dones} \n rewards : {rewards}')
            while obs.shape[0] < args.num_steps_rollout: # run rollout until buffer is full 
                global_step += args.num_envs # global steps are collected steps 
                if args.train_mode == 'train':
                    # save checkpoint at intervals
                    if args.save_checkpoint and global_step % args.checkpoint_interval_steps == 0:
                        model_path = f"{args.render_dir}/checkpoint_step{global_step}.pth"
                        torch.save({
                                        'model_state_dict': agent.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'global_step': global_step,
                                        'iteration' : iteration, 
                                    }, model_path)
                        print(f"Model with complete rollouts saved at {model_path}!")
                    eps_obs.append(next_obs)
                eps_dones.append(next_done)

                if envs.reset_env_bool == True: # initialize random actions at reset of env
                    rand_init_counter = args.rand_init_steps # Initialize a counter for random initialization steps that counts down (at reset of env after termination)

                if rand_init_counter > 0 and args.train_mode == 'train': # take random action at initialization of episode
                    # print(f'step{envs.global_step} random init counter : {rand_init_counter}')
                    with torch.no_grad(): # TODO rightnow envs : single env -> adjust for parallel envs
                        curr_mask = envs.get_action_mask() 
                        action = envs.action_space.sample(mask=curr_mask)  # sample random action with action mask
                        # if isinstance(action, torch.Tensor):
                        #     action = action.cpu().numpy()
                        envs.add_rand_action(action)
                        # if args.train_mode == 'train':
                        action, logprob, _, value = agent.get_action_and_value(x=next_obs, fixed_action=action, action_mask=curr_mask, epsilon_greedy=args.epsilon_greedy) # get logprob and value for random action

                    rand_init_counter -= 1

                else: # ALGO LOGIC: take action according to policy
                    with torch.no_grad():
                        curr_mask = envs.get_action_mask()
                        if args.train_mode == 'train': # get action with epsilon greedy
                            action, logprob, _, value = agent.get_action_and_value(x=next_obs, action_mask=curr_mask, epsilon_greedy=args.epsilon_greedy)
                        elif args.train_mode == 'inference': # get action without randomness
                            action, logprob, _, value = agent.get_action_and_value(x=next_obs, action_mask=curr_mask, epsilon_greedy=0)
                
                # log episode action info
                eps_actions.append(action)
                eps_logprobs.append(logprob)
                eps_values.append(value.flatten())

                # make step with action and log reward and termination data
                # next_obs, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action)
                next_obs, reward, terminations, truncations, info = envs.step(action)
                next_done = np.array([np.logical_or(terminations, truncations)]).astype(int)
                next_obs, next_done = torch.Tensor(np.array(next_obs)).to(device), torch.Tensor(next_done).to(device)

                
                eps_rewards.append(torch.tensor(reward).to(device).view(-1)) # TODO use reward from termination?

                if terminations == True or truncations == True:
                    writer.add_scalar("charts/episodic_return", info['final_info']["episode"]["reward"], global_step)
                    writer.add_scalar("charts/episodic_length", info['final_info']["episode"]["length"], global_step)

                    if truncations == True: # incomplete design
                        if np.random.rand() < args.collect_complete_epsilon: # only log incomplete designs with epsilon greedy
                            # save episode data to buffer
                            obs = np.vstack((obs, np.array(eps_obs)))
                            actions = np.vstack((actions, np.array(eps_actions)))
                            logprobs = np.vstack((logprobs, np.array(eps_logprobs)))
                            values = np.vstack((values, np.array(eps_values)))
                            rewards = np.vstack((rewards, np.array(eps_rewards)))
                            dones = np.vstack((dones, np.array(eps_dones)))

                    if terminations == True: # complete design
                        # save epsiode data to buffer
                        obs = np.vstack((obs, np.array(eps_obs)))
                        actions = np.vstack((actions, np.array(eps_actions)))
                        logprobs = np.vstack((logprobs, np.array(eps_logprobs)))
                        values = np.vstack((values, np.array(eps_values)))
                        # term_eps_rewards = np.full((len(eps_rewards), 1), reward) # # use last reward for all steps in episode
                        # amortize reward over all steps in episode
                        num_steps = len(eps_rewards)
                        # Precompute the denominator so the max value normalizes to 1
                        denom = np.log(1 + np.e * ((num_steps - 1) / num_steps))
                        # Vectorized decay schedule
                        term_eps_rewards = (
                            reward
                            * (np.log(1 + np.e * (np.arange(num_steps) / num_steps)) / denom)
                            + 0.1
                        ).reshape(num_steps, 1)
                        # print(f'term_eps_rewards : {term_eps_rewards}')
                        # add batch dimension to term_eps_rewards
                        rewards = np.vstack((rewards, np.array(term_eps_rewards)))
                        # rewards = np.vstack((rewards, np.array(eps_rewards)))
                        dones = np.vstack((dones, np.array(eps_dones)))

                        if envs.render_mode == "rgb_list": # save video
                            assert args.render_dir is not None, "Please provide a directory path render_dir for saving the rendered video."
                            save_video(
                                        frames=envs.get_render_list(),
                                        video_folder=args.render_dir,
                                        fps=envs.metadata["render_fps"],
                                        # video_length = ,
                                        # name_prefix = f"train_iter-{iteration}", # (f"{path_prefix}-episode-{episode_index}.mp4")
                                        episode_index = term_eps_idx, # why need +1?
                                        # step_starting_index=step_starting_index,
                                        episode_trigger = video_save_trigger
                            )
                        term_eps_idx += 1 # count episodes where designs were completed (terminated)
                        if args.save_h5:
                            # Save data to hdf5 file
                            save_episode_hdf5(h5f, term_eps_idx, envs.unwrapped.curr_fea_graph, envs.unwrapped.frames, envs.unwrapped.curr_frame_grid)
                            # Flush (save) data to disk (optional - may slow down training)
                            h5f.flush()

                    envs.reset(seed=args.seed)
                    rand_init_counter = args.rand_init_steps # reset random initialization counter
                    # reset episode buffer
                    eps_obs = []
                    eps_actions = []
                    eps_logprobs = []
                    eps_values = []
                    eps_dones = []
                    eps_rewards = []

            # turn buffer into tensor for training
            obs = torch.tensor(obs).to(device)[:args.num_steps_rollout]
            actions = torch.tensor(actions).to(device)[:args.num_steps_rollout]
            logprobs = torch.tensor(logprobs).to(device)[:args.num_steps_rollout]
            values = torch.tensor(values).to(device)[:args.num_steps_rollout]
            dones = torch.tensor(dones).to(device)[:args.num_steps_rollout]
            rewards = torch.tensor(rewards).to(device)[:args.num_steps_rollout]


                

        else: # rollout loop to collect all trajectories
            for step in range(start_step, args.num_steps_rollout): # total number of steps regardless of number of eps
                global_step += args.num_envs # shape is (args.num_steps_rollout, args.num_envs, envs.single_observation_space.shape)
                if args.train_mode == 'train':
                    # save checkpoint at intervals
                    if args.save_checkpoint and global_step % args.checkpoint_interval_steps == 0:
                        model_path = f"{args.render_dir}/checkpoint_step{global_step}.pth"
                        torch.save({
                                        'model_state_dict': agent.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'global_step': global_step,
                                        'iteration' : iteration, 
                                    }, model_path)
                        print(f"Model saved at {model_path}!")
                    obs[step] = next_obs
                dones[step] = next_done

                if envs.reset_env_bool == True: # initialize random actions at reset of env
                    rand_init_counter = args.rand_init_steps # Initialize a counter for random initialization steps that counts down (at reset of env after termination)
                
                if rand_init_counter > 0 and args.train_mode == 'train': # sample random action at initialization of episode
                    # print(f'step{envs.global_step} random init counter : {rand_init_counter}')
                    with torch.no_grad(): # TODO rightnow envs : single env -> adjust for parallel envs
                        curr_mask = envs.get_action_mask() 
                        action = envs.action_space.sample(mask=curr_mask)  # sample random action with action mask
                        if isinstance(action, torch.Tensor):
                            action = action.cpu().numpy()
                        envs.add_rand_action(action)
                        # if args.train_mode == 'train':
                        action, logprob, _, value = agent.get_action_and_value(x=next_obs, fixed_action=action, action_mask=curr_mask, epsilon_greedy=args.epsilon_greedy) # get logprob and value for random action
                        actions[step] = action
                        logprobs[step] = logprob # log only if not nan
                        values[step] = value.flatten()

                    rand_init_counter -= 1

                else: # ALGO LOGIC: action according to policy
                    with torch.no_grad():
                        curr_mask = envs.get_action_mask()
                        if args.train_mode == 'train': # get action with epsilon greedy
                            action, logprob, _, value = agent.get_action_and_value(x=next_obs, action_mask=curr_mask, epsilon_greedy=args.epsilon_greedy)
                            # log action info
                            actions[step] = action
                            logprobs[step] = logprob
                            values[step] = value.flatten()
                        elif args.train_mode == 'inference': # get action without randomness
                            action, logprob, _, value = agent.get_action_and_value(x=next_obs, action_mask=curr_mask, epsilon_greedy=0)
                
                # make step with action and log reward and termination data
                # next_obs, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action)
                next_obs, reward, terminations, truncations, info = envs.step(action)
                next_done = np.array([np.logical_or(terminations, truncations)]).astype(int)
                next_obs, next_done = torch.Tensor(np.array(next_obs)).to(device), torch.Tensor(next_done).to(device)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                
                if terminations == True or truncations == True:
                    writer.add_scalar("charts/episodic_return", info['final_info']["episode"]["reward"], global_step)
                    writer.add_scalar("charts/episodic_length", info['final_info']["episode"]["length"], global_step)

                    if terminations == True: # complete design
                        if envs.render_mode == "rgb_list":
                            assert args.render_dir is not None, "Please provide a directory path render_dir for saving the rendered video."
                            save_video(
                                        frames=envs.get_render_list(),
                                        video_folder=args.render_dir,
                                        fps=envs.metadata["render_fps"],
                                        # video_length = ,
                                        # name_prefix = f"train_iter-{iteration}", # (f"{path_prefix}-episode-{episode_index}.mp4")
                                        episode_index = term_eps_idx, # why need +1?
                                        # step_starting_index=step_starting_index,
                                        episode_trigger = video_save_trigger
                            )
                        term_eps_idx += 1 # count episodes where designs were completed (terminated)
                        if args.save_h5:
                            # Save data to hdf5 file
                            save_episode_hdf5(h5f, term_eps_idx, envs.unwrapped.curr_fea_graph, envs.unwrapped.frames, envs.unwrapped.curr_frame_grid)
                            # Flush (save) data to disk (optional - may slow down training)
                            h5f.flush()

                    envs.reset(seed=args.seed)
                    rand_init_counter = args.rand_init_steps # reset random initialization counter


        # Train policy (actor, critic)
        if args.train_mode == 'train' and iteration != start_iteration:

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps_rollout)):
                    if t == args.num_steps_rollout - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            if args.obs_mode == "frame_grid_singleint":
                b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            elif args.obs_mode == "frame_grid":
                b_obs = obs.reshape((-1,) + (args.num_stacked_obs,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    try: 
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(x=b_obs[mb_inds], fixed_action=b_actions.long()[mb_inds], epsilon_greedy=0)
                    except ValueError as e:
                        print(f'Error in train get_action_and_value : {e}')
                        print(f'input obs :\n {b_obs[mb_inds]}')
                        print(f'actions : \n{b_actions[mb_inds]}')
                        print(f'logprobs : \n{b_logprobs[mb_inds]}')
                        print(f'inds : \n{mb_inds}')
                        # continue
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    writer.add_scalar("gradients/grad_norm", get_grad_norm(agent), global_step) # DEBUG check if gradients explode
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step) # indirect indicator of how long it takes to complete an episode
            writer.add_scalar("charts/time", time.time() - start_time, global_step)
            writer.add_scalar("charts/episode", term_eps_idx, global_step)

            writer.add_scalar("logprobs/b_logprobs_max", b_logprobs.max(), global_step)
            writer.add_scalar("logprobs/b_logprobs_min", b_logprobs.min(), global_step)
            writer.add_scalar("logprobs/b_logprobs_std", b_logprobs.std(), global_step)
            writer.add_scalar("logprobs/b_logprobs_mean", b_logprobs.mean(), global_step)

            writer.add_scalar("values/b_values_max", b_values.max(), global_step)
            writer.add_scalar("values/b_values_min", b_values.min(), global_step)
            writer.add_scalar("values/b_values_std",b_values.std(), global_step)
            writer.add_scalar("values/b_values_mean", b_values.mean(), global_step)


            for name, param in agent.named_parameters():
                if param.requires_grad:
                    writer.add_scalar(f"weights/{name}_mean", param.data.mean().item(), global_step)
                    writer.add_scalar(f"weights/{name}_std", param.data.std().item(), global_step)
        
        # reset buffer
        if args.collect_complete: # collect complete trajectories
            obs = np.empty((0,) + obs_shape)
            actions = np.empty((0,)+ action_shape)
            logprobs = np.empty((0,)+ logprobs_shape)
            values = np.empty((0,)+ values_shape)
            rewards = np.empty((0,)+ reward_shape)
            dones = np.empty((0,)+ dones_shape)

    envs.close()
    writer.close()