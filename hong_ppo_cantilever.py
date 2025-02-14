'''
Modify ppo.py to 
- call from other script (create def main)
'''
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
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

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
else:
    print("CUDA is not available. Using CPU.")


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
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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

    def get_action_and_value(self, x, fixed_action=None, action_mask=None):
        '''
        get next action based on actor network output
        x : next_obs from previous step (batch_size, obs_space)
        action_mask : np.ndarray of shape (n,) and dtype np.int8 where 1 represents valid actions and 0 invalid / infeasible

        * actions are selected with action mask, but log probabilities are calculated without action mask
        return : action, log_prob, entropy, value
        # TODO what are the probs of unused cells? how does it affect overall logit distribution and approxmiate kl divergence?
        '''
        logits = self.actor(x) # (action_space, )
        probs = Categorical(logits=logits) # softmax
        # print(f'action probs : {probs.probs}')
        
        # Action selection 
        if fixed_action == None:
            if action_mask is not None: # mask invalid actions to get next action
                # print(f'applying action mask : {action_mask}')
                masked_probs = probs.probs * action_mask
                norm_masked_probs = masked_probs / masked_probs.sum()  # Normalize to ensure it's a valid probability distribution
                # print(f'    normalized masked action probs : \n{norm_masked_probs}') # TODO debug nan
                action = Categorical(probs=norm_masked_probs).sample()
            else: # sample action without mask
                print(f'action mask is None')
                action = probs.sample()
        else:
            action = fixed_action
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

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
    # return True

def train(args_param):
    '''
    Train the agent using PPO algorithm
    args : Args object containing hyperparameters
    '''
    global args # to use same args in video_save_trigger
    args = args_param

    # args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # args.num_iterations = args.total_timesteps // args.batch_size
    args.num_iterations = int(args.total_timesteps // args.batch_size)
    # Convert the timestamp to a human-readable format
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = args.exp_name
    # run_name = f"{args.env_id}_seed{args.seed}_{current_time}_{name}"
    run_name = f"{args.env_id}_{name}"
    if args.track_wb:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True, # saves the code used for the run to WandB servers at start
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

    # env setup
    # SyncVectorEnv is a wrapper to vectorize environments to run in parallel
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    # )
    # envs = gym.make("Cantilever-v0", render_mode=args.render_mode, render_dir=args.render_dir)
    envs = gym.make(
                    # args.env_id, 
                    # render_mode=args.render_mode, 
                    # render_save_interval=args.render_interval, 
                    # render_dir=args.render_dir, 
                    # rand_init_seed=args.rand_init_seed
                    
                    id=args.env_id,
                    render_mode=args.render_mode,
                    render_interval_eps=args.render_interval,
                    render_interval_consecutive=args.render_interval_count,
                    render_dir = args.render_dir,
                    max_episode_length = 400,
                    obs_mode='frame_grid',
                    rand_init_seed = args.rand_init_seed,) # TODO 
    
    print(f'env action space : {envs.action_space}')
    if isinstance(envs, gym.Env): # single env
        assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    if isinstance(envs, gym.vector.SyncVectorEnv): # for parallel envs
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported" 

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup (takes into consideration multiple environments)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Episode count 
    term_eps_idx =0 # count episodes where designs were completed (terminated)

    # create h5py file
    if args.save_h5:
        # Open the HDF5 file at the beginning of rollout
        h5dir = 'train_h5'
        save_hdf5_filename = os.path.join(h5dir, f'{run_name}.h5')
        h5f = h5py.File(save_hdf5_filename, 'a', track_order=True)  # Use 'w' to overwrite or 'a' to append

        with h5py.File(save_hdf5_filename, 'a', track_order=True) as f:
            save_env_render_properties(f, envs)

    # Start iterations
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Set random initialization 
        rand_init_counter = args.rand_init_steps # Initialize a counter for random initialization steps that counts down
        assert args.rand_init_steps < envs.cantilever_length_f, f"rand_init_steps {args.rand_init_steps} should be shorter than direct cantilever length {envs.cantilever_length_f}" #make sure that the rand_init_step is shorter than cantilever length
        if args.rand_init_steps > 0:
            envs.n_rand_init_steps = args.rand_init_steps # set number of random initialization steps in envs
        # print(f'envs.n_rand_init_steps : {envs.n_rand_init_steps}')
        # Rollout
        for step in range(0, args.num_steps): # total number of steps regardless of number of eps
            global_step += args.num_envs # shape is (args.num_steps, args.num_envs, envs.single_observation_space.shape) 
            obs[step] = next_obs
            dones[step] = next_done

            if rand_init_counter > 0: # sample random action at initialization of episode
                with torch.no_grad(): # TODO rightnow envs : single env -> adjust for parallel envs
                    curr_mask = envs.get_action_mask() #  np.ndarray of shape (n,) and dtype np.int8 where 1 represents valid actions and 0 invalid / infeasible actions.
                    # valid_idx = np.where(curr_mask == 1)[0]
                    # print(f'valid idx : {valid_idx}')
                    # print(f'random curr mask actions : {[envs.action_converter.decode(x) for x in valid_idx]}')  # get decoded action values for value 1 in curr_mask
                    
                    action = envs.action_space.sample(mask=curr_mask)  # sample random action with action maskz
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    # else:
                    next_obs = torch.Tensor(next_obs).to(device)
                    
                    # print(f'random action {args.rand_init_steps - rand_init_counter} \ {args.rand_init_steps} : {envs.action_converter.decode(action)}')  
                    # envs.rand_init_actions += [action] # register random action int to envs.rand_init_actions for vis
                    envs.add_rand_action(action)
                    # print(f'envs.rand_init_actions : {envs.rand_init_actions}')
                rand_init_counter -= 1

            else: # ALGO LOGIC: action according to policy
                with torch.no_grad():
                    curr_mask = envs.get_action_mask()
                    decoded_curr_mask = [envs.action_converter.decode(idx) for idx in np.where(curr_mask == 1)[0]]
                    # print(f'curr mask actions : {decoded_curr_mask}')  # get decoded action values for value 1 in curr_mask
                    action, logprob, _, value = agent.get_action_and_value(x=next_obs, action_mask=curr_mask)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # print(f'action : {action} | {envs.action_converter.decode(action)}') # DEBUG decode reverts action to 0,0,0,0?!
                # print(f'action : {action} ')
            
            # TRY NOT TO MODIFY: execute the game and log data.
            # next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # next_obs, reward, terminations, truncations, info = envs.step(action.cpu().numpy())
            next_obs, reward, terminations, truncations, info = envs.step(action)
            next_done = np.array([np.logical_or(terminations, truncations)]).astype(int)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            if terminations == True or truncations == True:
                # print("terminated or truncated!")
                # print(f"global_step={global_step}, episodic_return={info['final_info']['episode']['r']}, episodic_length={info['final_info']['episode']['l']}")
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
                                    name_prefix = f"train_iter-{iteration}", # (f"{path_prefix}-episode-{episode_index}.mp4")
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


            

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
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

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(x=b_obs[mb_inds], fixed_action=b_actions.long()[mb_inds])
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
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step) # indirect indicator of how long it takes to complete an episode

    envs.close()
    writer.close()
