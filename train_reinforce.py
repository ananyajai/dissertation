import random
import csv
import numpy as np 
import ast
from tqdm import tqdm
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from q_table_data import load_q_table, dump_q_table
from epsilon_scheduling import epsilon_decay
from evaluate import evaluate
from load_episode import load_episode_data

import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F

from neural_network import Network

gamma = 1.0


def generate_testing_data(eval_eps: int, market_params: tuple, file: str):
    """
    Generates testing data by running market session eval_eps 
    times and saving the trajectory data to a csv file.

    Args:
        eval_eps (int): Total number of times to run market session.
        market_params (tuple): Parameters for running market session.
        file (str): File path where the trajectory from each market session is stored.
    """
    test_obs = []
    test_actions = []
    test_rewards = []

    for _ in range(eval_eps):
        market_session(*market_params)
        obs_list, action_list, reward_list = load_episode_data(file)
        test_obs.extend(obs_list)
        test_actions.extend(action_list)
        test_rewards.extend(reward_list)
    
    with open('testing_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Observations', 'Actions', 'Rewards'])
        for obs, action, reward in zip(test_obs, test_actions, test_rewards):
            writer.writerow([obs, action, reward])

    print("Testing data has been generated")


def q_value_function(state, action):
        action_one_hot = torch.zeros(action_size)
        action_one_hot[action] = 1
        state_action = torch.cat([state, action_one_hot])
        q_value = value_net(state_action.unsqueeze(0))
        return q_value.item()


def update(
        observations: List[np.ndarray], actions: List[int], rewards: List[float],
        ) -> Dict[str, float]:
        # Initialise loss and returns
        p_loss = 0
        v_loss = 0
        G = 0
        traj_length = len(observations)

        # Normalize rewards
        rewards = np.array(rewards)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards) + 1e-10  # Add a small value to avoid division by zero
        normalised_rewards = (rewards - reward_mean) / reward_std

        # Flatten each observation and create one-hot encodings for actions
        flattened_observations = [obs.flatten() for obs in observations]
        one_hot_actions = [np.eye(action_size, dtype=int)[int(action)] for action in actions]
        
        # Concatenate flattened observations with one-hot actions
        state_action_pairs = [np.concatenate((obs, act)) for obs, act in zip(flattened_observations, one_hot_actions)]
        
        # Convert state-action pairs to a tensor
        state_action_tensor = torch.tensor(state_action_pairs, dtype=torch.float32)

        # Compute baseline values using the current value network
        baseline_values = value_net(state_action_tensor).squeeze()

        # Compute action probabilities using the current policy
        eps = 1e-10
        # action_probabilities = policy_net(obs_tensor) + eps
        
        # Precompute returns G for every timestep
        G = [ 0 for n in range(traj_length) ]
        G[-1] = normalised_rewards[-1]
        for t in range(traj_length - 2, -1, -1):
            G[t] = normalised_rewards[t] + gamma * G[t + 1]

        G = torch.tensor(G, dtype=torch.float32)
        advantage = G - baseline_values
        # p_loss = torch.mean(-advantage * torch.log(action_probabilities[torch.arange(traj_length), actions]))
        v_loss = F.mse_loss(baseline_values, G)

        # Backpropogate and perform optimisation step for the policy
        # policy_optim.zero_grad()
        # p_loss.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)       # Gradient clipping
        # policy_optim.step()

        # Backpropogate and perform optimisation step for the value function
        value_optim.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)       # Gradient clipping
        value_optim.step()
        
        return {"v_loss": float(v_loss)}


def train(total_eps: int, market_params: tuple, eval_freq: int, epsilon) -> DefaultDict:
    generate_testing_data(CONFIG['eval_episodes'], market_params, 'episode_seller.csv')

    # Dictionary to store training statistics
    stats = defaultdict(list)
    mean_return_list = []
    value_loss_list = []

    for episode in range(1, total_eps + 1):
        # Run a market session to generate the episode data
        market_session(*market_params)

        try:
            file = 'episode_seller.csv'
            obs_list, action_list, reward_list = load_episode_data(file)

            # Run the REINFORCE algorithm
            update_results = update(obs_list, action_list, reward_list)
            
            # Store the update results
            for key, value in update_results.items():
                stats[key].append(value)

        except Exception as e:
            pass

        # market_params[3]['sellers'][1][2]['value_func'] = value_net

        # Evaluate the policy at specified intervals
        if episode % eval_freq == 0:
            # print(f"Episode {episode}: {update_results}")
            
            # mean_return_seller = evaluate(
            #     episodes=CONFIG['eval_episodes'], market_params=market_params, 
            #     value_net=value_net, file='episode_seller.csv')
            # tqdm.write(f"EVALUATION: EP {episode} - MEAN RETURN SELLER {mean_return_seller}")
            # mean_return_list.append(mean_return_seller)

            mean_return_seller, value_loss = evaluate(
                episodes=CONFIG['eval_episodes'], market_params=market_params, value_net=value_net
                )
            tqdm.write(f"EVALUATION: EP {episode} - MEAN RETURN SELLER {mean_return_seller}, VALUE LOSS {value_loss}")
            mean_return_list.append(mean_return_seller)
            value_loss_list.append(value_loss)

    return stats, mean_return_list, value_loss_list

     

state_size = 40
action_size = 3
# policy_net = Network(
#     dims=(40, 32, 21), output_activation=nn.Softmax(dim=-1)
#     )

value_net = Network(dims=(state_size+action_size, 32, 32, 1), output_activation=None)

# policy_optim = Adam(policy_net.parameters(), lr=1e-4, eps=1e-3)
value_optim = Adam(value_net.parameters(), lr=1e-4, eps=1e-3)


CONFIG = {
    "total_eps": 500000,
    "eval_freq": 5000,
    "eval_episodes": 5000,
    "gamma": 1.0,
    "epsilon": 1.0,
}

# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 60.0

sellers_spec = [('GVWY', 9), ('REINFORCE', 1, {'epsilon': 1.0, 'value_func': value_net})]
buyers_spec = [('GVWY', 10)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

range1 = (50, 150)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (50, 150)
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 60

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-poisson'}

dump_flags = {'dump_strats': False, 'dump_lobs': False, 'dump_avgbals': False, 'dump_tape': False, 'dump_blotters': False}
verbose = False

# Train the agent
training_stats, eval_returns_list, value_loss_list = train(
    CONFIG['total_eps'],
    market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
    eval_freq=CONFIG['eval_freq'],
    epsilon=CONFIG['epsilon']
    )


policy_loss = training_stats['p_loss']
plt.plot(policy_loss, linewidth=1.0)
plt.title("Policy Loss vs Episode")
plt.xlabel("Episode number")
# plt.savefig("policy_loss.png")
# plt.close()
# plt.show()

value_loss = training_stats['v_loss']
plt.plot(value_loss, linewidth=1.0)
plt.title("Value Loss - Training Data")
plt.xlabel("Episode number")
plt.savefig("training_value_loss.png")
plt.close()
# plt.show()

x_ticks = np.arange(CONFIG['eval_freq'], CONFIG['total_eps']+1, CONFIG['eval_freq'])
plt.plot(x_ticks, value_loss_list, linewidth=1.0)
plt.title("Value Loss - Testing Data")
plt.xlabel("Episode number")
plt.savefig("testing_value_loss.png")
# plt.close()
# plt.show()

plt.plot(x_ticks, eval_returns_list, linewidth=1.0)
plt.title("Mean returns - REINFORCE")
plt.xlabel("Episode number")
# plt.savefig("mean_returns.png")
# plt.show()
