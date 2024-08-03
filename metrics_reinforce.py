import csv
import numpy as np
from tqdm import tqdm
from send2trash import send2trash
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, DefaultDict, Tuple
from epsilon_scheduling import epsilon_decay
from early_stopping import EarlyStopping
from update import update

from neural_network import Network
import torch
from torch.optim import Adam
import torch.nn.functional as F


CONFIG = {
    "total_eps": 3000,
    "eval_freq": 300,
    "train_data_eps": 700,
    "eval_data_eps": 100,
    "val_data_eps": 100,
    "gamma": 0.3,
    "epsilon": 1.0,
    "batch_size": 32
}

# Define the value function neural network
state_size = 12
action_size = 10
value_net = Network(dims=(state_size+action_size, 32, 32, 1), output_activation=None)
value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)


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


def generate_data(total_eps: int, market_params: tuple, eps_file: str, output_file: str):
    """
    Generates testing data by running market session eval_eps 
    times and saving the trajectory data to a csv file.

    Args:
        eval_eps (int): Total number of times to run market session.
        market_params (tuple): Parameters for running market session.
        file (str): File path where the trajectory from each market session is stored.
    """
    obs_list = []
    action_list = []
    reward_list = []

    try:
        send2trash([eps_file, output_file])
    except:
        pass 

    for _ in range(total_eps):
        market_session(*market_params)
        eps_obs, eps_action, eps_reward = load_episode_data(eps_file)
        for obs, action, reward in zip(eps_obs, eps_action, eps_reward):
            if not np.all(obs == 0):  # Check if the observation is not a zero tensor
                obs_list.append(obs)
                action_list.append(action)
                reward_list.append(reward)

    return np.array(obs_list), np.array(action_list), np.array(reward_list)


def load_episode_data(file: str) -> Tuple[List, List, List]:
    obs_list, action_list, reward_list = [], [], []

    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            obs_str = row[0].strip('()').split(", ")
            obs_list.append(np.array([float(x.strip("'")) for x in obs_str]))        # Convert the string values to floats
            action_list.append((float(row[1])))
            reward_list.append(float(row[2]))

    return obs_list, action_list, reward_list


def train(
        train_obs, train_actions, train_rewards,
        val_obs, val_actions, val_rewards,
        test_obs, test_actions, test_rewards,
        total_eps: int, eval_freq: int, market_params: tuple, 
        gamma: float, batch_size: int=32
    ) -> DefaultDict:
    # Dictionary to store training statistics
    stats = defaultdict(list)
    mean_return_list = []
    valid_loss_list = []
    test_loss_list = []

    # obs_list, action_list, reward_list = load_episode_data('training_data.csv')
    for episode in range(1, total_eps + 1):
        ep_value_loss = []

        try:
            # Process data in batches
            for i in range(0, len(train_obs), batch_size):
                obs_batch = train_obs[i:i + batch_size]
                action_batch = train_actions[i:i + batch_size]
                reward_batch = train_rewards[i:i + batch_size]

                update_results = update(value_net, value_optim, obs_batch, action_batch, reward_batch, gamma=gamma)

                for key, value in update_results.items():
                    ep_value_loss.append(value)

        except Exception as e:
            pass

        # Aggregate v_loss for the episode
        avg_v_loss = np.mean(ep_value_loss)
        stats['v_loss'].append(avg_v_loss)

        # Evaluate the policy at specified intervals
        if episode % eval_freq == 0:
            val_mean_return, val_value_loss = evaluate(
                val_obs, val_actions, val_rewards,
                market_params=market_params, value_net=value_net
            )
            print(f"VALIDATION: EPOCH {episode} - MEAN RETURN {val_mean_return}, VALUE LOSS {val_value_loss}")
            valid_loss_list.append(val_value_loss)

            early_stop = EarlyStopping()
            if early_stop.should_stop(val_value_loss, value_net):
                print(f"Early stopping at epoch {episode} with validation loss {val_value_loss}")
                break

            mean_return_seller, value_loss = evaluate(
                test_obs, test_actions, test_rewards,
                market_params=market_params, value_net=value_net
            )
            tqdm.write(f"TESTING: EPOCH {episode} - MEAN RETURN {mean_return_seller}, VALUE LOSS {value_loss}")
            mean_return_list.append(mean_return_seller)
            test_loss_list.append(value_loss)

    return stats, mean_return_list, valid_loss_list, test_loss_list


def evaluate(
        obs_list, action_list, reward_list, 
        market_params: tuple, value_net
    ) -> Tuple[float, float]:
    total_return = 0.0

    # Compute the return using the value_net
    returns = []
    with torch.no_grad():  # No gradient calculation needed during evaluation
        for obs, action, reward in zip(obs_list, action_list, reward_list):
            # Flatten observation and create one-hot encoding for action
            flattened_obs = obs.flatten()
            one_hot_action = np.eye(action_size, dtype=int)[int(action)]
            
            # Concatenate flattened observation with one-hot action
            state_action_pair = np.concatenate((flattened_obs, one_hot_action))
            
            # Convert state-action pair to a tensor
            obs_tensor = torch.tensor(state_action_pair, dtype=torch.float32).unsqueeze(0)

            value = value_net(obs_tensor).item()
            returns.append(value)
            total_return += reward

    # Calculate mean return
    mean_return = total_return / len(reward_list)
    
    # Calculate value loss (MSE)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    rewards_tensor = torch.tensor(reward_list, dtype=torch.float32)
    value_loss = F.mse_loss(returns_tensor, rewards_tensor)
    total_value_loss = value_loss.item()

    return mean_return, total_value_loss


# Generate training data
train_obs, train_actions, train_rewards = generate_data(CONFIG['train_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv', 
              output_file='training_data.csv'
              )

# Generate validation data
val_obs, val_actions, val_rewards = generate_data(CONFIG['val_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv', 
              output_file='validation_data.csv'
              )

# Generate testing data
test_obs, test_actions, test_rewards = generate_data(CONFIG['eval_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv', 
              output_file='testing_data.csv'
              )


stats, mean_return_list, valid_loss_list, test_loss_list = train(
        train_obs, train_actions, train_rewards,
        val_obs, val_actions, val_rewards,
        test_obs, test_actions, test_rewards,
        total_eps=CONFIG['total_eps'],
        eval_freq=CONFIG["eval_freq"],
        market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose),
        gamma=0.3,
        batch_size=CONFIG["batch_size"]
    )

value_loss = stats['v_loss']
plt.plot(value_loss, linewidth=1.0)
plt.title(f"Value Loss - Training Data")
plt.xlabel("Epoch")
# plt.close()
plt.show()
