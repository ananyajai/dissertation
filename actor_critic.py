import csv
import numpy as np
from tqdm import tqdm
from send2trash import send2trash
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from epsilon_scheduling import linear_epsilon_decay
from early_stopping import EarlyStopping

from neural_network import Network
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F


CONFIG = {
    "total_eps": 10,
    "eval_freq": 1,
    "train_data_eps": 2100,
    "val_data_eps": 600,
    "eval_data_eps": 300,
    "policy_improv": 5,
    "epsilon": 1.0,
    "batch_size": 64
}
# Define the value function neural network
state_size = 14
action_size = 50
value_net = Network(dims=(state_size+action_size, 32, 32, 1), output_activation=None)
value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

policy_net = Network(
    dims=(state_size, 32, 32, action_size), output_activation=nn.Softmax(dim=-1)
    )
policy_optim = Adam(policy_net.parameters(), lr=1e-7, eps=1e-3, weight_decay=1e-4)

# colours = ['#085ea8', '#5379b7', '#7e95c5', '#a5b3d4', '#cbd1e2', 
#            '#f1cfce', '#eeadad', '#e88b8d', '#df676e', '#d43d51']

# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 60.0

# range1 = (50, 100)
# range2 = (100, 150)
# supply_schedule = [{'from': start_time, 'to': 20.0, 'ranges': [range1], 'stepmode': 'fixed'},
#                    {'from': 20.0, 'to': 40.0, 'ranges': [range2], 'stepmode': 'fixed'},
#                    {'from': 40.0, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (50, 150)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

demand_schedule = supply_schedule

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 60

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-fixed'}

sellers_spec = [('GVWY', 4), ('REINFORCE', 1, {'epsilon': 1.0, 'max_order_price': supply_schedule[0]['ranges'][0][1]})]
buyers_spec = [('GVWY', 5)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

dump_flags = {'dump_strats': False, 'dump_lobs': False, 'dump_avgbals': True, 'dump_tape': False, 'dump_blotters': False}
verbose = False

market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose)


def generate_data(total_eps: int, market_params: tuple, eps_file: str, value_net, gamma: float) -> Tuple[List, List, List]:
    """
    Generates training data by running market session total_eps times.

    Args:
        total_eps (int): Total number of times to run market session.
        market_params (tuple): Parameters for running market session.
        eps_file (str): File path where the output of each market session is stored.
        value_net (nn.Module): The neural network model representing the Q-function.
        gamma (float): Discount factor.

    Returns:
        Tuple: Normalized observations, actions, and target Q-values.
    """
    obs_list, action_list, q_list = [], [], []

    # Remove previous data files if they exist
    try:
        send2trash([eps_file])
    except:
        pass 
    
    for i in range(total_eps):
        market_session(*market_params)
        eps_obs, eps_actions, eps_rewards = load_episode_data(eps_file)
        
        for t in range(len(eps_rewards)):
            obs = eps_obs[t]
            action = eps_actions[t]
            reward = eps_rewards[t]

            # Calculate next_obs and next_action
            if t == len(eps_rewards) - 1:
                next_obs = None
                next_action = None
                q_t = reward  # No next state, so Q_t = R_t
            else:
                next_obs = eps_obs[t + 1]
                next_action = eps_actions[t + 1]
                
                # Compute Q_tilda(s_t+1, a_t+1) using value_net
                with torch.no_grad():
                    flattened_next_obs = next_obs.flatten()
                    one_hot_next_action = np.eye(action_size, dtype=int)[int(next_action)]
                    next_state_action_pair = np.concatenate((flattened_next_obs, one_hot_next_action))
                    next_state_action_tensor = torch.tensor(next_state_action_pair, dtype=torch.float32).unsqueeze(0)
                    q_tilda = value_net(next_state_action_tensor).item()
                
                # Compute Q_t using the Bellman equation
                q_t = reward + gamma * q_tilda
            
            obs_list.append(obs)
            action_list.append(action)
            q_list.append(q_t)

    # Convert lists to numpy arrays
    obs_array = np.array(obs_list)
    action_array = np.array(action_list)
    q_array = np.array(q_list)

    # Normalize observations
    obs_mean = np.mean(obs_array, axis=0)
    obs_std = np.std(obs_array, axis=0) + 1e-10
    normalised_obs = (obs_array - obs_mean) / obs_std

    # Normalize Q-values
    q_mean = np.mean(q_array)
    q_std = np.std(q_array) + 1e-10
    normalised_q = (q_array - q_mean) / q_std

    return normalised_obs, action_array, normalised_q


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

def train_actor_critic(
        train_obs, train_actions, train_q_values,
        val_obs, val_actions, val_q_values,
        test_obs, test_actions, test_q_values,
        epochs: int, eval_freq: int, gamma: float,
        policy_net, value_net, policy_optim, value_optim, 
        market_params, batch_size: int=32,
    ) -> DefaultDict:

    # Dictionary to store training statistics
    stats = defaultdict(list)
    valid_policy_loss_list = []
    valid_value_loss_list = []
    test_policy_loss_list = []
    test_value_loss_list = []

    rl_return_list = []
    gvwy_return_list = []

    for iteration in range(1, epochs + 1):
        ep_value_loss = []
        ep_policy_loss = []

        try:
            # Process data in batches
            for i in range(0, len(train_obs), batch_size):
                obs_batch = train_obs[i:i + batch_size]
                action_batch = train_actions[i:i + batch_size]
                q_value_batch = train_q_values[i:i + batch_size]

                # Update the networks
                update_results = update_actor_critic(
                    policy_net, value_net, policy_optim, value_optim,
                    obs_batch, action_batch, q_value_batch, gamma
                )

                ep_value_loss.append(update_results['v_loss'])
                ep_policy_loss.append(update_results['p_loss'])
                
        except Exception as e:
            pass
        
        avg_policy_loss = np.mean(ep_policy_loss)
        avg_value_loss = np.mean(ep_value_loss)
        stats['policy_loss'].append(avg_policy_loss)
        stats['value_loss'].append(avg_value_loss)

        # Evaluate the policy at specified intervals
        if iteration % eval_freq == 0:
            torch.save(value_net.state_dict(), 'value_net_checkpoint.pt')
            value_net.load_state_dict(torch.load('value_net_checkpoint.pt'))
            value_net.eval()

            # Evaluate validation losses
            validation_policy_loss, validation_value_loss = evaluate_policy(
                val_obs, val_actions, val_q_values, policy_net, value_net, gamma
            )
            valid_policy_loss_list.append(validation_policy_loss)
            valid_value_loss_list.append(validation_value_loss)

            # Evaluate testing losses
            testing_policy_loss, testing_value_loss = evaluate_policy(
                test_obs, test_actions, test_q_values, policy_net, value_net, gamma
            )
            test_policy_loss_list.append(testing_policy_loss)
            test_value_loss_list.append(testing_value_loss)

            mean_rl_return, mean_gvwy_return = eval_mean_returns(
                10000, value_net, market_params, 
                model_path='value_net_checkpoint.pt'
            )
            rl_return_list.append(mean_rl_return)
            gvwy_return_list.append(mean_gvwy_return)

            # Generate new data based on the updated policy
            train_obs, train_actions, train_q_values = generate_data(CONFIG['train_data_eps'],                                    
              market_params=market_params,
              eps_file='episode_seller.csv',
              value_net=value_net, gamma=0.0
            )

            # Update epsilon value using linear decay
            epsilon = linear_epsilon_decay(iteration, CONFIG['policy_improv'])
            market_params = list(market_params)    
            market_params[3]['sellers'][1][2]['epsilon'] = epsilon
            market_params[3]['sellers'][1][2]['value_func'] = value_net
            market_params = tuple(market_params)

    return stats, valid_policy_loss_list, valid_value_loss_list, test_policy_loss_list, test_value_loss_list, value_net, rl_return_list, gvwy_return_list


def update_actor_critic(
        policy_net, value_net, policy_optim, value_optim,
        observations, actions, q_values, gamma
    ) -> Dict[str, float]:
    value_loss = 0
    policy_loss = 0
    
    # Flatten each observation and create one-hot encodings for actions
    flattened_observations = [obs.flatten() for obs in observations]
    one_hot_actions = [np.eye(action_size, dtype=int)[int(action)] for action in actions]
    state_action_pairs = [np.concatenate((obs, act)) for obs, act in zip(flattened_observations, one_hot_actions)]
    
    # Convert state-action pairs to tensors
    state_action_tensor = torch.tensor(state_action_pairs, dtype=torch.float32)
    obs_tensor = torch.tensor(flattened_observations, dtype=torch.float32)
    action_tensor = torch.tensor(actions, dtype=torch.int64)
    q_tensor = torch.tensor(q_values, dtype=torch.float32)

    # Compute baseline values using the current value network
    baseline_values = value_net(state_action_tensor).squeeze()
    
    # Compute action probabilities using the current policy network
    eps = 1e-10
    action_probabilities = policy_net(obs_tensor) + eps
    log_probs = torch.log(action_probabilities.gather(1, action_tensor.unsqueeze(1)).squeeze(1))

    # Compute advantages
    advantages = q_tensor - baseline_values.detach()

    # Calculate policy and value losses
    policy_loss = -torch.mean(log_probs * advantages)  # Negative because we want to maximize the advantage
    value_loss = F.mse_loss(baseline_values, q_tensor)
    
    # Perform optimization step for the policy network
    policy_optim.zero_grad()
    policy_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Gradient clipping
    policy_optim.step()
    
    # Perform optimization step for the value network
    value_optim.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)  # Gradient clipping
    value_optim.step()
    
    return {"p_loss": float(policy_loss), "v_loss": float(value_loss)}


def evaluate(
        obs_list: List[np.ndarray], action_list: List[int], q_values_list: List[float], value_net
    ) -> float:

    # Compute the predicted Q-values using the value_net
    predicted_q_values = []
    with torch.no_grad():  # No gradient calculation needed during evaluation
        for obs, action in zip(obs_list, action_list):
            # Flatten observation and create one-hot encoding for action
            flattened_obs = obs.flatten()
            one_hot_action = np.eye(action_size, dtype=int)[int(action)]
            
            # Concatenate flattened observation with one-hot action
            state_action_pair = np.concatenate((flattened_obs, one_hot_action))
            
            # Convert state-action pair to a tensor
            obs_tensor = torch.tensor(state_action_pair, dtype=torch.float32).unsqueeze(0)

            # Predict Q-value for the state-action pair
            q_value = value_net(obs_tensor).item()
            predicted_q_values.append(q_value)
    
    # Calculate value loss (MSE) between predicted and actual Q-values
    predicted_q_tensor = torch.tensor(predicted_q_values, dtype=torch.float32)
    q_values_tensor = torch.tensor(q_values_list, dtype=torch.float32)
    value_loss = F.mse_loss(predicted_q_tensor, q_values_tensor)
    total_value_loss = value_loss.item()

    return total_value_loss

def evaluate_policy(
        obs_list, action_list, reward_list, policy_net, value_net, gamma
    ) -> Tuple[float, float]:
    log_probs = []
    values = []
    advantages = []

    with torch.no_grad():
        obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
        actions_tensor = torch.tensor(action_list, dtype=torch.int64)
        
        # Forward pass to get action probabilities from policy network
        action_probabilities = policy_net(obs_tensor)
        log_probs = torch.log(action_probabilities.gather(1, actions_tensor.unsqueeze(1)).squeeze(1))
        
        # Calculate values using the value network
        flattened_obs = [obs.flatten() for obs in obs_list]
        state_action_pairs = [np.concatenate((obs, np.eye(action_size, dtype=int)[int(action)]))
                              for obs, action in zip(flattened_obs, action_list)]
        state_action_tensor = torch.tensor(state_action_pairs, dtype=torch.float32)
        values = value_net(state_action_tensor).squeeze()
        
        # Calculate returns and advantages
        discounted_rewards = []
        G = 0
        for r in reversed(reward_list):
            G = r + gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        advantages = discounted_rewards - values

        # Calculate policy and value losses
        policy_loss = -torch.mean(log_probs * advantages)
        value_loss = F.mse_loss(values, discounted_rewards)

    return policy_loss.item(), value_loss.item()



def eval_mean_returns(num_trials, value_net, market_params, model_path:str = 'value_net_checkpoint.pt'):
    value_net.load_state_dict(torch.load(model_path))
    value_net.eval()

    rl_return = 0.0
    gvwy_return = 0.0

    updated_market_params = list(market_params)    
    updated_market_params[3]['sellers'][1][2]['value_func'] = value_net
    updated_market_params[3]['sellers'][1][2]['epsilon'] = 0.0         # No exploring

    for _ in range(num_trials):
        market_session(*updated_market_params)

        with open('session_1_avg_balance.csv', 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip().split(',')
            gvwy_value = float(last_line[7])  # Assuming the value is in the 8th column (index 7)
            gvwy_return += gvwy_value

            rl_value = float(last_line[11])
            rl_return += rl_value
            
    mean_rl_return = rl_return / num_trials
    mean_gvwy_return = gvwy_return / num_trials

    return mean_rl_return, mean_gvwy_return


# Generate training data
train_obs, train_actions, train_q = generate_data(CONFIG['train_data_eps'],                                    
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv',
              value_net=value_net, gamma=0.0
              )

# Generate validation data
val_obs, val_actions, val_q = generate_data(CONFIG['val_data_eps'],                                                 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv',
              value_net=value_net, gamma=0.0 
              )

# Generate testing data
test_obs, test_actions, test_q = generate_data(CONFIG['eval_data_eps'], 
              market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
              eps_file='episode_seller.csv',
              value_net=value_net, gamma=0.0
              )

# Train the value function
stats, valid_policy_loss_list, valid_value_loss_list, test_policy_loss_list, test_value_loss_list, value_net, rl_return_list, gvwy_return_list = train_actor_critic(
        train_obs, train_actions, train_q,
        val_obs, val_actions, val_q,
        test_obs, test_actions, test_q,
        epochs=CONFIG['total_eps'],
        eval_freq=CONFIG["eval_freq"],
        gamma=0.2, 
        policy_net=policy_net, value_net=value_net, 
        policy_optim=policy_optim, value_optim=value_optim,
        market_params=market_params, batch_size=CONFIG["batch_size"]
    )

# print(f"Training Policy Loss: {stats['policy_loss']}")
# print(f"Training Value Loss: {stats['value_loss']}")
# print(f"Validation Policy Loss: {valid_policy_loss_list}")
# print(f"Validation Value Loss: {valid_value_loss_list}")
# print(f"Testing Policy Loss: {test_policy_loss_list}")
# print(f"Testing Value Loss: {test_value_loss_list}")
# print(f"RL Returns: {rl_return_list}")
# print(f"GVWY Returns: {gvwy_return_list}")

# # Plot training and validation losses for the policy network (actor)
# plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# plt.plot(stats['policy_loss'], 'c', linewidth=1.0, label='Training Policy Loss')
# plt.plot(valid_policy_loss_list, 'g', linewidth=1.0, label='Validation Policy Loss')
# plt.title("Policy (Actor) Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# # Plot training and validation losses for the value network (critic)
# plt.subplot(1, 2, 2)
# plt.plot(stats['value_loss'], 'c', linewidth=1.0, label='Training Value Loss')
# plt.plot(valid_value_loss_list, 'g', linewidth=1.0, label='Validation Value Loss')
# plt.title("Value (Critic) Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.tight_layout()
# plt.savefig("policy_valid_loss.png")
# # plt.show()

# # Plot testing losses for both policy and value networks
# plt.figure(figsize=(14, 6))

# plt.subplot(1, 2, 1)
# plt.plot(test_policy_loss_list, linewidth=1.0, label='Testing Policy Loss')
# plt.title("Policy (Actor) Loss - Testing Data")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(test_value_loss_list, linewidth=1.0, label='Testing Value Loss')
# plt.title("Value (Critic) Loss - Testing Data")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.tight_layout()
# plt.savefig("policy_test_loss.png")
# # plt.show()

plt.plot(rl_return_list, 'c', label='RL')
plt.plot(gvwy_return_list, 'g', label='GVWY')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Mean Returns')
plt.title('Policy Improvement')
plt.savefig('policy_improv_AC.png')
# plt.show()
