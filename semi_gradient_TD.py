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
from update import update

from neural_network import Network
import torch
from torch import nn, Tensor
from torch.optim import Adam
import torch.nn.functional as F


CONFIG = {
    "total_eps": 20,
    "eval_freq": 1,
    "train_data_eps": 3500,
    "val_data_eps": 1000,
    "eval_data_eps": 500,
    "policy_improv": 10,
    "epsilon": 1.0,
    "batch_size": 128
}

colours = ['#03045e','#085ea8', '#7e95c5', '#eeadad', '#df676e', '#d43d51']
five_colours = ['#03045e','#085ea8', '#7e95c5', '#eeadad', '#d43d51']
mb = '#085ea8'
mp = '#d43d51'

# Define the value function neural network
state_size = 15
action_size = 50
# value_net = Network(dims=(state_size+action_size, 32, 32, 32, 1), output_activation=None)
# value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

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

sellers_spec = [('ZIC', 9), ('DRL', 1, {'epsilon': 1.0, 'max_order_price': supply_schedule[0]['ranges'][0][1]})]
buyers_spec = [('ZIC', 10)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

dump_flags = {'dump_strats': False, 'dump_lobs': False, 'dump_avgbals': True, 'dump_tape': False, 'dump_blotters': False}
verbose = False

market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose)


def generate_TD_data(
        total_eps: int, market_params: tuple, eps_file: str, 
        value_net, gamma: float, norm_params=None
    ) -> Tuple[List, List, List]:
    """
    Generates training data by running market session total_eps times.

    Args:
        total_eps (int): Total number of times to run market session.
        market_params (tuple): Parameters for running market session.
        eps_file (str): File path where the output of each market session is stored.
        value_net (nn.Module): The neural network model representing the Q-function.
        gamma (float): Discount factor.
        norm_params (tuple): Normalisation parameters (obs_mean, obs_std, q_mean, q_std).

    Returns:
        Tuple: Normalised observations, actions, and normalised target Q-values.
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

            if t == len(eps_rewards) - 1:
                q_t = reward  # No next state, so Q_t = R_t
            else:
                next_obs = eps_obs[t + 1]
                next_action = eps_actions[t + 1]
                
                with torch.no_grad():
                    flattened_next_obs = next_obs.flatten()
                    one_hot_next_action = np.eye(action_size, dtype=int)[int(next_action)]
                    next_state_action_pair = np.concatenate((flattened_next_obs, one_hot_next_action))
                    next_state_action_tensor = torch.tensor(next_state_action_pair, dtype=torch.float32).unsqueeze(0)
                    q_tilda = value_net(next_state_action_tensor).item()
                
                q_t = reward + gamma * q_tilda
            
            obs_list.append(obs)
            action_list.append(action)
            q_list.append(q_t)

    obs_array = np.array(obs_list)
    action_array = np.array(action_list)
    q_array = np.array(q_list)

    # Determine normalisation parameters if they aren't provided
    if norm_params is None:
        obs_mean = np.mean(obs_array, axis=0)
        obs_std = np.std(obs_array, axis=0) + 1e-10
        q_mean = np.mean(q_array)
        q_std = np.std(q_array) + 1e-10
        norm_params = (obs_mean, obs_std, q_mean, q_std)
    else:
        obs_mean, obs_std, q_mean, q_std = norm_params

    normalised_obs = (obs_array - obs_mean) / obs_std
    normalised_q = (q_array - q_mean) / q_std

    return normalised_obs, action_array, normalised_q, norm_params
 

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
        train_obs, train_actions, train_q_values,
        val_obs, val_actions, val_q_values,
        test_obs, test_actions, test_q_values,
        epochs: int, eval_freq: int,
        value_net, value_optim, 
        batch_size: int=32,
    ) -> DefaultDict:

    value_net.train()
    # Dictionary to store training statistics
    stats = defaultdict(list)
    valid_loss_list = []
    test_loss_list = []

    for iteration in range(1, epochs + 1):
        ep_value_loss = []

        try:
            # Process data in batches
            for i in range(0, len(train_obs), batch_size):
                obs_batch = train_obs[i:i + batch_size]
                action_batch = train_actions[i:i + batch_size]
                q_value_batch = train_q_values[i:i + batch_size]

                update_results = update(value_net, value_optim, obs_batch, action_batch, q_value_batch)

                for key, value in update_results.items():
                    ep_value_loss.append(value)
                
        except Exception as e:
            pass

        # Aggregate v_loss for the episode
        avg_v_loss = np.mean(ep_value_loss)
        stats['v_loss'].append(avg_v_loss)

        # Evaluate the policy at specified intervals
        if iteration % eval_freq == 0:
            torch.save(value_net.state_dict(), 'value_net_checkpoint.pt')
            value_net.load_state_dict(torch.load('value_net_checkpoint.pt'))
            value_net.eval()

            validation_loss = evaluate(
                val_obs, val_actions, val_q_values, value_net=value_net
            )
            valid_loss_list.append(validation_loss)
            # print(f"Epoch {iteration} - Validation Loss: {validation_loss}")

            early_stop = EarlyStopping()
            if early_stop.should_stop(validation_loss, value_net):
                print(f"Early stopping at epoch {iteration} with validation loss {validation_loss}")
                break

            testing_loss = evaluate(
                test_obs, test_actions, test_q_values, value_net=value_net
            )
            test_loss_list.append(testing_loss)

    return stats, valid_loss_list, test_loss_list, value_net


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


def eval_mean_returns(num_trials, value_net, market_params, norm_params:tuple=(0, 1), model_path:str = 'value_net_checkpoint.pt'):
    value_net.load_state_dict(torch.load(model_path))
    value_net.eval()

    rl_return = 0.0
    gvwy_return = 0.0

    updated_market_params = list(market_params)    
    updated_market_params[3]['sellers'][1][2]['value_func'] = value_net
    updated_market_params[3]['sellers'][1][2]['epsilon'] = 0.0         # No exploring
    updated_market_params[3]['sellers'][1][2]['norm_params'] = norm_params

    for _ in range(num_trials):
        market_session(*updated_market_params)

        with open('session_1_avg_balance.csv', 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip().split(',')
            gvwy_value = float(last_line[11])  
            gvwy_return += gvwy_value

            rl_value = float(last_line[7])
            rl_return += rl_value
            
    mean_rl_return = rl_return / num_trials
    mean_gvwy_return = gvwy_return / num_trials

    return mean_rl_return, mean_gvwy_return


rl_returns_list = []
gvwy_returns_list = []

value_net = Network(dims=(state_size+action_size, 32, 32, 32, 1), output_activation=None)
value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

for iter in range(0, CONFIG['policy_improv']+1):
    print(f"GPI - {iter}")

    # Generate training data and calculate normalization parameters
    train_obs, train_actions, train_q, norm_params = generate_TD_data(
        CONFIG['train_data_eps'], market_params, 'episode_seller.csv', 
        value_net, gamma=0.0
    )

    # Generate validation data using the same normalization parameters
    val_obs, val_actions, val_q, _ = generate_TD_data(
        CONFIG['val_data_eps'], market_params, 'episode_seller.csv', 
        value_net, gamma=0.0, norm_params=norm_params
    )

    # Generate testing data using the same normalization parameters
    test_obs, test_actions, test_q, _ = generate_TD_data(
        CONFIG['eval_data_eps'], market_params, 'episode_seller.csv', 
        value_net, gamma=0.0, norm_params=norm_params
    )

    # Policy improvement
    mean_rl_return, mean_gvwy_return = eval_mean_returns(
                num_trials=10000, value_net=value_net, 
                market_params=market_params,
                norm_params=norm_params[:2]
            )
    
    print(f"EVALUATION: ITERATION {iter} - RL RETURN {mean_rl_return}, GVWY RETURN {mean_gvwy_return}")
    rl_returns_list.append(mean_rl_return)
    gvwy_returns_list.append(mean_gvwy_return)
    # zic_returns_list.append(mean_zic_return)

    # Train the value function
    stats, valid_loss_list, test_loss_list, value_net = train(
        train_obs, train_actions, train_q,
        val_obs, val_actions, val_q,
        test_obs, test_actions, test_q,
        epochs=CONFIG['total_eps'],
        eval_freq=CONFIG["eval_freq"],
        value_net=value_net, value_optim=value_optim,
        batch_size=CONFIG["batch_size"]
    )

    # Update epsilon value using linear decay
    epsilon = linear_epsilon_decay(iter, CONFIG['policy_improv'])
    market_params = list(market_params)    
    market_params[3]['sellers'][1][2]['epsilon'] = epsilon
    market_params[3]['sellers'][1][2]['value_func'] = value_net
    market_params = tuple(market_params)

    value_loss = stats['v_loss']
    plt.plot(value_loss, mb, linewidth=1.0, label='Training Loss')
    plt.plot(valid_loss_list, mp, linewidth=1.0, label='Validation Loss')
    plt.title(f"Iteration {iter:02d}")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(f'TD_value_loss_{iter:02d}.png')
    plt.close()
    # plt.show()


# Plotting
plt.plot(rl_returns_list, mb)
plt.xlabel('Iterations')
plt.ylabel('Average Returns')
plt.savefig('TD_gpi_with_zic.png')
# plt.show()
plt.close()


plt.plot(rl_returns_list, mb, label='RL')
plt.plot(gvwy_returns_list, mp, label='ZIC')
# plt.plot(zic_returns_list, '#03045e', label='ZIC')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Average Returns')
plt.savefig('TD_gpi_show_zic.png')
# plt.show()

# # Train the value function
# stats, valid_loss_list, test_loss_list, value_net = train(
#         train_obs, train_actions, train_q,
#         val_obs, val_actions, val_q,
#         test_obs, test_actions, test_q,
#         epochs=CONFIG['total_eps'],
#         eval_freq=CONFIG["eval_freq"],
#         value_net=value_net, value_optim=value_optim,
#         batch_size=CONFIG["batch_size"]
#     )

# value_loss = stats['v_loss']
# plt.plot(value_loss, mb, linewidth=1.0, label='Training Loss')
# plt.plot(valid_loss_list, mp, linewidth=1.0, label='Validation Loss')
# plt.title(f"Value Loss")
# plt.xlabel("Epochs")
# plt.legend()
# # plt.savefig("training_valid_loss.png")
# # plt.close()
# plt.show()

# plt.plot(test_loss_list, linewidth=1.0)
# plt.title(f"Value Loss - Testing Data")
# plt.xlabel("Epoch")
# # plt.savefig("testing_loss.png")
# # plt.close()
# plt.show()

# plt.plot(x_ticks, mean_return_list, linewidth=1.0, label='RL')
# plt.plot(x_ticks, gvwy_returns_list, linewidth=1.0, label='Giveaway')
# plt.title(f"Mean Returns")
# plt.xlabel("Epoch")
# plt.legend()
# # plt.savefig("mean_returns.png")
# # plt.close()
# plt.show()

# mean_returns_list = []
# gvwy_returns_list = []

# # Generate training data and calculate normalization parameters
# train_obs, train_actions, train_q, norm_params = generate_TD_data(CONFIG['train_data_eps'], market_params, 'episode_seller.csv', value_net, gamma=0.0)

# # Generate validation data using the same normalization parameters
# val_obs, val_actions, val_q, _ = generate_TD_data(CONFIG['val_data_eps'], market_params, 'episode_seller.csv', value_net, gamma=0.0, norm_params=norm_params)

# # Generate testing data using the same normalization parameters
# test_obs, test_actions, test_q, _ = generate_TD_data(CONFIG['eval_data_eps'], market_params, 'episode_seller.csv', value_net, gamma=0.0, norm_params=norm_params)

# for iter in range(1, CONFIG['policy_improv']+1):
    
#     print(f"GPI - {iter}")

#     # Policy improvement
#     mean_rl_return, mean_gvwy_return = eval_mean_returns(
#                 num_trials=10000, value_net=value_net, 
#                 market_params=market_params
#             )
    
#     print(f"EVALUATION: ITERATION {iter} - MEAN RETURN {mean_rl_return}")
#     mean_returns_list.append(mean_rl_return)
#     gvwy_returns_list.append(mean_gvwy_return)

#     # Policy evaluation
#     stats, valid_loss_list, test_loss_list, value_net = train(
#             train_obs, train_actions, train_q,
#             val_obs, val_actions, val_q,
#             test_obs, test_actions, test_q,
#             epochs=CONFIG['total_eps'],
#             eval_freq=CONFIG["eval_freq"],
#             gamma=0.2, value_net=value_net, value_optim=value_optim,
#             batch_size=CONFIG["batch_size"]
#         )


#     # Update epsilon value using linear decay
#     epsilon = linear_epsilon_decay(iter, CONFIG['policy_improv'])
#     market_params = list(market_params)    
#     market_params[3]['sellers'][1][2]['epsilon'] = epsilon
#     market_params[3]['sellers'][1][2]['value_func'] = value_net
#     market_params = tuple(market_params)

#     value_loss = stats['v_loss']
#     plt.plot(value_loss, 'c', linewidth=1.0, label='Training Loss')
#     plt.plot(valid_loss_list, 'g', linewidth=1.0, label='Validation Loss')
#     plt.title(f"Value Loss")
#     plt.xlabel("Epoch")
#     plt.legend()
#     plt.savefig(f"training_valid_loss_{iter}.png")
#     plt.close()
#     # plt.show()

#     # Generate training data and calculate normalization parameters
#     train_obs, train_actions, train_q, _ = generate_TD_data(
#         CONFIG['train_data_eps'], market_params, 
#         'episode_seller.csv', value_net, 
#         gamma=0.0, norm_params=norm_params
#     )

#     # Generate validation data using the same normalization parameters
#     val_obs, val_actions, val_q, _ = generate_TD_data(
#         CONFIG['val_data_eps'], market_params, 
#         'episode_seller.csv', value_net, 
#         gamma=0.0, norm_params=norm_params
#     )

#     # Generate testing data using the same normalization parameters
#     test_obs, test_actions, test_q, _ = generate_TD_data(
#         CONFIG['eval_data_eps'], market_params, 
#         'episode_seller.csv', value_net, 
#         gamma=0.0, norm_params=norm_params
#     )



# # Plotting
# plt.plot(mean_returns_list, 'c', label='RL')
# plt.plot(gvwy_returns_list, 'g', label='GVWY')
# plt.legend()
# plt.xlabel('Iterations')
# plt.ylabel('Mean Returns')
# plt.title('Policy Improvement')
# plt.savefig('policy_improvement_TD.png')
# # plt.show()


# gamma_list = np.linspace(0, 1, 11)

# # Set up the subplot grid
# fig_training, axs_training = plt.subplots(3, 4, figsize=(20, 15))
# fig_testing, axs_testing = plt.subplots(3, 4, figsize=(20, 15))
# # fig_validation, axs_validation = plt.subplots(3, 4, figsize=(20, 15))
# # fig_returns, axs_returns = plt.subplots(3, 4, figsize=(20, 15))

# # Flatten the axes arrays for easy indexing
# axs_training = axs_training.flatten()
# axs_testing = axs_testing.flatten()
# # axs_validation = axs_validation.flatten()
# # axs_returns = axs_returns.flatten()

# # Remove the last subplot (12th) if not needed
# fig_training.delaxes(axs_training[-1])
# fig_testing.delaxes(axs_testing[-1])
# # fig_validation.delaxes(axs_validation[-1])
# # fig_returns.delaxes(axs_returns[-1])

# # Start training
# for i, gamma in enumerate(gamma_list):
#     # Reinitialise the neural network and optimizer for each gamma value
#     value_net = Network(dims=(state_size + action_size, 32, 32, 32, 1), output_activation=None)
#     value_optim = Adam(value_net.parameters(), lr=1e-3, eps=1e-3)

#     # Generate training data and calculate normalisation parameters
#     train_obs, train_actions, train_q, norm_params = generate_TD_data(
#         CONFIG['train_data_eps'], market_params, 'episode_seller.csv', 
#         value_net, gamma=gamma
#     )

#     # Generate validation data using the same normalisation parameters
#     val_obs, val_actions, val_q, _ = generate_TD_data(
#         CONFIG['val_data_eps'], market_params, 'episode_seller.csv', 
#         value_net, gamma=gamma, norm_params=norm_params
#     )

#     # Generate testing data using the same normalisation parameters
#     test_obs, test_actions, test_q, _ = generate_TD_data(
#         CONFIG['eval_data_eps'], market_params, 'episode_seller.csv', 
#         value_net, gamma=gamma, norm_params=norm_params
#     )

#     stats, valid_loss_list, test_loss_list, value_net = train(
#         train_obs, train_actions, train_q,
#         val_obs, val_actions, val_q,
#         test_obs, test_actions, test_q,
#         epochs=CONFIG['total_eps'],
#         eval_freq=CONFIG["eval_freq"],
#         value_net=value_net, value_optim=value_optim,
#         batch_size=CONFIG["batch_size"]
#     )

#     value_loss = stats['v_loss']

#     # # Plot mean return
#     # axs_returns[i].plot(mean_return_list, 'c', linewidth=1.0)
#     # axs_returns[i].set_title(f"Mean Return, γ={gamma:.1f}")
#     # axs_returns[i].set_xlabel("Iteration")

#     # Plot training loss
#     axs_training[i].plot(value_loss, mb, linewidth=1.0, label='Training Loss')
#     axs_training[i].plot(valid_loss_list, mp, linewidth=1.0, label='Validation Loss')
#     axs_training[i].set_title(f"γ={gamma:.1f}", fontsize=18)
#     axs_training[i].set_xlabel("Epochs", fontsize=16)
#     axs_training[i].set_ylabel("Loss", fontsize=16)
#     axs_training[i].legend()

#     # Plot testing loss
#     x_ticks = np.arange(CONFIG['eval_freq'], CONFIG['total_eps'] + 1, CONFIG['eval_freq'])
#     axs_testing[i].plot(x_ticks, test_loss_list, '#03045e')
#     axs_testing[i].set_title(f"γ={gamma:.1f}", fontsize=18)
#     axs_testing[i].set_xlabel("Epochs", fontsize=16)
#     axs_testing[i].set_ylabel("Loss", fontsize=16)

#     # # Plot validation loss
#     # axs_validation[i].plot(x_ticks, valid_loss_list, 'g')
#     # axs_validation[i].set_title(f"Validation Loss, γ={gamma:.1f}")
#     # axs_validation[i].set_xlabel("Epoch")
#     # axs_validation[i].set_ylabel("Loss")

# # Adjust layout
# fig_training.tight_layout()
# # fig_returns.tight_layout()
# fig_testing.tight_layout()
# # fig_validation.tight_layout()

# # # Save figures
# fig_training.savefig("TD_valid_loss_gammas.png")
# # # fig_returns.savefig("mean_return_gammas.png")
# fig_testing.savefig("TD_test_loss_gammas.png")
# # # fig_validation.savefig("validation_loss_gammas.png")

# # plt.show()