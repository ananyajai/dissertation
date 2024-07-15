import random
import csv
import numpy as np 
from tqdm import tqdm
from BSE import market_session
from matplotlib import pyplot as plt
from collections import defaultdict
from typing import List, Dict, DefaultDict, Tuple
from q_table_data import load_q_table, dump_q_table
from epsilon_scheduling import epsilon_decay


gamma = 1.0
alpha = 1e-4


def load_episode_data(file: str) -> Tuple[List, List, List]:
    obs_list, action_list, reward_list = [], [], []

    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            obs_str = row[0].strip('[]').split()[1:]
            obs_list.append(np.array([float(x.strip("'")) for x in obs_str]))        # Convert the string values to floats
            action_list.append((float(row[1])))
            reward_list.append(float(row[2]))

    return obs_list, action_list, reward_list


def learn(obs: List[int], actions: List[int], rewards: List[float], type, episode) -> Dict:
    # Load the current q_table from the CSV file
    try:
        if type == 'Buyer':
            q_table = load_q_table('q_table_buyer.csv')
        elif type == 'Seller':
            q_table = load_q_table('q_table_seller.csv')
    except FileNotFoundError:
        q_table = defaultdict(lambda: 0)

    sa_counts = defaultdict(lambda: 0)
    traj_length = len(rewards)
    # G = 0
    # state_action_list = list(zip([tuple(o) for o in obs], actions))

    # # Iterate over the trajectory backwards
    # for t in range(traj_length - 1, -1, -1):
    #     state_action_pair = (tuple(obs[t]), actions[t])

    #     # Check if this is the first visit to the state-action pair
    #     if state_action_pair not in state_action_list[:t]:
    #         G = gamma*G + rewards[t]

    #         # Monte-Carlo update rule
    #         sa_counts[state_action_pair] = sa_counts.get(state_action_pair, 0) + 1
    #         q_table[state_action_pair] += (
    #             G - q_table[state_action_pair]
    #             ) / sa_counts.get(state_action_pair, 0)
    
    # Precompute returns G for every timestep
    G = [ 0 for n in range(traj_length) ]
    G[-1] = rewards[-1]
    for t in range(traj_length - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]

    # Update Q-values using every-visit MC method
    for t in range(traj_length):
        state_action_pair = (tuple(obs[t]), actions[t])
        sa_counts[state_action_pair] += 1
        q_table[state_action_pair] += (G[t] - q_table[state_action_pair]) / sa_counts[state_action_pair]

    # Save the updated q_table back to the CSV file
    if type == 'Buyer':
        dump_q_table(q_table, 'q_table_buyer.csv')
    elif type == 'Seller':
        dump_q_table(q_table, 'q_table_seller.csv')
    
    return q_table


# Should change epsilon to 0.0 here so the agent doesn't explore
# but rather just tries to maximise rewards
def evaluate(episodes: int, market_params: tuple, q_table: DefaultDict, file) -> float:
    total_return = 0.0
    mean_return_list = []

    updated_market_params = list(market_params)    
    if file == 'q_table_buyer.csv':
        updated_market_params[3]['buyers'][0][2]['q_table_buyer'] = 'q_table_buyer.csv'
        updated_market_params[3]['buyers'][0][2]['epsilon'] = 0.0                           # No exploring
    elif file == 'q_table_seller.csv':
        updated_market_params[3]['sellers'][0][2]['q_table_seller'] = 'q_table_seller.csv'
        updated_market_params[3]['sellers'][0][2]['epsilon'] = 0.0                          # No exploring

    for _ in range(episodes):
        balance = 0.0
        market_session(*market_params)

        # Read the episode file
        with open(file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                reward = float(row[2])
                balance += reward

        # Profit made by the RL agent at the end of the trading window
        total_return += balance
    
    mean_return = total_return / episodes
    mean_return_list.append(mean_return)

    return mean_return, mean_return_list


def train(total_eps: int, market_params: tuple, eval_freq: int, epsilon) -> DefaultDict:
    for episode in range(1, total_eps + 1):
        market_session(*market_params)
        
        # # Update market_params to include the current epsilon
        # updated_market_params = list(market_params)
        # updated_market_params[3]['sellers'][0][2]['epsilon'] = epsilon
        
        # # Epsilon scheduling
        # epsilon = epsilon_decay('linear', episode, total_eps)
        # # # Run one market session to get observations, actions, and rewards
        # market_session(*updated_market_params)

        # Check if there's a buy trader
        try:
            file = 'episode_buyer.csv'
            obs_list, action_list, reward_list = load_episode_data(file)
            # Learn from the experience with the MC update
            q_table_buyer = learn(obs_list, action_list, reward_list, 'Buyer')
            
        except:
            pass
        
        # Check if there's a sell trader
        try:
            file = 'episode_seller.csv'
            obs_list, action_list, reward_list = load_episode_data(file)
            # Learn from the experience with the MC update
            q_table_seller = learn(obs_list, action_list, reward_list, 'Seller', episode)
        except:
            pass
        
        # Perform evaluation every `eval_freq` episodes
        if episode % eval_freq == 0:
            print(f"Training Episode {episode}/{total_eps}")

            # mean_return_buyer, mean_return_list = evaluate(
            #     episodes=CONFIG['eval_episodes'], market_params=market_params, 
            #     q_table='q_table_buyer.csv', file='episode_buyer.csv'
            #     )
            
            mean_return_seller, mean_return_list = evaluate(
                episodes=CONFIG['eval_episodes'], market_params=market_params, 
                q_table='q_table_seller.csv', file='episode_seller.csv'
                )
            tqdm.write(f"EVALUATION: EP {episode} - MEAN RETURN SELLER {mean_return_seller}")



CONFIG = {
    "total_eps": 100,
    "eval_freq": 10,
    "eval_episodes": 10,
    "gamma": 1.0,
    "epsilon": 1.0,
}

# Define market parameters
sess_id = 'session_1'
start_time = 0.0
end_time = 30.0

sellers_spec = [('RL', 2, {'epsilon': 1.0})]
buyers_spec = [('ZIC', 20)]

trader_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

range1 = (4, 6)
supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

range2 = (4, 6)
demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

# new customer orders arrive at each trader approx once every order_interval seconds
order_interval = 30

order_schedule = {'sup': supply_schedule, 'dem': demand_schedule,
                'interval': order_interval, 'timemode': 'drip-poisson'}

dump_flags = {'dump_strats': False, 'dump_lobs': False, 'dump_avgbals': False, 'dump_tape': False, 'dump_blotters': False}
verbose = False


# Training the RL agent with evaluation
q_table = train(total_eps=CONFIG['total_eps'], 
                market_params=(sess_id, start_time, end_time, trader_spec, order_schedule, dump_flags, verbose), 
                eval_freq=CONFIG['eval_freq'],
                epsilon=CONFIG['epsilon'])

