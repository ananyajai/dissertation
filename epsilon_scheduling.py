def linear_epsilon_decay(timestep, max_timestep, eps_start=0.97, eps_min=0.2, eps_decay=0.9):
    decay_steps = eps_decay * max_timestep
    epsilon = max(eps_min, eps_start - (eps_start - eps_min) * min(1.0, timestep / decay_steps))

    return epsilon


max_timestep = 50
eps_decay = 0.8

# for i in range(0, max_timestep+1):
#     print(linear_epsilon_decay(i, max_timestep))  