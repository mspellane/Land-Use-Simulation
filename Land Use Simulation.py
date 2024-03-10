
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


grid_size= 25
zones={'residential':0,
       'commercial': 1}
development_costs={'L':0,
                   'M':1,
                   'H':2}
n_actions = len(zones)*len(development_costs)

intention_grid = np.full((grid_size, grid_size), -1)

def normalize_state(grid, cash1, cash2, max_cash):
    normalized_grid = grid.flatten() / len(development_costs)
    normalized_cash1 = cash1 / max_cash
    normalized_cash2 = cash2 / max_cash
    return np.concatenate((normalized_grid, [normalized_cash1, normalized_cash2]))

learning_rate = 0.01
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01

model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(grid_size * grid_size + 2,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(n_actions)
])

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(grid_size * grid_size + 2,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(n_actions)
])

optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model1.compile(loss="mse", optimizer=optimizer1)
model2.compile(loss="mse", optimizer=optimizer2)

def choose_action(state, epsilon, model, intention_grid, agent_id):
    if np.random.rand() < epsilon:
        action = np.random.randint(n_actions)
    else:
        action = np.argmax(model.predict(state.reshape(1, -1))[0])

    current_position = (state[:-2] * len(development_costs)).astype(int).reshape(grid_size, grid_size)
    zone_index, cost_index = divmod(action, len(development_costs))
    selected_cost = list(development_costs.keys())[cost_index]

    while current_position[current_position == development_costs[selected_cost]].size > 0 and intention_grid[current_position == development_costs[selected_cost]][0] != -1 and intention_grid[current_position == development_costs[selected_cost]][0] != agent_id:
        action = (action + 1) % n_actions
        zone_index, cost_index = divmod(action, len(development_costs))
        selected_cost = list(development_costs.keys())[cost_index]

    return action



def update_q_values(state, action, reward, next_state, done, model):
    target = model.predict(state.reshape(1, -1))[0]
    if done:
        target[action] = reward
    else:
        target[action] = reward + gamma * np.max(model.predict(next_state.reshape(1, -1))[0])
    return target

def train_step(state, action, target, model):
    model.fit(state.reshape(1, -1), target.reshape(1, -1), epochs=1, verbose=0)
    
def display_grid(grid, agent_grid, episode):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(ticks=[0, 1, 2, 3], label='Zones and Agents')
    plt.title(f'Grid at Episode {episode}')

    for y in range(grid_size):
        for x in range(grid_size):
            if agent_grid[y, x] != -1:
                zone_type = "Residential" if grid[y, x] == 0 or grid[y, x] == 2 else "Commercial"
                plt.text(x, y, f"Agent {agent_grid[y, x] + 1}\n{zone_type}", ha="center", va="center", color="white", fontsize=9)

    plt.show()

def simulate_game_step(state, action, grid, cash1, cash2, agent_grid, intention_grid, agent_id):
    current_position = (state[:-2] * len(development_costs)).astype(int).reshape(grid_size, grid_size)
    zone_index, cost_index = divmod(action, len(development_costs))
    selected_zone = list(zones.keys())[zone_index]
    selected_cost = list(development_costs.keys())[cost_index]

    if selected_cost == 'L':
        action_cost = 25
    elif selected_cost == 'M':
        action_cost = 50
    else:  # 'H'
        action_cost = 100

    agent = agent_id
    if agent == 0:
        cash = cash1
    else:
        cash = cash2

    if cash < action_cost:
        reward = -100
        done = False
    else:
        grid[current_position == development_costs[selected_cost]] = zones[selected_zone]
        agent_grid[current_position == development_costs[selected_cost]] = agent
        intention_grid[current_position == development_costs[selected_cost]] = -1

        if agent == 0:
            cash1 -= action_cost
        else:
            cash2 -= action_cost

        if selected_zone == 'residential':
            rent = 100
        elif selected_zone == 'commercial':
            rent = 200

        property_value_increase = action_cost * 0.1
        reward = property_value_increase + rent

        done = (grid != -1).all()

    next_state = normalize_state(grid, cash1, cash2, max_cash)
    return next_state, reward, done, agent


def plot_performance_metrics(rewards1, rewards2, exploration_rates1, exploration_rates2):
    accumulated_rewards1 = np.cumsum(rewards1)
    accumulated_rewards2 = np.cumsum(rewards2)
    avg_rewards1 = pd.Series(rewards1).rolling(window=100).mean()
    avg_rewards2 = pd.Series(rewards2).rolling(window=100).mean()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    ax1.plot(accumulated_rewards1, label="Agent 1")
    ax1.plot(accumulated_rewards2, label="Agent 2")
    ax1.set_title("Accumulated Rewards")
    ax1.set_ylabel("Rewards")
    ax1.legend()

    ax2.plot(avg_rewards1, label="Agent 1")
    ax2.plot(avg_rewards2, label="Agent 2")
    ax2.set_title("Average Rewards (Rolling 100 episodes)")
    ax2.set_ylabel("Rewards")
    ax2.legend()

    ax3.plot(exploration_rates1, label="Agent 1")
    ax3.plot(exploration_rates2, label="Agent 2")
    ax3.set_title("Exploration Rates")
    ax3.set_xlabel("Episodes")
    ax3.set_ylabel("Epsilon")
    ax3.legend()

    plt.show()


n_episodes = 1000
max_steps_per_episode = 100
max_cash = 10000

rewards1 = []
rewards2 = []
steps = []
exploration_rates1 = []
exploration_rates2 = []

agent_grid = np.full((grid_size, grid_size), -1)

for episode in range(n_episodes):
    grid = np.random.randint(len(development_costs), size=(grid_size, grid_size))
    cash1 = np.random.randint(max_cash // 2, max_cash)
    cash2 = np.random.randint(max_cash // 2, max_cash)
    state = normalize_state(grid, cash1, cash2, max_cash)
    agent_grid.fill(-1)
    intention_grid.fill(-1)

    episode_rewards1 = 0
    episode_rewards2 = 0
    episode_steps = 0

    for step in range(max_steps_per_episode):
        action1 = choose_action(state, epsilon, model1, intention_grid, 0)
        action2 = choose_action(state, epsilon, model2, intention_grid, 1)

        next_state, reward, done, agent = simulate_game_step(state, action1, grid, cash1, cash2, agent_grid, intention_grid, 0)

        if agent == 0:
            target1 = update_q_values(state, action1, reward, next_state, done, model1)
            train_step(state, action1, target1, model1)
            cash1 += reward
            episode_rewards1 += reward

        next_state, reward, done, agent = simulate_game_step(state, action2, grid, cash1, cash2, agent_grid, intention_grid, 1)
        if agent == 1:
            target2 = update_q_values(state, action2, reward, next_state, done, model2)
            train_step(state, action2, target2, model2)
            cash2 += reward
            episode_rewards2 += reward

        state = next_state
        episode_steps += 1

        if done:
            break

    if episode % 250 == 0:
        display_grid(grid, agent_grid, episode)

    rewards1.append(episode_rewards1)
    rewards2.append(episode_rewards2)
    steps.append(episode_steps)
    exploration_rates1.append(epsilon)
    exploration_rates2.append(epsilon)

    epsilon = max(epsilon * epsilon_decay, min_epsilon)
    
plot_performance_metrics(rewards1, rewards2, exploration_rates1, exploration_rates2) 
