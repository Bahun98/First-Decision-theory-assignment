import random
import matplotlib.pyplot as plt
import numpy as np
MOVING = True # extra cleaner plot with moving-average of reward
if MOVING:
    from scipy.signal import savgol_filter


class GridMDP:
    def __init__(self, x, y, slipping_rate, slipSwitch):
        self.x = x
        self.y = y
        self.griddy = (x, y)
        self.start = (0, 0)
        self.slip = slipping_rate
        self.slipSwitch = slipSwitch
        self.end_positive = (x, 0)
        #self.end_negative = [(i, 0) for i in range(1, x)]
        self.bed = [i for i in range(1, x-1)]
        self.actions = ['up', 'down', 'left', 'right']
        self.costOfLife = -0.01

    def get_reward(self, state):
        if state[0] == self.x and state[1] == 0:
            return 1
        elif state[0] in self.bed and state[1] == 0:
            return -1
        else:
            return 0


    def get_actions(self, state):
        x, y = state
        possible_actions = []
        if x > 0:
            possible_actions.append('left')
        if x < self.x:
            possible_actions.append('right')
        if y > 0:
            possible_actions.append('down')
        if y < self.y:
            possible_actions.append('up')
        return possible_actions

def q_learning(grid, episodes, steps, alpha=0.1, gamma=0.9, epsilon= 0.2):
    q_table = {(i, j): {action: 0 for action in grid.actions} for i in range(grid.x + 1) for j in range(grid.y + 1)}
    rewPerEpis = []
    #Q = np.zeros([grid.griddy, len(grid.actions)])

    for a in range(episodes):
        state = (0, 0)  # Start at (0, 0)
        totalRev = 0
        # In case the epsilon is high we can decrease it each episode, this way there will be a lot of exploring
        # in the beginning and next to non as the episode numbers increase
        epsilon = epsilon * 0.99
        for b in range(steps):
            reward = 0
            if random.uniform(0, 1) < epsilon:
                action = random.choice(grid.get_actions(state))
            else:
                
                action = max(q_table[state], key=q_table[state].get)

            
            # Slipping is turned on when the user signaled it to be so and the random value generated 
            # is lower than the chance provided
            slipping = False

            if grid.slipSwitch: 
                if random.uniform(0, 1) < grid.slip:
                    slipping = True
            
            # Taking one or two steps based on slipping and possibility. 
            # When slipping would make the agent go out of bound it only takes one step instead of two.
            if slipping:
                if action == 'up':
                    next_state = (state[0], min(state[1] + 2 if state[1] + 2 <= grid.y else state[1] + 1, grid.y))
                elif action == 'down':
                    next_state = (state[0], max(state[1] - 2 if state[1] - 2 >= 0 else state[1] - 1, 0))
                elif action == 'left':
                    next_state = (max(state[0] - 2 if state[0] - 2 >= 0 else state[0] - 1, 0), state[1])
                elif action == 'right':
                    next_state = (min(state[0] + 2 if state[0] + 2 <= grid.x else state[0] + 1, grid.x), state[1])
            else:
                if action == 'up':
                    next_state = (state[0], min(state[1] + 1, grid.y))
                elif action == 'down':
                    next_state = (state[0], max(state[1] - 1, 0))
                elif action == 'left':
                    next_state = (max(state[0] - 1, 0), state[1])
                elif action == 'right':
                    next_state = (min(state[0] + 1, grid.x), state[1])
            # Update reward with the new states assigned reward and the costOfLife
            reward = grid.get_reward(next_state) + grid.costOfLife
            # In case the agent goes to a negative termination zone there is an extra punishment, to further encite it to avoid these zones
            if next_state[0] in grid.bed and next_state[1] == 0:
                reward -= 1

            #Q[state, action] += learning_rate * (reward + discount_factor * np.max(Q[new_state, :]) - Q[state, action])

            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * max(q_table[next_state].values()) - q_table[state][action])
            
            #Update accumulated rewards for this episode
            totalRev += reward
            
            # Quit episode in case the new state is a terminal one
            if next_state[0] == grid.x and next_state[1] == 0 or next_state[0] in grid.bed and next_state[1] == 0:
                print(f"Episode {a + 1} finished in steps: {b + 1} with reward {totalRev}")
                break
            
            # Update the state for the next iteration
            state = next_state  

        rewPerEpis.append(totalRev)
    return q_table, rewPerEpis


# Example usage
grid = GridMDP(4, 2, 0.2, False)
q_table, rewPerEpis = q_learning(grid, episodes=1000, steps=100)

def print_q_table(q_table):
    for state, actions in q_table.items():
        print(f"State: {state}")
        for action, value in actions.items():
            print(f"  Action: {action}, Q-value: {value}")
        print("")

# Print the learned Q-table
print_q_table(q_table)

# Moving averge
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(rewPerEpis, label='Total Reward')
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')

if MOVING:
    moving_average_window = 20
    moving_averages = savgol_filter(rewPerEpis, moving_average_window, 3)
    plt.plot(moving_averages, label=f'Moving Average (Window {moving_average_window})', color='orange')

plt.show()

def get_policy_value(q_table):
    policy = {}
    value = {}
    for state, actions in q_table.items():
        best_action = max(actions, key=actions.get)
        policy[state] = best_action
        value[state] = actions[best_action]
    return policy, value

# Retrieve policy and value from the learned Q-table
policy, value = get_policy_value(q_table)

# Plotting the policy with categorical colors assigned to actions and annotations
def plot_policy(policy):
    x = [state[0] for state in policy.keys()]
    y = [state[1] for state in policy.keys()]
    actions = list(policy.values())

    # Map actions to numerical values
    action_to_value = {'up': 1, 'down': 2, 'left': 3, 'right': 4}
    values = [action_to_value[action] for action in actions]

    fig, ax = plt.subplots()
    ax.set_title('Policy (Best Action)')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Scatter plot with numerical values
    sc = ax.scatter(x, y, c=values, cmap='viridis', s=500, marker='s')

    # Create a colorbar with action annotations
    cbar = plt.colorbar(sc, ticks=[1, 2, 3, 4])
    cbar.ax.set_yticklabels(['up', 'down', 'left', 'right'])

    plt.xticks(np.arange(0, 5, 1))
    plt.yticks(np.arange(0, 3, 1))
    plt.grid(visible=True)
    plt.show()

# Plotting the value function
def plot_value_function(value):
    x = [state[0] for state in value.keys()]
    y = [state[1] for state in value.keys()]
    values = list(value.values())

    fig, ax = plt.subplots()
    ax.set_title('Value Function (Max Q-value)')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    sc = ax.scatter(x, y, c=values, cmap='plasma', s=500, marker='s')
    plt.colorbar(sc)
    plt.xticks(np.arange(0, 5, 1))
    plt.yticks(np.arange(0, 3, 1))
    plt.grid(visible=True)
    plt.show()

# Plotting policy and value function
plot_policy(policy)
plot_value_function(value)




