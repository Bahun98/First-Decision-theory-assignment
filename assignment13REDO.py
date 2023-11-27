import random
import numpy as np

transitions = {
    's0': {
        'a': {'s0': 0.5, 's1': 0.5},
        'b': {'s2': 1.0}
    },
    's1': {
        'a': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
        'b': {'s1': 0.95, 's2': 0.05}
    },
    's2': {
        'a': {'s0': 0.4, 's2': 0.6},
        'b': {'s0': 0.3, 's1': 0.3, 's2': 0.4}
    }
}

rewards = {
    's1': {'a': {'s0': 5}},
    's2': {'b': {'s0': -1}}
}

def next_state(current_state):
    possible_actions = transitions.get(current_state)

    if random.uniform(0, 1) <= 0.5:
        chosen_action = 'a'
    else:
        chosen_action = 'b'

    probabilities = possible_actions[chosen_action]

    states = list(probabilities.keys())
    weights = list(probabilities.values())
    
    chosen_state = random.choices(states, weights=weights)[0]

    if current_state == 's1' and chosen_state == 's0':
        reward = 5
    elif current_state == 's2' and chosen_state == 's0':
        reward = -1
    else:
       reward = 0

    print(f"The chosen state is: {chosen_state} with the reward: {reward}")
    return reward

#Testing s1
current_state = 's1'
rewards = []
for i in range(10):
    rewards.append(next_state(current_state))

print(rewards)

# Testing s2
current_state = 's2'

rewards = []
for i in range(10):
    
    rewards.append(next_state(current_state))
print(rewards)

#Proof of code
def next_state(current_state):
    possible_actions = transitions.get(current_state)

    chosen_action = 'a'

    probabilities = possible_actions[chosen_action]

    states = list(probabilities.keys())
    weights = list(probabilities.values())
    
    chosen_state = random.choices(states, weights=weights)[0]

    if current_state == 's1' and chosen_state == 's0':
        reward = 5
    elif current_state == 's2' and chosen_state == 's0':
        reward = -1
    else:
       reward = 0

    return reward
print('\nResult of proof: ')
current_state = 's1'
rewards = []
for i in range(10000):
    rewards.append(next_state(current_state)*2)
print(np.average(rewards))


