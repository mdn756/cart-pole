from custom_env import custom_cart_pole

env = custom_cart_pole()
# Start a new episode
obs, info = env.reset()

done = False
total_reward = 0

# Run the simulation until the episode is done
while not done:
   env.render()  # Render the environment
   #time.sleep(0.05)  # Adjust this value for smoother rendering

   # Get next action and perform it
   action = env.action_space.sample()  # Random action
   obs, rew, done, truncated, info = env.step(action)  # Correct unpacking
   
   total_reward += rew  # Accumulate reward
   
   #print(f"{obs} -> {rew}")

print(f"Total reward: {total_reward}")
#env.close()

import numpy as np
import matplotlib.pyplot as plt
import torch

num_inputs = 4
num_actions = 2

model = torch.nn.Sequential(
    torch.nn.Linear(num_inputs, 128, bias=False, dtype=torch.float32),
    torch.nn.ReLU(),
    torch.nn.Linear(128, num_actions, bias = False, dtype=torch.float32),
    torch.nn.Softmax(dim=1)
)

def run_episode(max_steps_per_episode = 10000,render=False):    
    states, actions, probs, rewards = [],[],[],[]
    state, info = env.reset()

    for _ in range(max_steps_per_episode):
        if render:
            env.render()
        action_probs = model(torch.from_numpy(np.expand_dims(state,0)).float())[0]
        action = np.random.choice(num_actions, p=np.squeeze(action_probs.detach().numpy()))
        nstate, reward, done, truncated, info = env.step(action)
        if done:
            break
        states.append(state)
        actions.append(action)
        probs.append(action_probs.detach().numpy())
        rewards.append(reward)
        state = nstate
    return np.vstack(states), np.vstack(actions), np.vstack(probs), np.vstack(rewards)
# Clean up by closing the environment

eps = 0.0001

def discounted_rewards(rewards,gamma=0.99,normalize=True):
    ret = []
    s = 0
    for r in rewards[::-1]:
        s = r + gamma * s
        ret.insert(0, s)
    if normalize:
        ret = (ret-np.mean(ret))/(np.std(ret)+eps)
    return ret

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_on_batch(x, y):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    optimizer.zero_grad()
    predictions = model(x)
    loss = -torch.mean(torch.log(predictions) * y)
    loss.backward()
    optimizer.step()
    return loss
alpha = 1e-4

history = []
for epoch in range(1000):
    states, actions, probs, rewards = run_episode()
    one_hot_actions = np.eye(2)[actions.T][0]
    gradients = one_hot_actions-probs
    dr = discounted_rewards(rewards)
    gradients *= dr
    target = alpha*np.vstack([gradients])+probs
    train_on_batch(states,target)
    history.append(np.sum(rewards))
    if epoch%100==0:
        print(f"{epoch} -> {np.sum(rewards)}")

plt.plot(history)

_ = run_episode(render=True)


