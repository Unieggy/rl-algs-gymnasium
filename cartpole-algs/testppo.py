import gymnasium as gym
import torch
from ppo import ActorCritic, ENV_ID, DEVICE  # reuse your class & constants

# create env with render
env = gym.make(ENV_ID, render_mode="human")
obs, _ = env.reset(seed=0)

# rebuild the same net structure
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
net = ActorCritic(obs_dim, act_dim).to(torch.device(DEVICE))
net.load_state_dict(torch.load("ppo_cartpole.pth"))  # load saved weights
net.eval()

# run one episode
done = False
ep_return = 0
while not done:
    x = torch.from_numpy(obs).float().to(DEVICE)
    with torch.no_grad():
        logits, _ = net(x)
        action = torch.argmax(logits).item()
    obs, reward, terminated, truncated, _ = env.step(action)
    ep_return += reward
    done = terminated or truncated

print("Watched episode return:", ep_return)
env.close()
