import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ENV_ID         = "CartPole-v1"
TOTAL_STEPS    = 1_000_000
ROLLOUT_LEN    = 2048        # steps per data collection
UPDATE_EPOCHS  = 10          # how many passes on the rollout
MINIBATCH_SIZE = 256
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_EPS       = 0.2
ENT_COEF       = 0.01
VF_COEF        = 0.5
LR             = 3e-4
MAX_GRAD_NORM  = 0.5
SEED           = 42
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu" 

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.pi = nn.Linear(128, act_dim)  # policy head (logits)
        self.v  = nn.Linear(128, 1)        # value head  (scalar)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.pi(h)
        value  = self.v(h).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action  = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, value

    def evaluate_actions(self, x, actions):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy, value

# ---------------- rollout buffer ----------------
class RolloutBuffer:
    def __init__(self, size, obs_dim, device):
        self.size = size
        self.device = device
        self.obs      = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.actions  = torch.zeros(size, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(size, dtype=torch.float32, device=device)
        self.rewards  = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones    = torch.zeros(size, dtype=torch.float32, device=device)
        self.values   = torch.zeros(size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns    = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.values[self.ptr]   = value
        self.ptr += 1

    def compute_gae(self, last_value, gamma, lam):
        gae = 0.0
        for t in reversed(range(self.size)):
            nonterminal = 1.0 - self.dones[t]
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            gae = delta + gamma * lam * nonterminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def normalize_advantages(self, eps=1e-8):
        a = self.advantages
        self.advantages = (a - a.mean()) / (a.std() + eps)

    def minibatches(self, batch_size, shuffle=True):
        idxs = np.arange(self.size)
        if shuffle:
            np.random.shuffle(idxs)
        for start in range(0, self.size, batch_size):
            j = idxs[start:start + batch_size]
            yield (self.obs[j], self.actions[j], self.logprobs[j],
                   self.advantages[j], self.returns[j])

# -------------------- training --------------------
def train():
    # seeding (python, numpy, torch) — minimal but helpful
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=SEED)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device(DEVICE)
    net = ActorCritic(obs_dim, act_dim).to(device)
    optim_ = optim.Adam(net.parameters(), lr=LR)

    steps_done = 0
    ep_return = 0.0
    returns_log = []

    while steps_done < TOTAL_STEPS:
        # 1) collect rollout
        buf = RolloutBuffer(ROLLOUT_LEN, obs_dim, device)
        for _ in range(ROLLOUT_LEN):
            x = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                action, logp, ent, value = net.act(x)
            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            buf.add(
                obs=torch.as_tensor(obs, dtype=torch.float32, device=device),
                action=action.detach(),
                logprob=logp.detach(),
                reward=torch.tensor(reward, dtype=torch.float32, device=device),
                done=torch.tensor(float(done), dtype=torch.float32, device=device),
                value=value.detach(),
            )

            ep_return += reward
            steps_done += 1
            obs = next_obs
            if done:
                returns_log.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

            if steps_done >= TOTAL_STEPS:
                break

        # bootstrap V(s_last) and compute GAE/returns once per rollout
        x_last = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            _, last_v = net.forward(x_last)
        buf.compute_gae(last_value=last_v.detach(), gamma=GAMMA, lam=GAE_LAMBDA)
        buf.normalize_advantages()

        # 2) PPO update (multiple epochs, minibatches)
        for _ in range(UPDATE_EPOCHS):
            for mb_obs, mb_act, mb_logp_old, mb_adv, mb_ret in buf.minibatches(MINIBATCH_SIZE):
                new_logp, entropy, value = net.evaluate_actions(mb_obs, mb_act)
                ratio = torch.exp(new_logp - mb_logp_old)  # π_new/π_old

                # clipped surrogate objective (actor)
                unclipped = ratio * mb_adv
                clipped   = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                # critic loss
                value_loss = (value - mb_ret).pow(2).mean()

                # entropy bonus (maximize entropy -> subtract in loss)
                entropy_bonus = entropy.mean()

                loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy_bonus

                optim_.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
                optim_.step()
                torch.save(net.state_dict(), "ppo_cartpole.pth")


    env.close()
    print("Training complete. Last 10 episode returns:", returns_log[-10:])
        # ---------- evaluation run with rendering ----------
    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset(seed=0)
    net.eval()
    done = False
    ep_return = 0
    while not done:
        x = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            logits, _ = net(x)
            action = torch.argmax(logits).item()  # pick best action (no sampling)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += reward
        done = terminated or truncated
    print("Evaluation return:", ep_return)
    env.close()


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train()
