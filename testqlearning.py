import numpy as np
import gymnasium as gym

# ---- SAME discretization as training (copy exactly) ----
OBS_LOW  = np.array([-2.4, -3.0, -0.2095, -3.5], dtype=np.float32)
OBS_HIGH = np.array([ 2.4,  3.0,  0.2095,  3.5], dtype=np.float32)
N_BINS   = np.array([9, 9, 15, 15], dtype=int)

def create_bins():
    bins = []
    for lo, hi, n in zip(OBS_LOW, OBS_HIGH, N_BINS):
        edges = np.linspace(lo, hi, n - 1, dtype=np.float32)
        bins.append(edges)
    return bins

BINS = create_bins()

def discretize(obs: np.ndarray) -> tuple:
    clipped = np.clip(obs, OBS_LOW, OBS_HIGH)
    idxs = []
    for val, edges in zip(clipped, BINS):
        idxs.append(int(np.digitize(val, edges)))
    return tuple(idxs)

# ---- LOAD the trained Q-table ----
Q = np.load("qtable_cartpole.npy")   # shape should be (9,9,15,15,2)
print("Loaded Q shape:", Q.shape)

# ---- WATCH: greedy policy (no exploration) ----
def watch(num_episodes=10, seed=0):
    env = gym.make("CartPole-v1", render_mode="human")
    for k in range(num_episodes):
        obs, info = env.reset(seed=seed + k)
        s = discretize(obs)
        done = False
        ep_ret = 0
        while not done:
            # Q[s] is a length-2 vector of action values
            a = int(np.argmax(Q[s]))                 # pick the best action greedily
            obs, r, terminated, truncated, info = env.step(a)
            s = discretize(obs)
            ep_ret += r
            done = terminated or truncated
        print(f"[eval] episode {k+1}: return={ep_ret}")
    env.close()

if __name__ == "__main__":
    watch(num_episodes=10, seed=0)
