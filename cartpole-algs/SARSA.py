import numpy as np
import gymnasium as gym
ALPHA = 0.07      # learning rate (update size)
GAMMA = 0.99      # discount factor
EPS_START = 1.0   # start fully exploring
EPS_MIN = 0.05    # keep some exploration
EPS_DECAY = 0.997 # per-episode decay: eps <- max(EPS_MIN, eps * EPS_DECAY)

N_EPISODES = 150000 # number of training episodes
env=gym.make("CartPole-v1")
obs,info=env.reset(seed=42)
n_actions=env.action_space.n

obs_low=np.array([-2.4,-3.0,-0.2095,-3.5],dtype=np.float32)
obs_high=np.array([2.4,3.0,0.2095,3.5],dtype=np.float32)

n_bins=np.array([9,9,15,15],dtype=int)
def create_bins():
    bins=[]
    for lo,hi,n in zip(obs_low,obs_high,n_bins):
        edges=np.linspace(lo,hi,n-1,dtype=np.float32)
        bins.append(edges)
    return bins

bins=create_bins()

def discretize(obs:np.ndarray):
    clipped=np.clip(obs,obs_low,obs_high)
    idxs=[]
    for val,edges in zip(clipped,bins):
        idx=int(np.digitize(val,edges))
        idxs.append(idx)
    return tuple(idxs)
q_shape=tuple(n_bins.tolist())+(n_actions,)
Q=np.zeros(q_shape,dtype=np.float32)

rng=np.random.default_rng(42)
def eps_greedy_action(Q:np.ndarray,s:tuple,eps:float):
    if rng.random()<=eps:
        return rng.integers(0,n_actions)
    return int(np.argmax(Q[s]))

def train():
    eps=EPS_START
    returns_log=[]
    for ep in range(1,N_EPISODES+1):
        obs,info=env.reset()
        s=discretize(obs)
        a=eps_greedy_action(Q,s,eps)
        ep_return=0.0
        done=False
        while not done:
            obs_next,r,terminated,truncated,info=env.step(a)
            done=terminated or truncated
            s_next=discretize(obs_next)
            ep_return += float(r)
            if not done:
                a_next=eps_greedy_action(Q,s_next,eps)
                best_next=Q[s_next+(a_next,)]    
                td_target=r+GAMMA*best_next
                td_error=td_target-Q[s+(a,)]
                Q[s+(a,)]+=ALPHA*td_error
                s, a = s_next, a_next
            else:
                td_target=r
                td_error  = td_target - Q[s + (a,)]
                Q[s + (a,)] += ALPHA * td_error
                
        eps=max(EPS_MIN,eps*EPS_DECAY)
        returns_log.append(ep_return)
        if ep%1000==0:
            avg100=np.mean(returns_log[-100:]
            )
            print(f"Episode {ep:4d} | eps={eps:.3f} | return={ep_return:.1f} | avg100={avg100:.1f}")
    return returns_log
            
if __name__ == "__main__":
    returns = train()
    print("Training done. Last 10 returns:", returns[-10:])
    np.save("qtable_cartpoleSarsa.npy", Q)
    print("Saved Q-table to qtable_cartpoleSarsa.npy")
