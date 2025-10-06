#DO CHANGE THE FILE PATHS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import json
import time
from collections import deque
import traci
import sumolib
from datetime import datetime
import sys
# Set SUMO_HOME environment variable
SUMO_HOME = os.path.abspath(r'E:\SUMO')
os.environ['SUMO_HOME'] = SUMO_HOME

# Add SUMO tools to Python path
if os.path.exists(os.path.join(SUMO_HOME, 'tools')):
    tools_path = os.path.join(SUMO_HOME, 'tools')
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
        print('SUMO not installed or SUMO_HOME environment variable not set correctly.')

# Add SUMO bin to PATH if it exists
bin_path = os.path.join(SUMO_HOME, 'bin')
if os.path.exists(bin_path):
    if 'PATH' in os.environ:
        if bin_path not in os.environ['PATH']:
            os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
    else:
        os.environ['PATH'] = bin_path


# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class ReplayBuffer:#store experience tuples for training so that we can sample from them later preventing overfitting
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, mean


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2


class SAC:
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr=3e-4, 
                 gamma=0.99, tau=0.005, alpha=0.2, auto_entropy=True):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy = auto_entropy
        
        # Actor
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critic
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Entropy temperature
        if self.auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if evaluate:
            _, _, action = self.actor.sample(state)
            action = torch.tanh(action)
        else:
            action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Update Critic
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        new_actions, log_pi, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()
    
    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy else None,
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if self.auto_entropy and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']


class SUMOEnvironment:
    def __init__(self, sumo_cfg, num_envs=3, use_gui=True):
        self.sumo_cfg = sumo_cfg
        self.num_envs = num_envs
        self.use_gui = use_gui
        self.connections = []
        self.traffic_lights = []
    def start_sumo(self, port, seed):
        sumoBinary = sumolib.checkBinary('sumo-gui' if self.use_gui else 'sumo')
        sumo_cmd = [
        sumoBinary,
        "-c", self.sumo_cfg,
        "--start",
        "--quit-on-end",
        "--remote-port", str(port),
        "--seed", str(seed),
        "--no-warnings"
    ]
        traci.start(sumo_cmd, port=port)
        print(sumoBinary)
        return traci.getConnection(label=str(port))
       
    
    def reset(self, seed):
        # Close existing connections
        for conn in self.connections:
            try:
                conn.close()
            except:
                pass
        self.connections = []
        
        # Start parallel environments
        base_port = 8813
        for i in range(self.num_envs):
            conn = self.start_sumo(base_port + i, seed + i)
            self.connections.append(conn)
            
            # Get traffic light IDs from first environment
            if i == 0:
                self.traffic_lights = conn.trafficlight.getIDList()
        
        return self._get_state()
    
    def _get_state(self):
        """Get state from all parallel environments"""
        states = []
        for conn in self.connections:
            state = []
            for tl_id in self.traffic_lights:
                # Queue lengths
                lanes = conn.trafficlight.getControlledLanes(tl_id)
                for lane in lanes:
                    queue = conn.lane.getLastStepHaltingNumber(lane)
                    waiting_time = conn.lane.getWaitingTime(lane)
                    state.extend([queue, waiting_time])
                
                # Current phase
                phase = conn.trafficlight.getPhase(tl_id)
                state.append(phase)
            
            states.append(state)
        
        # Return average state across environments
        return np.mean(states, axis=0)
    
    def step(self, action):
        """Execute action in all parallel environments"""
        rewards = []
        dones = []
        
        for i, conn in enumerate(self.connections):
            # Apply action (set traffic light phases)
            for j, tl_id in enumerate(self.traffic_lights):
                phase = int((action[j] + 1) * (conn.trafficlight.getPhaseCount(tl_id) // 2))
                phase = min(max(0, phase), conn.trafficlight.getPhaseCount(tl_id) - 1)
                conn.trafficlight.setPhase(tl_id, phase)
            
            # Simulation step
            conn.simulationStep()
            
            # Calculate reward (negative of total waiting time)
            total_waiting = 0
            for tl_id in self.traffic_lights:
                lanes = conn.trafficlight.getControlledLanes(tl_id)
                for lane in lanes:
                    total_waiting += conn.lane.getWaitingTime(lane)
            
            reward = -total_waiting / 100.0  # Normalize
            rewards.append(reward)
            
            # Check if simulation ended
            done = conn.simulation.getMinExpectedNumber() <= 0
            dones.append(done)
        
        next_state = self._get_state()
        avg_reward = np.mean(rewards)
        done = all(dones)
        
        return next_state, avg_reward, done
    
    def close(self):
        for conn in self.connections:
            try:
                conn.close()
            except:
                pass


def train_sac_sumo(sumo_cfg_path, num_episodes=1000, max_steps=5000, 
                   num_seeds=5, num_envs=3, results_dir="sac_results"):
    """
    Train SAC on SUMO with multiple seeds and parallel environments
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize environment to get dimensions
    temp_env = SUMOEnvironment(sumo_cfg_path, num_envs=1)
    temp_env.reset(seed=0)
    state_dim = len(temp_env._get_state())
    action_dim = len(temp_env.traffic_lights)
    temp_env.close()
    
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    
    # Training parameters
    batch_size = 256
    buffer_size = 1000000
    
    best_performance = -float('inf')
    best_config = {}
    all_results = []
    
    # Train with different seeds
    for seed_idx in range(num_seeds):
        print(f"\n{'='*60}")
        print(f"Training with Seed {seed_idx}")
        print(f"{'='*60}")
        
        # Set random seeds
        torch.manual_seed(seed_idx)
        np.random.seed(seed_idx)
        
        # Initialize
        env = SUMOEnvironment(sumo_cfg_path, num_envs=num_envs)
        agent = SAC(state_dim, action_dim)
        replay_buffer = ReplayBuffer(buffer_size)
        
        episode_rewards = []
        episode_lengths = []
        start_time = time.time()
        
        for episode in range(num_episodes):
            state = env.reset(seed=seed_idx * 1000 + episode)
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps):
                # Select action
                if len(replay_buffer) < batch_size:
                    action = np.random.uniform(-1, 1, action_dim)
                else:
                    action = agent.select_action(state)
                
                # Environment step
                next_state, reward, done = env.step(action)
                
                # Store transition
                replay_buffer.push(state, action, reward, next_state, done)
                
                # Update agent
                if len(replay_buffer) > batch_size:
                    agent.update(replay_buffer, batch_size)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Seed {seed_idx} | Episode {episode+1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Length: {episode_length}")
        
        env.close()
        
        # Calculate performance metrics
        training_time = time.time() - start_time
        avg_reward = np.mean(episode_rewards[-100:])
        std_reward = np.std(episode_rewards[-100:])
        max_reward = np.max(episode_rewards)
        
        results = {
            'seed': seed_idx,
            'avg_reward_last_100': float(avg_reward),
            'std_reward_last_100': float(std_reward),
            'max_reward': float(max_reward),
            'training_time': training_time,
            'total_episodes': num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
        
        all_results.append(results)
        
        print(f"\nSeed {seed_idx} Results:")
        print(f"  Average Reward (last 100): {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Max Reward: {max_reward:.2f}")
        print(f"  Training Time: {training_time:.2f}s")
        
        # Save model
        model_path = os.path.join(results_dir, f"sac_seed_{seed_idx}.pt")
        agent.save(model_path)
        
        # Check if this is the best performing model
        if avg_reward > best_performance:
            best_performance = avg_reward
            best_config = {
                'seed': seed_idx,
                'model_path': model_path,
                'avg_reward': float(avg_reward),
                'std_reward': float(std_reward),
                'max_reward': float(max_reward),
                'state_dim': state_dim,
                'action_dim': action_dim,
                'num_envs': num_envs,
                'training_time': training_time
            }
            # Save best model separately
            best_model_path = os.path.join(results_dir, "best_model.pt")
            agent.save(best_model_path)
    
    # Save all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"training_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'best_config': best_config,
            'all_results': all_results,
            'hyperparameters': {
                'num_episodes': num_episodes,
                'max_steps': max_steps,
                'num_seeds': num_seeds,
                'num_envs': num_envs,
                'batch_size': batch_size,
                'buffer_size': buffer_size,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'device': str(device)
            }
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"\nBest Performance:")
    print(f"  Seed: {best_config['seed']}")
    print(f"  Average Reward: {best_config['avg_reward']:.2f}")
    print(f"  Model saved at: {best_config['model_path']}")
    print(f"\nAll results saved to: {results_file}")
    
    return best_config, all_results


if __name__ == "__main__":
    # Example usage
    SUMO_CFG = "E:\CODING\gityoutube\test1.sumocfg"  # Update with your SUMO config file
    
    # Train the agent
    best_config, results = train_sac_sumo(
        sumo_cfg_path=SUMO_CFG,
        num_episodes=500,
        max_steps=3000,
        num_seeds=5,
        num_envs=3,
        results_dir="sac_sumo_results"
    )
    
    print("\nTraining finished!")
    print(f"Best model achieved average reward of {best_config['avg_reward']:.2f}")
