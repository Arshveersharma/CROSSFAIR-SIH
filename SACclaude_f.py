# Complete SAC + SUMO training script (Windows-safe paths using forward slashes)

import os
import sys
import shutil
import socket
import time
import json
import subprocess
from typing import Optional
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import traci
import sumolib

# -------------------------
# Configuration (Windows-safe using forward slashes)
# -------------------------
PREFERRED_SUMO_HOME = "E:/SUMO"  # set to your SUMO installation if present (forward slashes)
SUMO_CFG_DEFAULT = "C:/Users/Jashan/Desktop/coding/python/test.sumocfg"  # requested config path
SUMO_LOG_DIR = "sumo_logs"  # relative folder for SUMO logs

# If SUMO_HOME exists, set it (only if the directory actually exists)
if os.path.isdir(PREFERRED_SUMO_HOME):
    os.environ["SUMO_HOME"] = PREFERRED_SUMO_HOME

# -------------------------
# Helpers for SUMO binary discovery & ports
# -------------------------
def find_sumo_binary(use_gui=True):
    """
    Try to locate sumo-gui or sumo. Raises FileNotFoundError if not found.
    """
    candidates = ['sumo-gui', 'sumo'] if use_gui else ['sumo', 'sumo-gui']

    # 1) try sumolib.checkBinary
    for c in candidates:
        try:
            return sumolib.checkBinary(c)
        except Exception:
            pass

    # 2) try PATH lookup
    for c in candidates:
        which_path = shutil.which(c)
        if which_path:
            return which_path

    # 3) common Windows install paths
    possible_paths = [
        "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe",
        "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe",
        "C:/Program Files/Eclipse/Sumo/bin/sumo.exe",
        "C:/Program Files/Eclipse/Sumo/bin/sumo-gui.exe",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p

    raise FileNotFoundError("Could not find SUMO binary. Ensure SUMO is installed and SUMO_HOME or PATH is set correctly.")

def is_port_free(port):
    """Return True if TCP port is free on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False

def find_free_port(start=8813, max_tries=200):
    port = start
    for _ in range(max_tries):
        if is_port_free(port):
            return port
        port += 1
    raise RuntimeError("Could not find a free port for SUMO/TraCI")

# -------------------------
# Device info
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    try:
        print("GPU:", torch.cuda.get_device_name(0))
    except Exception:
        pass

# -------------------------
# Replay buffer and networks (SAC)
# -------------------------
class ReplayBuffer:
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
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
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
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
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
        new_actions, log_pi, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
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
        if self.auto_entropy and checkpoint.get('log_alpha') is not None:
            self.log_alpha = checkpoint['log_alpha']

# -------------------------
# SUMO environment with subprocess logging (robust)
# -------------------------
class SUMOEnvironment:
    def __init__(self, sumo_cfg, num_envs=1, use_gui=True, sumo_log_dir=SUMO_LOG_DIR):
        self.sumo_cfg = sumo_cfg
        self.num_envs = num_envs
        self.use_gui = use_gui
        self.connections = []
        self.traffic_lights = []
        self._sumo_processes = []  # list of tuples (port, Popen, log_path, logf)
        self.sumo_log_dir = sumo_log_dir
        os.makedirs(self.sumo_log_dir, exist_ok=True)

    def _make_sumo_cmd(self, sumoBinary: str, port: int, seed: int) -> list:
        return [
            sumoBinary,
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--remote-port", str(port),
            "--seed", str(seed),
            "--no-warnings"
        ]

    def start_sumo(self, port: int, seed: int, label: Optional[str] = None, timeout: float = 15.0):
        sumoBinary = find_sumo_binary(use_gui=self.use_gui)
        if label is None:
            label = f"SUMO_{port}"

        cmd_list = self._make_sumo_cmd(sumoBinary, port, seed)
        log_file_path = os.path.join(self.sumo_log_dir, f"sumo_{port}.log")

        # Clear log file if exists
        open(log_file_path, "wb").close()

        try:
            logf = open(log_file_path, "ab")
        except Exception as e:
            raise RuntimeError(f"Could not open SUMO log file at {log_file_path}: {e}")

        try:
            popen = subprocess.Popen(
                cmd_list,
                stdout=logf,
                stderr=subprocess.STDOUT,
                shell=False
            )
        except Exception as e:
            logf.close()
            raise RuntimeError(f"Failed to launch SUMO process with command: {' '.join(cmd_list)}\nUnderlying error: {e}") from e

        self._sumo_processes.append((port, popen, log_file_path, logf))

        # Try to connect using traci.connect with timeout
        t0 = time.time()
        conn = None
        last_traci_error = None

        while True:
            # If process exited, show log and error
            retcode = popen.poll()
            if retcode is not None:
                try:
                    with open(log_file_path, "r", encoding="utf-8", errors="replace") as lf:
                        log_text = lf.read()
                except Exception:
                    log_text = f"<could not read log file {log_file_path}>"

                raise RuntimeError(
                    f"SUMO process (port={port}) exited prematurely with return code {retcode}.\n"
                    f"Command: {' '.join(cmd_list)}\n"
                    f"Check SUMO log for errors at: {log_file_path}\n\n"
                    f"---- SUMO LOG START ----\n{log_text}\n---- SUMO LOG END ----\n"
                )

            try:
                conn = traci.connect(host="127.0.0.1", port=port, label=label)
                break
            except Exception as e:
                last_traci_error = e
                if time.time() - t0 > timeout:
                    try:
                        with open(log_file_path, "r", encoding="utf-8", errors="replace") as lf:
                            log_text = lf.read()
                    except Exception:
                        log_text = f"<could not read log file {log_file_path}>"

                    raise RuntimeError(
                        f"Timed out ({timeout}s) while waiting for TraCI on port {port}.\n"
                        f"SUMO command: {' '.join(cmd_list)}\n"
                        f"Last traci error: {last_traci_error}\n"
                        f"Check SUMO log at {log_file_path} for details.\n\n"
                        f"---- SUMO LOG START ----\n{log_text}\n---- SUMO LOG END ----\n"
                    ) from last_traci_error
                time.sleep(0.1)

        return conn

    def reset(self, seed):
        # Close existing traci connections
        for conn in self.connections:
            try:
                conn.close()
            except Exception:
                pass
        self.connections = []

        # Terminate any old SUMO processes we started
        for port, proc, log_path, logf in self._sumo_processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
            try:
                logf.close()
            except Exception:
                pass
        self._sumo_processes = []

        base_port = find_free_port(start=8813)
        for i in range(self.num_envs):
            port = base_port + i
            if not is_port_free(port):
                raise RuntimeError(f"Port {port} is not free; can't start SUMO on that port.")
            label = f"SUMO_{port}"
            conn = self.start_sumo(port, seed + i, label=label)
            self.connections.append(conn)
            if i == 0:
                self.traffic_lights = conn.trafficlight.getIDList()

        return self._get_state()

    def _get_state(self):
        states = []
        for conn in self.connections:
            state = []
            for tl_id in self.traffic_lights:
                lanes = conn.trafficlight.getControlledLanes(tl_id)
                for lane in lanes:
                    queue = conn.lane.getLastStepHaltingNumber(lane)
                    waiting_time = conn.lane.getWaitingTime(lane)
                    state.extend([queue, waiting_time])
                phase = conn.trafficlight.getPhase(tl_id)
                state.append(phase)
            states.append(state)
        return np.mean(states, axis=0)

    def step(self, action):
        rewards = []
        dones = []

        for i, conn in enumerate(self.connections):
            for j, tl_id in enumerate(self.traffic_lights):
                try:
                    phase_count = conn.trafficlight.getPhaseCount(tl_id)
                    phase = int((action[j] + 1) * (phase_count // 2))
                    phase = min(max(0, phase), phase_count - 1)
                    conn.trafficlight.setPhase(tl_id, phase)
                except Exception:
                    pass

            conn.simulationStep()

            total_waiting = 0
            for tl_id in self.traffic_lights:
                lanes = conn.trafficlight.getControlledLanes(tl_id)
                for lane in lanes:
                    total_waiting += conn.lane.getWaitingTime(lane)

            reward = -total_waiting / 100.0
            rewards.append(reward)

            done = conn.simulation.getMinExpectedNumber() <= 0
            dones.append(done)

        next_state = self._get_state()
        avg_reward = np.mean(rewards) if rewards else 0.0
        done = all(dones) if dones else False

        return next_state, avg_reward, done

    def close(self):
        for conn in self.connections:
            try:
                conn.close()
            except Exception:
                pass
        self.connections = []
        for port, proc, log_path, logf in self._sumo_processes:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
            try:
                logf.close()
            except Exception:
                pass
        self._sumo_processes = []
        self._sumo_processes = []

# -------------------------
# Training loop
# -------------------------
def train_sac_sumo(sumo_cfg_path, num_episodes=1000, max_steps=5000,
                   num_seeds=5, num_envs=3, results_dir="sac_results", use_gui=True):
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(sumo_cfg_path):
        raise FileNotFoundError(f"SUMO config file not found at: {sumo_cfg_path}")

    # Initialize environment to determine state/action dims
    temp_env = SUMOEnvironment(sumo_cfg_path, num_envs=1, use_gui=use_gui)
    temp_env.reset(seed=0)
    state_dim = len(temp_env._get_state())
    action_dim = len(temp_env.traffic_lights)
    temp_env.close()

    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    # Training hyperparams
    batch_size = 256
    buffer_size = 1000000

    best_performance = -float('inf')
    best_config = {}
    all_results = []

    for seed_idx in range(num_seeds):
        print("\n" + "="*60)
        print(f"Training with Seed {seed_idx}")
        print("="*60)

        torch.manual_seed(seed_idx)
        np.random.seed(seed_idx)

        env = SUMOEnvironment(sumo_cfg_path, num_envs=1, use_gui=use_gui)
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
                if len(replay_buffer) < batch_size:
                    action = np.random.uniform(-1, 1, action_dim)
                else:
                    action = agent.select_action(state)

                next_state, reward, done = env.step(action)
                replay_buffer.push(state, action, reward, next_state, done)

                if len(replay_buffer) > batch_size:
                    agent.update(replay_buffer, batch_size)

                episode_reward += reward
                episode_length += 1
                state = next_state

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Seed {seed_idx} | Episode {episode+1}/{num_episodes} | Avg Reward: {avg_reward:.2f} | Length: {episode_length}")

        env.close()

        training_time = time.time() - start_time
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 1 else 0.0
        std_reward = np.std(episode_rewards[-100:]) if len(episode_rewards) >= 1 else 0.0
        max_reward = np.max(episode_rewards) if len(episode_rewards) >= 1 else 0.0

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

        model_path = os.path.join(results_dir, f"sac_seed_{seed_idx}.pt")
        agent.save(model_path)

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
            best_model_path = os.path.join(results_dir, "best_model.pt")
            agent.save(best_model_path)

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

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    if best_config:
        print("\nBest Performance:")
        print(f"  Seed: {best_config.get('seed')}")
        print(f"  Average Reward: {best_config.get('avg_reward', 0.0):.2f}")
        print(f"  Model saved at: {best_config.get('model_path')}")
    print(f"\nAll results saved to: {results_file}")

    return best_config, all_results

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    SUMO_CFG = SUMO_CFG_DEFAULT
    USE_GUI = True  # set to False for headless SUMO

    best_config, results = train_sac_sumo(
        sumo_cfg_path=SUMO_CFG,
        num_episodes=500,
        max_steps=3000,
        num_seeds=5,
        num_envs=1,
        results_dir="sac_sumo_results",
        use_gui=USE_GUI
    )

    print("\nTraining finished!")
    if best_config:
        print(f"Best model achieved average reward of {best_config['avg_reward']:.2f}")
