import os
import sys

# Set SUMO_HOME environment variable
SUMO_HOME = os.path.abspath('D:/SumoTest')
os.environ['SUMO_HOME'] = SUMO_HOME

# Add SUMO tools to Python path
if os.path.exists(os.path.join(SUMO_HOME, 'tools')):
    tools_path = os.path.join(SUMO_HOME, 'tools')
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)

# Add SUMO bin to PATH if it exists
bin_path = os.path.join(SUMO_HOME, 'bin')
if os.path.exists(bin_path):
    if 'PATH' in os.environ:
        if bin_path not in os.environ['PATH']:
            os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
    else:
        os.environ['PATH'] = bin_path

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Configuration constants
USE_GUI = True  # Set to True to see the SUMO-GUI visualization
SUMO_CFG_FILE = "D:/SumoTest/test.sumocfg"


class SumoTrafficEnv(gym.Env):
    """
    SUMO environment (single intersection). Uses Gymnasium API:
      - reset() -> (obs, info)
      - step(action) -> (obs, reward, terminated, truncated, info)
    """

    def __init__(self, sumo_cfg_file, gui=False, max_steps=30, delta_time=5):
        super(SumoTrafficEnv, self).__init__()

        self.sumo_cfg_file = sumo_cfg_file
        self.gui = gui
        self.max_steps = max_steps
        self.delta_time = delta_time
        self.current_step = 0

        self.sumo_cmd = None
        self.connection_label = "default"

        self.traffic_light_id = None
        self.traffic_light_phases = []
        self.incoming_lanes = []

        self.observation_space = None
        self.action_space = None
        
        # Just initialize SUMO once and set up spaces
        self._init_sumo()
        self._setup_spaces()

    def _init_sumo(self):
        cfg = self.sumo_cfg_file
        if not os.path.isabs(cfg):
            cfg = os.path.abspath(cfg)
        print(f"[DEBUG] SUMO config path to check: {cfg}")

        if not os.path.exists(cfg):
            parent_dir = os.path.dirname(cfg) or "."
            listing = os.listdir(parent_dir) if os.path.exists(parent_dir) else []
            raise FileNotFoundError(
                f"SUMO config not found: {cfg}\n\n"
                f"Directory listing of {parent_dir}:\n{listing}\n\n"
                "Make sure the filename (ends with .sumocfg) and path are correct."
            )

        # Use specific SUMO location
        sumo_path = os.path.join("D:", "SumoTest", "bin")
        sumo_binary = os.path.join(sumo_path, "sumo-gui.exe" if self.gui else "sumo.exe")

        self.sumo_cmd = [sumo_binary, "-c", cfg, "--waiting-time-memory", "1000", "--start", "--begin", "0", "--step-length", "0.1"]
        print("Starting SUMO with command:", " ".join(self.sumo_cmd))

        try:
            traci.start(self.sumo_cmd, label=self.connection_label)
            traci.switch(self.connection_label)
        except Exception as e:
            raise RuntimeError(
                "Failed to start SUMO/TraCI. Run the command printed above manually in a terminal to see SUMO's error output.\n"
                f"Original error: {e}"
            )

        traffic_lights = traci.trafficlight.getIDList()
        if len(traffic_lights) == 0:
            traci.close()
            raise ValueError("No traffic lights found in the simulation (check your net / tls configuration).")

        self.traffic_light_id = traffic_lights[0]
        logic = traci.trafficlight.getAllProgramLogics(self.traffic_light_id)[0]
        self.traffic_light_phases = [i for i in range(len(logic.phases))]
        self.incoming_lanes = list(set(traci.trafficlight.getControlledLanes(self.traffic_light_id)))

        print(f"Traffic Light ID: {self.traffic_light_id}, phases: {self.traffic_light_phases}")
        print(f"Controlled lanes: {self.incoming_lanes}")

    def _setup_spaces(self):
        n_lanes = len(self.incoming_lanes)
        if n_lanes == 0:
            self.observation_space = spaces.Box(low=np.zeros(2, dtype=np.float32),
                                                high=np.array([50.0, 20.0], dtype=np.float32),
                                                dtype=np.float32)
            self.action_space = spaces.Discrete(1)
            return

        obs_low = np.zeros(n_lanes * 2, dtype=np.float32)
        obs_high = np.concatenate([
            np.full(n_lanes, 50.0, dtype=np.float32),
            np.full(n_lanes, 20.0, dtype=np.float32)
        ])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.traffic_light_phases))

        print(f"Observation space: {self.observation_space}")
        print(f"Action space: {self.action_space}")

    def _close_sumo(self):
        try:
            traci.switch(self.connection_label)
            traci.close()
        except Exception:
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        try:
            traci.switch(self.connection_label)
            traci.load(["-c", self.sumo_cfg_file])
        except:
            # If connection is lost, reinitialize
            self._init_sumo()
            
        self.current_step = 0
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        traci.trafficlight.setPhase(self.traffic_light_id, int(action))

        for _ in range(self.delta_time):
            traci.simulationStep()
            self.current_step += 1
            
            if self.gui:
                import time
                time.sleep(0.05)  # Smaller delay for smoother visualization

        obs = self._get_observation()
        reward = self._calculate_reward()

        # Only terminate on max steps, keep running even if no vehicles temporarily
        terminated = self.current_step >= self.max_steps
        truncated = False  # custom truncation condition can be added
        info = {
            'step': self.current_step,
            'vehicles_in_sim': traci.simulation.getMinExpectedNumber()
        }

        if terminated:
            self._close_sumo()

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        vehicle_counts = []
        avg_speeds = []

        for lane_id in self.incoming_lanes:
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
            vehicle_counts.append(min(vehicle_count, 50.0))
            avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            avg_speeds.append(min(avg_speed, 20.0))

        if len(vehicle_counts) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        obs = np.array(vehicle_counts + avg_speeds, dtype=np.float32)
        return obs

    def _calculate_reward(self):
        total_waiting_time = 0.0
        for lane_id in self.incoming_lanes:
            waiting_time = traci.lane.getWaitingTime(lane_id)
            total_waiting_time += waiting_time

        reward = -total_waiting_time / 100.0
        return reward

    def close(self):
        self._close_sumo()


def main():
    print("=== SUMO RL Traffic Control - Phase 1 ===")

    # Absolute path to your SUMO config file
    SUMO_CFG_FILE = r"D:\SumoTest\test.sumocfg"
    USE_GUI = True  # Set to True to open SUMO-GUI visualization

    print(f"[INFO] Using SUMO config: {SUMO_CFG_FILE}")

    print("\nCreating environment...")
    env = SumoTrafficEnv(
        sumo_cfg_file=SUMO_CFG_FILE,
        gui=USE_GUI,
        max_steps=30,
        delta_time=5
    )

    print("\nChecking environment...")
    try:
        check_env(env)
        print("Environment check passed!")
    except Exception as e:
        print(f"Environment check failed: {e}")
        return

    print("\nStarting simulation with random actions...")
    print("The SUMO-GUI window will open now.")
    print("1. Wait for the SUMO-GUI window to fully load")
    print("2. Use the slider at the top to adjust simulation speed")
    print("3. The simulation will run continuously with random traffic light changes")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    input("Press Enter when you're ready to start...")

    # Run for longer to see the effects
    for episode in range(5):  # Run 5 episodes
        print(f"\nEpisode {episode + 1}")
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(50):  # 50 steps per episode
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            print(f"Step {step}: action={action}, reward={reward:.2f}")
            
            if terminated or truncated:
                print(f"Episode {episode + 1} finished after {step + 1} steps")
                break
        
        print(f"Episode {episode + 1} total reward: {episode_reward:.2f}")
    
    # Don't close the environment here to keep simulation running

    print("Starting PPO Training...")
    env = SumoTrafficEnv(
        sumo_cfg_file=SUMO_CFG_FILE,
        gui=False,
        max_steps=30,
        delta_time=5
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=16,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    total_timesteps = 2000
    print(f"Training for {total_timesteps} timesteps...")

    try:
        model.learn(total_timesteps=total_timesteps)
        model.save("ppo_traffic_control_phase1")
        print("Model saved as 'ppo_traffic_control_phase1.zip'")

        print("\nTesting trained model...")
        obs, info = env.reset()
        total_reward = 0.0

        for step in range(50):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if step % 10 == 0:
                print(f"Test step {step}: action={action}, reward={reward:.2f}")
            if done:
                break

        print(f"Total test reward: {total_reward:.2f}")

    except Exception as e:
        import traceback
        print(f"Training failed: {e}")
        traceback.print_exc()
    finally:
        env.close()

    print("\n=== Phase 1 Complete ===")


if __name__ == "__main__":
    main()
