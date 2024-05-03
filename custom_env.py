import gymnasium as gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class custom_cart_pole(gym.Env):
    def __init__(self):
        self.state = None
        self.fig = None
        self.ax = None
        # Initialize observation space: cart position, cart velocity, pole angle, pole angular velocity
        self.observation_space = spaces.Box(
            low=np.array([-4.8, -np.inf, -0.418, -np.inf]),
            high=np.array([4.8, np.inf, 0.418, np.inf]),
            dtype=np.float32,
        )

        # Initialize action space: 0 (push left), 1 (push right)
        self.action_space = spaces.Discrete(2)

        # Set other properties
        self.gravity = 9.8
        self.cart_mass = 1.0
        self.pole_mass = 0.1
        self.total_mass = self.cart_mass + self.pole_mass
        self.pole_length = 0.5  # Half the pole's length
        self.force_mag = 10.0
        self.tau = 0.02  # Time interval for simulation

        # Initialize state and other variables
        self.state = None
        self.steps_beyond_done = None

    def reset(self, seed=None, options=None):
        # Reset the environment to an initial state
        np.random.seed(seed)
        # Random initialization: position, velocity, angle, angular velocity
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        # Unpack the state
        x, x_dot, theta, theta_dot = self.state

        # Apply force depending on the action
        force = self.force_mag if action == 1 else -self.force_mag

        # Physics calculations
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.pole_mass * self.pole_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.pole_length / 2)
        xacc = temp - (self.pole_mass * self.pole_length * thetaacc * costheta) / self.total_mass

        # Update state based on the physics
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        # Determine if the environment should end (done)
        done = x < -4.8 or x > 4.8 or theta < -0.418 or theta > 0.418

        if done:
            reward = 0.0
        else:
            reward = 1.0  # Reward for each step taken without failure

        return self.state, reward, done, False, {}

    def render(self, mode="human"):
        if self.state is None:
            return
        
        # Unpack the state
        x, x_dot, theta, theta_dot = self.state
        
        if self.fig is None:
            # Create a new figure and axis if not already initialized
            self.fig, self.ax = plt.subplots()
            self.ax.set_xlim(-5, 5)  # Track boundaries
            self.ax.set_ylim(-1, 2)  # Height range

        # Clear previous drawings
        self.ax.cla()
        self.ax.set_xlim(-5, 5)  # Track boundaries
        self.ax.set_ylim(-1, 2)  # Height range

        # Draw the track
        self.ax.plot([-5, 5], [0, 0], 'k-', linewidth=2)
        
        # Draw the cart
        cart = patches.Rectangle((x - 0.5, 0), 1, 0.3,  color='blue')
        self.ax.add_patch(cart)
        
        # Draw the pole
        pole_x = [x, x + np.sin(theta)]
        pole_y = [0.3, 0.3 + np.cos(theta)]
        self.ax.plot(pole_x, pole_y, 'k-', linewidth=2)
        
        # Show the updated plot
        plt.pause(0.01)  # Pause for a short moment to update the plot

    def close(self):
        # Clean up resources, if needed
        pass