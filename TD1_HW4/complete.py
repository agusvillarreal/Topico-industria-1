# Random Walks Dashboard with Fixed Visualization
import numpy as np
import math
import plotly.graph_objects as go
import panel as pn
import param
import pandas as pd

# Enable extensions
pn.extension('plotly')

# Define Vec2d class
class Vec2d(object):
    __slots__ = ['x', 'y']

    def __init__(self, x_or_pair, y = None):
        if y == None:            
            self.x = x_or_pair[0]
            self.y = x_or_pair[1]
        else:
            self.x = x_or_pair
            self.y = y
            
    def __add__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Vec2d(self.x + other[0], self.y + other[1])
        else:
            return Vec2d(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x - other.x, self.y - other.y)
        elif (hasattr(other, "__getitem__")):
            return Vec2d(self.x - other[0], self.y - other[1])
        else:
            return Vec2d(self.x - other, self.y - other)
    
    def get_length(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def rotated(self, angle):        
        cos = math.cos(angle)
        sin = math.sin(angle)
        x = self.x*cos - self.y*sin
        y = self.x*sin + self.y*cos
        return Vec2d(x, y)

# 1. Random Walk Trajectory Functions

# Brownian Motion
def generate_bm_trajectory(num_steps, speed, start_pos_x, start_pos_y, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    trajectory = np.zeros((num_steps, 3))
    current_pos = Vec2d(start_pos_x, start_pos_y)
    
    trajectory[0, 0] = current_pos.x
    trajectory[0, 1] = current_pos.y
    trajectory[0, 2] = 0
    
    for i in range(1, num_steps):
        angle = np.random.uniform(0, 2 * np.pi)
        step = Vec2d(np.cos(angle) * speed, np.sin(angle) * speed)
        current_pos = current_pos + step
        
        trajectory[i, 0] = current_pos.x
        trajectory[i, 1] = current_pos.y
        trajectory[i, 2] = i
    
    return trajectory

# Correlated Random Walk
def generate_crw_trajectory(num_steps, speed, start_pos_x, start_pos_y, cauchy_coef, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    trajectory = np.zeros((num_steps, 3))
    current_pos = Vec2d(start_pos_x, start_pos_y)
    direction = Vec2d(1, 0)
    
    trajectory[0, 0] = current_pos.x
    trajectory[0, 1] = current_pos.y
    trajectory[0, 2] = 0
    
    for i in range(1, num_steps):
        angle = np.random.standard_cauchy() * cauchy_coef
        direction = direction.rotated(angle)
        
        length = direction.get_length()
        if length > 0:
            direction = Vec2d(direction.x / length * speed, direction.y / length * speed)
        
        current_pos = current_pos + direction
        
        trajectory[i, 0] = current_pos.x
        trajectory[i, 1] = current_pos.y
        trajectory[i, 2] = i
    
    return trajectory

# Lévy Flight
def generate_lf_trajectory(num_steps, speed, start_pos_x, start_pos_y, cauchy_coef, levy_exponent, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    trajectory = np.zeros((num_steps, 3))
    current_pos = Vec2d(start_pos_x, start_pos_y)
    direction = Vec2d(1, 0)
    
    trajectory[0, 0] = current_pos.x
    trajectory[0, 1] = current_pos.y
    trajectory[0, 2] = 0
    
    for i in range(1, num_steps):
        angle = np.random.standard_cauchy() * cauchy_coef
        direction = direction.rotated(angle)
        
        step_length = np.random.pareto(levy_exponent) * speed
        
        length = direction.get_length()
        if length > 0:
            direction = Vec2d(direction.x / length * step_length, 
                            direction.y / length * step_length)
        
        current_pos = current_pos + direction
        
        trajectory[i, 0] = current_pos.x
        trajectory[i, 1] = current_pos.y
        trajectory[i, 2] = i
    
    return trajectory

# 2. Metric Calculation Functions

# Path Length
def calculate_pl(trajectory):
    positions = trajectory[:, :2]
    time = trajectory[:, 2]
    
    displacements = np.diff(positions, axis=0)
    step_lengths = np.sqrt(np.sum(displacements**2, axis=1))
    
    pl = np.zeros(len(time))
    pl[1:] = np.cumsum(step_lengths)
    
    return time, pl

# Mean Squared Displacement
def calculate_msd(trajectory):
    positions = trajectory[:, :2]
    n_points = len(positions)
    
    max_tau = min(n_points // 4, 1000)
    
    tau = np.arange(1, max_tau + 1)
    msd = np.zeros(max_tau)
    
    for i, t in enumerate(tau):
        sd = np.sum((positions[t:] - positions[:-t])**2, axis=1)
        msd[i] = np.mean(sd)
    
    return tau, msd

# Turning Angle Distribution
def calculate_tad(trajectory, bin_width=15):
    positions = trajectory[:, :2]
    
    movement_vectors = np.diff(positions, axis=0)
    angles = np.arctan2(movement_vectors[:, 1], movement_vectors[:, 0])
    turning_angles = np.diff(angles)
    turning_angles = np.mod(turning_angles + np.pi, 2 * np.pi) - np.pi
    turning_angles_deg = np.degrees(turning_angles)
    
    bins = np.arange(-180, 181, bin_width)
    angle_bins = (bins[:-1] + bins[1:]) / 2
    
    tad, _ = np.histogram(turning_angles_deg, bins=bins, density=True)
    
    return angle_bins, tad

# 3. Dashboard Implementation

class RandomWalksDashboard(param.Parameterized):
    # Parameter definitions
    rw_type = param.Selector(objects=["BM", "CRW", "LF"], default="CRW")
    num_steps = param.Integer(default=1000, bounds=(100, 5000))
    speed = param.Number(default=5.0, bounds=(1.0, 20.0))
    start_pos_x = param.Integer(default=0, bounds=(-100, 100))
    start_pos_y = param.Integer(default=0, bounds=(-100, 100))
    cauchy_coef = param.Number(default=0.7, bounds=(0.1, 0.9))
    levy_exponent = param.Number(default=1.5, bounds=(1.1, 3.0))
    metric_type = param.Selector(objects=["PL", "MSD", "TAD"], default="MSD")
    
    def __init__(self, **params):
        super(RandomWalksDashboard, self).__init__(**params)
        # Generate an initial trajectory
        self.trajectory = self.generate_trajectory()
        
        # Create initial plots
        self.trajectory_plot = pn.pane.Plotly(self.create_trajectory_plot())
        self.metric_plot = pn.pane.Plotly(self.create_metric_plot())
        
        # Create update button with callback
        self.update_button = pn.widgets.Button(name="Update", button_type="primary")
        self.update_button.on_click(self.update_plots)
    
    def generate_trajectory(self):
        """Generate trajectory based on current parameters"""
        try:
            common_params = {
                'num_steps': self.num_steps,
                'speed': self.speed,
                'start_pos_x': self.start_pos_x,
                'start_pos_y': self.start_pos_y
            }
            
            if self.rw_type == "BM":
                return generate_bm_trajectory(**common_params)
            elif self.rw_type == "CRW":
                return generate_crw_trajectory(**common_params, cauchy_coef=self.cauchy_coef)
            elif self.rw_type == "LF":
                return generate_lf_trajectory(**common_params, cauchy_coef=self.cauchy_coef, 
                                            levy_exponent=self.levy_exponent)
        except Exception as e:
            print(f"Error generating trajectory: {e}")
            # Return a simple default trajectory if there's an error
            return np.zeros((10, 3))
    
    def create_trajectory_plot(self):
        """Create 3D plot of the current trajectory"""
        try:
            fig = go.Figure()
            
            # Plot every nth point for better performance
            plot_every_n = max(1, len(self.trajectory) // 500)
            
            fig.add_trace(go.Scatter3d(
                x=self.trajectory[::plot_every_n, 0],
                y=self.trajectory[::plot_every_n, 1],
                z=self.trajectory[::plot_every_n, 2],
                mode='lines',
                name=self.rw_type,
                line=dict(color='red', width=3)
            ))
            
            fig.update_layout(
                title=f"{self.rw_type} Trajectory",
                scene=dict(
                    xaxis_title="X Position",
                    yaxis_title="Y Position",
                    zaxis_title="Time Step",
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1)
                ),
                height=600,
                width=600,
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            return fig
        except Exception as e:
            print(f"Error creating trajectory plot: {e}")
            # Return a simple empty figure
            fig = go.Figure()
            fig.update_layout(title="Error plotting trajectory")
            return fig
    
    def create_metric_plot(self):
        """Create plot for the selected metric"""
        try:
            # Calculate the selected metric
            if self.metric_type == "PL":
                time, metric_values = calculate_pl(self.trajectory)
                title = "Path Length"
                xlabel, ylabel = "Time Step", "Path Length"
                color = "green"
            elif self.metric_type == "MSD":
                time, metric_values = calculate_msd(self.trajectory)
                title = "Mean Squared Displacement"
                xlabel, ylabel = "Time Lag (τ)", "MSD"
                color = "purple"
            elif self.metric_type == "TAD":
                time, metric_values = calculate_tad(self.trajectory)
                title = "Turning Angle Distribution"
                xlabel, ylabel = "Turning Angle (degrees)", "Probability Density"
                color = "blue"
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time,
                y=metric_values,
                mode='lines',
                name=self.metric_type,
                line=dict(color=color, width=2)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                height=600,
                width=600,
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            return fig
        except Exception as e:
            print(f"Error creating metric plot: {e}")
            # Return a simple empty figure
            fig = go.Figure()
            fig.update_layout(title="Error plotting metric")
            return fig
    
    def update_plots(self, event):
        """Update both plots when the button is clicked"""
        try:
            # Generate new trajectory
            self.trajectory = self.generate_trajectory()
            
            # Update plots
            self.trajectory_plot.object = self.create_trajectory_plot()
            self.metric_plot.object = self.create_metric_plot()
        except Exception as e:
            print(f"Error updating plots: {e}")
    
    @param.depends('rw_type')
    def get_parameter_panel(self):
        """Return dynamic parameter widgets based on RW type"""
        # Common parameters for all RW types
        common_params = [
            pn.Param(self.param.num_steps, widgets={'num_steps': {'type': pn.widgets.IntSlider}}),
            pn.Param(self.param.speed, widgets={'speed': {'type': pn.widgets.FloatSlider}}),
            pn.Param(self.param.start_pos_x, widgets={'start_pos_x': {'type': pn.widgets.IntSlider}}),
            pn.Param(self.param.start_pos_y, widgets={'start_pos_y': {'type': pn.widgets.IntSlider}}),
        ]
        
        # Add type-specific parameters
        if self.rw_type == "BM":
            return pn.Column(*common_params)
        elif self.rw_type == "CRW":
            return pn.Column(
                *common_params,
                pn.Param(self.param.cauchy_coef, widgets={'cauchy_coef': {'type': pn.widgets.FloatSlider}}),
            )
        elif self.rw_type == "LF":
            return pn.Column(
                *common_params,
                pn.Param(self.param.cauchy_coef, widgets={'cauchy_coef': {'type': pn.widgets.FloatSlider}}),
                pn.Param(self.param.levy_exponent, widgets={'levy_exponent': {'type': pn.widgets.FloatSlider}}),
            )
    
    def panel(self):
        """Create and return the complete dashboard layout"""
        dashboard = pn.Column(
            pn.pane.Markdown("# Random Walks Dashboard", align='center'),
            pn.Row(
                # Left column - Controls
                pn.Column(
                    pn.pane.Markdown("## RW Type", margin=(10, 5, 5, 10)),
                    pn.Param(self.param.rw_type, widgets={'rw_type': {'type': pn.widgets.Select}}),
                    pn.pane.Markdown("## Parameters", margin=(10, 5, 5, 10)),
                    self.get_parameter_panel,
                    pn.pane.Markdown("## Metric Type", margin=(10, 5, 5, 10)),
                    pn.Param(self.param.metric_type, widgets={'metric_type': {'type': pn.widgets.Select}}),
                    self.update_button,
                    width=350
                ),
                # Center column - Trajectory
                pn.Column(
                    pn.pane.Markdown("## Trajectory", margin=(10, 5, 5, 10)),
                    self.trajectory_plot,
                    width=600
                ),
                # Right column - Metric
                pn.Column(
                    pn.pane.Markdown("## Metric", margin=(10, 5, 5, 10)),
                    self.metric_plot,
                    width=600
                )
            )
        )
        return dashboard

# Create and display the dashboard
dashboard = RandomWalksDashboard()
dashboard.panel().servable()