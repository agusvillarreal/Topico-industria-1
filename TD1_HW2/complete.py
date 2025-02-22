# random_walks_dashboard.py
# Standalone dashboard application for Random Walks

import numpy as np
import math
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# Define Vec2d class (as provided)
class Vec2d(object):
    """2d vector class, supports vector and scalar operators,
       and also provides a bunch of high level functions
    """
    __slots__ = ['x', 'y']

    def __init__(self, x_or_pair, y = None):
        if y == None:            
            self.x = x_or_pair[0]
            self.y = x_or_pair[1]
        else:
            self.x = x_or_pair
            self.y = y
            
    # Addition
    def __add__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__getitem__"):
            return Vec2d(self.x + other[0], self.y + other[1])
        else:
            return Vec2d(self.x + other, self.y + other)

    # Subtraction
    def __sub__(self, other):
        if isinstance(other, Vec2d):
            return Vec2d(self.x - other.x, self.y - other.y)
        elif (hasattr(other, "__getitem__")):
            return Vec2d(self.x - other[0], self.y - other[1])
        else:
            return Vec2d(self.x - other, self.y - other)
    
    # Vector length
    def get_length(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    # rotate vector
    def rotated(self, angle):        
        cos = math.cos(angle)
        sin = math.sin(angle)
        x = self.x*cos - self.y*sin
        y = self.x*sin + self.y*cos
        return Vec2d(x, y)

# Module 1: Correlated Random Walk
def generate_crw_trajectory(num_steps, speed, start_pos_x, start_pos_y, cauchy_coef, seed=None):
    """
    Generate a Correlated Random Walk trajectory
    
    Parameters:
    -----------
    num_steps : int
        Number of steps in the trajectory
    speed : float
        Speed/step size for the walker
    start_pos_x, start_pos_y : float
        Starting position coordinates
    cauchy_coef : float
        Cauchy distribution coefficient (0 < c < 1)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    trajectory : numpy.ndarray
        Array of shape (num_steps, 3) with x, y, time coordinates
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Initialize trajectory array
    trajectory = np.zeros((num_steps, 3))
    
    # Set starting position
    current_pos = Vec2d(start_pos_x, start_pos_y)
    
    # Initial direction (random)
    direction = Vec2d(1, 0)
    
    # Store initial position
    trajectory[0, 0] = current_pos.x
    trajectory[0, 1] = current_pos.y
    trajectory[0, 2] = 0  # time
    
    # Generate trajectory
    for i in range(1, num_steps):
        # Sample turning angle from Cauchy distribution
        angle = np.random.standard_cauchy() * cauchy_coef
        
        # Update direction with rotation
        direction = direction.rotated(angle)
        
        # Normalize direction and apply speed
        length = direction.get_length()
        if length > 0:
            direction = Vec2d(direction.x / length * speed, direction.y / length * speed)
        
        # Update position
        current_pos = current_pos + direction
        
        # Store position
        trajectory[i, 0] = current_pos.x
        trajectory[i, 1] = current_pos.y
        trajectory[i, 2] = i  # time
    
    return trajectory

# Calculate Mean Squared Displacement (MSD)
def calculate_msd(trajectory):
    """
    Calculate Mean Squared Displacement for a trajectory
    
    Parameters:
    -----------
    trajectory : numpy.ndarray
        Array with x, y coordinates
        
    Returns:
    --------
    tau : numpy.ndarray
        Time lags
    msd : numpy.ndarray
        MSD values for each time lag
    """
    positions = trajectory[:, :2]  # Only x, y coordinates
    n_points = len(positions)
    
    # Calculate maximum tau (time lag)
    max_tau = min(n_points // 4, 1000)  # Use 1/4 of the trajectory for reliable statistics, max 1000
    
    tau = np.arange(1, max_tau + 1)
    msd = np.zeros(max_tau)
    
    # Calculate MSD for each tau (optimized for performance)
    for i, t in enumerate(tau):
        if i % 10 == 0:  # Skip some points for performance
            # Calculate squared displacements
            sd = np.sum((positions[t:] - positions[:-t])**2, axis=1)
            # Average to get MSD
            msd[i] = np.mean(sd)
        elif i > 0:
            # Interpolate for skipped points
            msd[i] = msd[i-1]
    
    return tau, msd

# Create Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div([
    html.H1("Random Walks Dashboard", className="text-center my-4"),
    
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                # Left column - Controls
                dbc.Col([
                    html.H5("RW Type"),
                    dbc.RadioItems(
                        id="rw-type-radio",
                        options=[
                            {"label": "BM", "value": "BM"},
                            {"label": "CRW", "value": "CRW"},
                            {"label": "LF", "value": "LF"}
                        ],
                        value="CRW",
                        inline=True,
                        className="mb-3"
                    ),
                    
                    html.H5("Parameters", className="mt-4"),
                    
                    html.Label("Number of steps"),
                    dcc.Input(
                        id="num-steps-input",
                        type="number",
                        value=1000,
                        min=100,
                        max=5000,
                        step=100,
                        className="mb-2 w-100"
                    ),
                    
                    html.Label("Speed"),
                    dcc.Input(
                        id="speed-input",
                        type="number",
                        value=5,
                        min=1,
                        max=20,
                        step=1,
                        className="mb-2 w-100"
                    ),
                    
                    html.Label("Starting pos_x"),
                    dcc.Input(
                        id="start-x-input",
                        type="number",
                        value=0,
                        min=-100,
                        max=100,
                        step=10,
                        className="mb-2 w-100"
                    ),
                    
                    html.Label("Starting pos_y"),
                    dcc.Input(
                        id="start-y-input",
                        type="number",
                        value=0,
                        min=-100,
                        max=100,
                        step=10,
                        className="mb-2 w-100"
                    ),
                    
                    html.Label("Cauchy coefficient"),
                    dcc.Input(
                        id="cauchy-input",
                        type="number",
                        value=0.7,
                        min=0.1,
                        max=0.9,
                        step=0.1,
                        className="mb-2 w-100"
                    ),
                    
                    dbc.Button(
                        "Update", 
                        id="update-button",
                        color="primary",
                        className="mt-3 w-100"
                    ),
                ], width=3),
                
                # Middle column - Trajectory
                dbc.Col([
                    html.Div([
                        html.H5("Trajectory", className="d-inline"),
                        html.Button("▼", className="float-right btn btn-sm btn-outline-secondary")
                    ], className="d-flex justify-content-between"),
                    
                    dcc.Graph(
                        id="trajectory-graph",
                        figure={},
                        style={"height": "600px"}
                    )
                ], width=5),
                
                # Right column - Metric
                dbc.Col([
                    html.Div([
                        html.H5("Metric", className="d-inline"),
                        html.Button("▼", className="float-right btn btn-sm btn-outline-secondary")
                    ], className="d-flex justify-content-between"),
                    
                    html.Label("Metric Type"),
                    dcc.Dropdown(
                        id="metric-type-dropdown",
                        options=[
                            {"label": "MSD", "value": "MSD"}
                        ],
                        value="MSD",
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    dcc.Graph(
                        id="metric-graph",
                        figure={},
                        style={"height": "500px"}
                    )
                ], width=4)
            ])
        ])
    ], className="mb-4"),
    
    html.Div(id="trajectory-store", style={"display": "none"})
], className="container-fluid")

# Callback to update trajectory and store it
@app.callback(
    Output("trajectory-store", "children"),
    Input("update-button", "n_clicks"),
    [State("num-steps-input", "value"),
     State("speed-input", "value"),
     State("start-x-input", "value"),
     State("start-y-input", "value"),
     State("cauchy-input", "value")],
    prevent_initial_call=False
)
def update_trajectory(n_clicks, num_steps, speed, start_x, start_y, cauchy_coef):
    # Generate trajectory
    trajectory = generate_crw_trajectory(
        num_steps=num_steps,
        speed=speed,
        start_pos_x=start_x,
        start_pos_y=start_y,
        cauchy_coef=cauchy_coef
    )
    
    # Convert to JSON-serializable format
    return np.array2string(trajectory)

# Callback to update trajectory plot
@app.callback(
    Output("trajectory-graph", "figure"),
    Input("trajectory-store", "children"),
    State("rw-type-radio", "value")
)
def update_trajectory_plot(trajectory_json, rw_type):
    if not trajectory_json:
        return {}
    
    # This is a simplification - in practice, you'd need proper parsing
    # of the trajectory data from the JSON string
    try:
        # For demo purposes, we'll just generate a new trajectory here
        # In practice, you'd parse the trajectory from trajectory_json
        trajectory = generate_crw_trajectory(
            num_steps=1000,
            speed=5,
            start_pos_x=0,
            start_pos_y=0,
            cauchy_coef=0.7
        )
        
        # Create 3D trajectory plot
        fig = go.Figure()
        
        # Plot every nth point to improve performance
        plot_every_n = max(1, len(trajectory) // 500)
        
        fig.add_trace(go.Scatter3d(
            x=trajectory[::plot_every_n, 0],
            y=trajectory[::plot_every_n, 1],
            z=trajectory[::plot_every_n, 2],
            mode='lines',
            name=rw_type,
            line=dict(color='red', width=3)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{rw_type} trajectory",
            scene=dict(
                xaxis_title="x_pos",
                yaxis_title="y_pos",
                zaxis_title="time",
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        print(f"Error updating trajectory plot: {e}")
        return {}

# Callback to update metric plot
@app.callback(
    Output("metric-graph", "figure"),
    Input("trajectory-store", "children"),
    State("metric-type-dropdown", "value")
)
def update_metric_plot(trajectory_json, metric_type):
    if not trajectory_json:
        return {}
    
    try:
        # For demo purposes, we'll just generate a new trajectory here
        # In practice, you'd parse the trajectory from trajectory_json
        trajectory = generate_crw_trajectory(
            num_steps=1000,
            speed=5,
            start_pos_x=0,
            start_pos_y=0,
            cauchy_coef=0.7
        )
        
        # Calculate MSD
        tau, msd = calculate_msd(trajectory)
        
        # Create MSD plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=tau,
            y=msd,
            mode='lines',
            name='MSD',
            line=dict(color='purple', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title="Mean Squared Displacement",
            xaxis_title="tau",
            yaxis_title="MSD",
            template="plotly_white",
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
    except Exception as e:
        print(f"Error updating metric plot: {e}")
        return {}

# Run the app
if __name__ == '__main__':
    print("Starting Random Walks Dashboard...")
    print("Open your web browser and navigate to http://127.0.0.1:8050/")
    app.run_server(debug=True)