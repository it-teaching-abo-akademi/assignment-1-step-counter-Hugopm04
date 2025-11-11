# For csv retrieving and information proccesing.
import pandas as pd

# For better typing:
from pandas import DataFrame

# For displaying graphs.
import matplotlib.pyplot as plt

# For faster data proccesing.
import numpy as np

# For 3D graphs animated over time.
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure

class AccererometerSession():
    A_MODULE_NAME : str = "a module"
    ALIGNMENT_ANGLE_NAME : str = "angle"
    AX_NAME : str  = "ax"
    AY_NAME : str = "ay"
    AZ_NAME : str = "az"

    def __init__(self, name : str, df : DataFrame, taken_steps : int, full_visualization : bool = False):
        self._name = name
        self._df = df
        self._TAKEN_STEPS = taken_steps
        self._FULL_VISUALIZATION = full_visualization
        
        self._hills : dict[str, np.ndarray] = {
            "module" : np.ndarray(),
            "angle" : np.ndarray()
        }
        self._average_hill_module : float
        self._average_hill_angle : float

    @property
    def name(self) -> str:
        return self._name

    @property
    def hills(self):
        return self._hills

    @property
    def TAKEN_STEPS(self) -> int:
        return self._TAKEN_STEPS

    def describe_hills(self, display : bool):
        string = f"""
        Comparison between hill count and real steps taken:
            - Hill count: {len(self.hills["module"])}
            - Real step count: {self.TAKEN_STEPS}
        """
        self.log(string, display)

        self.calculate_hills_averages()

        string = f"""
        Average hills values:
            - Average module: {self.hills["average module"]}
            - Average angle: {self.hills["average angle"]}
        """

        self.log(string, display)

        string = f"""
        Needed threshold to match step count:
        """

    def calculate_hills_percentiles(self, percentile : float):
        module_percentile = np.percentile(self.hills["module"], percentile)
        angle_percentile = np.percentile(self.hills["angle"], percentile)

        self.hills[f"module percentile {percentile}"] = module_percentile
        self.hills[f"angle percentile {percentile}"] = angle_percentile

    def angle_ideal_threshold(self):
        pass

    def module_ideal_threshold(self):
        pass

    def calculate_hills_averages(self) -> None:
        average_module = np.mean(self.hills["module"])
        average_angle = np.mean(self.hills["angle"])

        self.hills["average module"] = average_module
        self.hills["average angle"] = average_angle

    def calculate_module_hills(self):
        values = self._df["a module"].values

        # A hill is defined such as an index i such that v[i-1] <= v[i] >= v[i+1]
        hills = np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1 # type: ignore

        # Getting the values from the indices:
        module_values = values[hills]
        alignment_angles = self._df["alignment angle"].iloc[hills]

        self.hills["module"] = module_values
        self.hills["angle"] = alignment_angles

    def generate_module_angle_plot(self):
        figure, plots = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        figure.suptitle("Module and Alignment Angle over Time")

        # Acceleration in z axis:
        plots[2].plot(self._df['time'], self._df['az'], color='tab:green')
        plots[2].set_ylabel('Az (m/s²)')
        plots[2].set_title('Z-axis Acceleration over Time')
        plots[2].grid(True, linestyle='--', alpha=0.6)

        plots[0].plot(self._df['time'], self._df['acceleration module'], color='tab:blue')
        plots[0].set_ylabel('Acceleration Module (m/s²)')
        plots[0].set_title('Acceleration Magnitude over Time')
        plots[0].grid(True, linestyle='--', alpha=0.6)

        plots[1].plot(self._df['time'], self._df['alignment angle'], color='tab:orange')
        plots[1].set_ylabel('Dominant-axis Angle (°)')
        plots[1].set_title('Alignment Angle over Time')
        plots[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()

    def calculate_alignment_angle(self):
        largest_component = np.maximum.reduce([self._df[AccererometerSession.AX_NAME].abs(), self._df["ay"].abs(), self._df["az"].abs()])
        
        # Cosine of the angle:
        ratio = largest_component / self._df[AccererometerSession.ACCELERATION_MODULE_NAME]
        
        # Angle itself:
        angle = np.arccos(ratio)
        
        # Angle to degrees:
        angle = angle * 180 / np.pi

        self._df[AccererometerSession.ALIGNMENT_ANGLE_NAME] = angle

    def calculate_acceleration_module(self):
        # Calculated as sqrt(x² + y² + z²)
        module = np.sqrt(self._df["ax"]**2 + self._df["ay"]**2 + self._df["az"]**2)
        self._df[AccererometerSession.ACCELERATION_MODULE_NAME] = module

    def remove_edges(self, bottom : int, top : int):
        self._df = self._df[bottom:]
        self._df = self._df[:-top]

    def generate_acceleration_plot(self):
        # Setting up a plot with 3 different subplots (one for each axis)
        plot, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        plot.suptitle(self.name)

        # Defining axis-x subplot
        axes[0].plot(self._df['time'], self._df['ax'], color='r')
        # Defining the title
        axes[0].set_title('X-axis Acceleration')
        # Adding the acceleration units to the subplot.
        axes[0].set_ylabel('m/s²')
        # Adding a grid for better visibility.
        axes[0].grid(True)

        # Repeating for y and z axis.
        axes[1].plot(self._df['time'], self._df['ay'], color='g')
        axes[1].set_title('Y-axis Acceleration')
        axes[1].set_ylabel('m/s²')
        axes[1].grid(True)

        axes[2].plot(self._df['time'], self._df['az'], color='b')
        axes[2].set_title('Z-axis Acceleration')
        axes[2].set_ylabel('m/s²')
        axes[2].grid(True)

        # Automatic adjust to ensure the subplots fit into the main plot.
        plt.tight_layout()
    
    def visualize_3d_vector(self, negative_limit=-10, positive_limit=10):
        # Creating the container figure:
        figure = plt.figure()
        figure.suptitle(self.name)

        # Adding the main plot with 3D capabilities 
        plot = figure.add_subplot(111, projection="3d")

        # Setting the limits and labels:
        plot.set_xlim(negative_limit, positive_limit)
        plot.set_ylim(negative_limit, positive_limit)
        plot.set_zlim(negative_limit, positive_limit)
        plot.set_xlabel("X acceleration")
        plot.set_ylabel("Y acceleration")
        plot.set_zlabel("Z acceleration")

        # --- Animation update function ---
        def update(time: int):
            # Clear previous frame
            plot.cla()

            # Keep the same limits and labels
            plot.set_xlim(negative_limit, positive_limit)
            plot.set_ylim(negative_limit, positive_limit)
            plot.set_zlim(negative_limit, positive_limit)
            plot.set_xlabel("X acceleration")
            plot.set_ylabel("Y acceleration")
            plot.set_zlabel("Z acceleration")
            plot.set_title(f"Time: {time}")

            # Get acceleration components
            ax = self._df['ax'][time]
            ay = self._df['ay'][time]
            az = self._df['az'][time]

            # Compute the module (magnitude)
            module = np.sqrt(ax**2 + ay**2 + az**2)

            # Normalize direction, but scale by magnitude
            # Avoid dividing by zero
            if module != 0:
                dx, dy, dz = ax / module, ay / module, az / module
            else:
                dx, dy, dz = 0, 0, 0

            # Draw arrow with dynamic length
            arrow = plot.quiver(
                0, 0, 0,              # start at origin
                dx, dy, dz,           # direction (unit)
                length=module,        # scale by module
                color='r'
            )

            return arrow,

        # Build and store animation
        anim = FuncAnimation(figure, update, frames=len(self._df), interval=110)

    def log(self, string : str, display : bool = None):
        if display == None:
            display = self._FULL_VISUALIZATION

        if display:
            print(string)

    def show_plot(self, display : bool = None):
        if display == None:
            display = self._FULL_VISUALIZATION
        
        if display:
            plt.show()
        else:
            plt.close("all")