# For better typing:
from pandas import DataFrame

# For displaying graphs.
import matplotlib.pyplot as plt

# For faster data proccesing.
import numpy as np

# For 3D graphs animated over time.
from matplotlib.animation import FuncAnimation

class AccererometerSession():
    def __init__(self, name : str, df : DataFrame, taken_steps : int, full_visualization : bool = False, dynamic_threshold_chunk_size : int = 20):
        """Stores data from an Accelerometer Session and can perform various calculations and visualizations.

        Args:
            name (str): Name of the session.
            df (DataFrame): DataFrame containing accelerometer data with columns 'time', 'ax', 'ay', 'az'.
            taken_steps (int): Number of steps actually taken during the session.
            full_visualization (bool, optional): Whether to display all plots and logs. Defaults to False.
            dynamic_threshold_chunk_size (int, optional): Size of chunks for dynamic threshold calculation. Defaults to 20.
        """

        self._name = name
        self._df = df
        self._TAKEN_STEPS = taken_steps
        self._FULL_VISUALIZATION = full_visualization
        self._DYNAMIC_CHUNK = dynamic_threshold_chunk_size

        self._hills : dict[str, np.ndarray] = {
            "module" : np.empty(0),
            "angle" : np.empty(0),
            "timestamp" : np.empty(0)
        }
        self._dynamic_thresholds : dict[str, list[float]] = {
            "module" : [],
            "module max" : [],
            "module min" : [],
            "angle" : [],
            "angle max" : [],
            "angle min" : [],
            "time" : []
        }

        self._average_hill_module : float
        self._average_hill_angle : float
        self._ideal_module : float 
        self._ideal_angle : float
        self._estmiated_steps : int
        self._estimated_dynamic_steps : int

    def setup(self):
        """
        Prepares the session data by calculating necessary metrics.
        """
        # Calculating acceleration vector module:
        self.calculate_acceleration_module()
        # Adding the angle between the biggest component and the full vector:
        self.calculate_alignment_angle()

        # Calculating the hills of the module:
        self.calculate_module_hills()

        # Calculating average module and angle:
        self.calculate_hills_averages()
        # Calculating ideal module and angle to use as threshold for step counting.
        self.calculate_ideal_threshold()

        self._calculate_dynamic_thresholds()
        

    @property
    def name(self) -> str:
        return self._name

    @property
    def TAKEN_STEPS(self) -> int:
        return self._TAKEN_STEPS

    @property
    def ideal_module(self) -> float:
        return self._ideal_module
    
    @property
    def ideal_angle(self) -> float:
        return self._ideal_angle

    def create_dynamic_thresholds_plot(self):
        """Generates a plot showing dynamic thresholds over time."""

        # Setting up the plot
        figure, subplots = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Setting the main title
        figure.suptitle(f"Dynamic Thresholds for {self._name}")

        # Z-axis acceleration
        subplots[2].plot(self._df["time"], self._df["az"], color="tab:green")
        subplots[2].set_title("Z-axis Acceleration")
        subplots[2].grid(True, linestyle="--", alpha=0.6)

        # Acceleration module
        subplots[0].plot(self._df["time"], self._df["module"], color="tab:blue")
        self._add_threshold_lines(subplots[0], "module")
        subplots[0].set_title("Acceleration Module")
        subplots[0].grid(True, linestyle="--", alpha=0.6)

        # Alignment angle
        subplots[1].plot(self._df["time"], self._df["angle"], color="tab:orange")
        self._add_threshold_lines(subplots[1], "angle")
        subplots[1].set_title("Alignment Angle")
        subplots[1].grid(True, linestyle="--", alpha=0.6)

        # Calculating estimated steps and accuracy
        accuracy = 1 - abs(self._estimated_dynamic_steps - self.TAKEN_STEPS) / self.TAKEN_STEPS
        accuracy = accuracy * 100
        self.log(f"{self._name}: Estimated steps (Dynamic threshold): {self._estimated_dynamic_steps}, Real steps: {self._TAKEN_STEPS}, Accuracy: {accuracy}%", True)
        return accuracy

    def _add_threshold_lines(self, subplot, key):
        """Adds horizontal lines to the given subplot representing dynamic thresholds.
        Args:
            subplot: The subplot to which the lines will be added.
            key: The key indicating whether to add lines for 'module' or 'angle'."""

        # Extracting relevant data
        times = self._dynamic_thresholds["time"]
        thresholds = self._dynamic_thresholds[key]
        maxs = self._dynamic_thresholds[f"{key} max"]
        mins = self._dynamic_thresholds[f"{key} min"]


        # Determine the end of the current horizontal segment.
        # For each chunk, the threshold line should extend from its starting timestamp
        # (the first sample in the chunk) up to the start of the next chunk.
        for time, threshold, max, min in zip(times, thresholds, maxs, mins):
            next_timestamp = (
                times[np.where(times == time)[0][0] + 1]
                if time != times[-1]
                # If this is the last chunk, extend the line up to the very last timestamp in the dataset.
                else self._df["time"].iloc[-1]
            )
            # Adding horizontal lines for threshold, max, and min values.
            subplot.hlines(threshold, time, next_timestamp, color="red", linestyle="--", label="Threshold")
            subplot.hlines(max, time, next_timestamp, color="purple", linestyle=":", label="Max")
            subplot.hlines(min, time, next_timestamp, color="orange", linestyle=":", label="Min")

    def _calculate_dynamic_thresholds(self):
        """Calculates dynamic thresholds for step detection over chunks of data."""

        # Extracting hills data
        hills_module = self._hills["module"]
        hills_angle = self._hills["angle"]
        hills_timestamp = self._hills["timestamp"]
        modules = self._df["module"].values
        angles = self._df["angle"].values
        timestamps = self._df["time"].values
        estimated_steps = 0

        # Processing data in chunks to calculate dynamic thresholds
        for i in range(0, len(modules), self._DYNAMIC_CHUNK):
            # Getting the current chunk as slices
            module_chunk = modules[i: i + self._DYNAMIC_CHUNK]
            angle_chunk = angles[i: i + self._DYNAMIC_CHUNK]
            timestamp_chunk = timestamps[i: i + self._DYNAMIC_CHUNK]

            # Skip empty chunks
            if len(module_chunk) == 0:
                continue
            
            # Calculating thresholds for the current chunk for module
            max = np.max(module_chunk)
            min = np.min(module_chunk)
            module_threshold = 0.5 * (max + min)
            self._dynamic_thresholds["module"].append(module_threshold)
            self._dynamic_thresholds["module max"].append(max)
            self._dynamic_thresholds["module min"].append(min)

            # Calculating thresholds for the current chunk for angle
            max = np.max(angle_chunk)
            min = np.min(angle_chunk)
            angle_threshold = 0.5 * (max + min)
            self._dynamic_thresholds["angle"].append(angle_threshold)
            self._dynamic_thresholds["angle max"].append(max)
            self._dynamic_thresholds["angle min"].append(min)
            
            # Storing the starting timestamp of the chunk
            self._dynamic_thresholds["time"].append(timestamp_chunk[0])

            # Counting estimated steps in this chunk
            are_steps = ((hills_timestamp >= timestamp_chunk[0]) & (hills_timestamp <= timestamp_chunk[-1])) & (hills_module > module_threshold) & (hills_angle < angle_threshold)
            estimated_steps += np.sum(are_steps)

        # Converting lists to numpy arrays for easier handling later
        for key in self._dynamic_thresholds:
            self._dynamic_thresholds[key] = np.array(self._dynamic_thresholds[key])

        # Storing the total estimated steps
        self._estimated_dynamic_steps = estimated_steps    

    def create_estimated_steps_plot(self, module_threshold : float, angle_threshold : float):
        """Generates a plot showing estimated steps using static thresholds over time."""

        # Setting up the plot
        fig, subplots = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

        # Setting Z-axis acceleration
        subplots[0].plot(self._df["time"], self._df["az"], label='Z acceleration', color='tab:blue')
        subplots[0].set_ylabel("Z acceleration (m/s²)")
        subplots[0].set_title(f"Estimated Steps for {self._name}")
        
        # Acceleration module
        subplots[1].plot(self._df["time"], self._df["module"], label='|a| (module)', color='tab:orange')
        subplots[1].axhline(module_threshold, color='red', linestyle='--', label=f'Module threshold = {self._ideal_module}')
        subplots[1].set_ylabel("|a| (m/s²)")
        
        # Alignment angle
        subplots[2].plot(self._df["time"], self._df["angle"], label='Alignment angle', color='tab:green')
        subplots[2].axhline(angle_threshold, color='red', linestyle='--', label=f'Angle threshold = {self._ideal_angle}')
        subplots[2].set_ylabel("Angle (°)")
        
        # Add one vertical line for each detected step
        step_times = self._hills["timestamp"][
            (self._hills["module"] > module_threshold) &
            (self._hills["angle"] < angle_threshold)
        ]
        for subplot in subplots:
            for step_time in step_times:
                subplot.axvline(step_time, color='grey', linestyle='-', alpha=0.5)
        

        # Legend
        for ax in subplots:
            ax.legend(loc='upper right')
        
        # Automatic layout adjustment
        plt.tight_layout()

        # Calculating estimated steps and accuracy
        estimated_steps = self.values_higher(module_threshold, angle_threshold)
        accuracy = 1 - abs(estimated_steps - self.TAKEN_STEPS) / self.TAKEN_STEPS
        accuracy = accuracy * 100

        self.log(f"{self._name}: Estimated steps (Static threshold): {estimated_steps}, Real steps: {self._TAKEN_STEPS}, Accuracy: {accuracy}%", True)
        return accuracy

    def describe_hills(self, display : bool = None):
        """Prints a description of the hills detected in the session."""

        if display == None:
            display = self._FULL_VISUALIZATION
        
        self.log(f"Describing hills of {self._name}:", display)

        string = f"""Comparison between hill count and real steps taken:
    - Hill count: {len(self._hills["module"])}
    - Real step count: {self.TAKEN_STEPS}
        """
        self.log(string, display)

        string = f"""Average hills values:
    - Average module: {self._hills["average module"]}
    - Average angle: {self._hills["average angle"]}
        """

        self.log(string, display)

        string = f"""Needed threshold to match step count:
    - Module: {self._ideal_module}
    - Angle: {self._ideal_angle}
    - Real Step Count: {self.TAKEN_STEPS}, Steps according to this threshold: {self._estmiated_steps}
        """
        self.log(string, display)

    def calculate_ideal_threshold(self) -> None:
        """Calculates the ideal module and angle thresholds to match the taken steps."""

        modules = self._hills["module"]
        angles = self._hills["angle"]

        n = len(modules)
        higher_elements_count = np.zeros(n, dtype=int)
        i = 0

        # Testing all possible combinations of module and angle thresholds
        for module, angle in zip(modules, angles):
            higher_elements_count[i] = self.values_higher(module, angle)
            i += 1

        # Finding the thresholds that yield a step count closest to the actual taken steps
        best_index = np.argmin(np.abs(higher_elements_count - self.TAKEN_STEPS))
        steps = higher_elements_count[best_index]

        # Storing the results
        self._estmiated_steps = steps
        self._ideal_module = modules[best_index]
        self._ideal_angle = angles[best_index]

    def values_higher(self, module : float, angle : float):
        """Counts the number of hills where the module is higher than the given module threshold and the angle is lower than the given angle threshold."""

        modules = self._hills["module"]
        angles = self._hills["angle"]
        count = np.sum((modules > module) & (angles < angle))
        return count

    def calculate_hills_averages(self) -> None:
        """Calculates the average module and angle of the detected hills."""

        average_module = np.mean(self._hills["module"])
        average_angle = np.mean(self._hills["angle"])

        self._hills["average module"] = average_module
        self._hills["average angle"] = average_angle

    def calculate_module_hills(self):
        """Identifies the hills (local maxima) in the acceleration module data."""

        values = self._df["module"].values

        # A hill is defined such as an index i such that v[i-1] <= v[i] >= v[i+1]
        hills = np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1 # type: ignore

        # Getting the values from the indices:
        module_values = values[hills]
        alignment_angles = self._df["angle"].iloc[hills].values
        timestamps = self._df["time"].iloc[hills].values

        self._hills["module"] = (module_values)
        self._hills["angle"] = alignment_angles
        self._hills["timestamp"] = timestamps

    def generate_module_angle_plot(self):
        """Generates a plot showing acceleration in Z, the module, and the alignment angle over time."""

        # Setting up the plot
        figure, plots = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Setting the main title
        figure.suptitle(f"Module and Alignment Angle over Time ({self._name})")

        # Acceleration in z axis:
        plots[2].plot(self._df['time'], self._df['az'], color='tab:green')
        plots[2].set_ylabel('Az (m/s²)')
        plots[2].set_title('Z-axis Acceleration')
        plots[2].grid(True, linestyle='--', alpha=0.6)

        # Acceleration module and alignment angle:
        plots[0].plot(self._df['time'], self._df['module'], color='tab:blue')
        plots[0].set_ylabel('Acceleration Module (m/s²)')
        plots[0].set_title('Acceleration Module')
        plots[0].grid(True, linestyle='--', alpha=0.6)

        plots[1].plot(self._df['time'], self._df['angle'], color='tab:orange')
        plots[1].set_ylabel('Dominant-axis Angle (°)')
        plots[1].set_title('Alignment Angle')
        plots[1].grid(True, linestyle='--', alpha=0.6)

        # Automatic layout adjustment
        plt.tight_layout()

    def calculate_alignment_angle(self):
        """Calculates the alignment angle between the largest acceleration component and the overall acceleration vector."""

        # Finding the largest component among ax, ay, az
        largest_component = np.maximum.reduce([self._df["ax"].abs(), self._df["ay"].abs(), self._df["az"].abs()])
        
        # Cosine of the angle:
        ratio = largest_component / self._df["module"]
        
        # Angle itself:
        angle = np.arccos(ratio)
        
        # Angle to degrees:
        angle = angle * 180 / np.pi

        self._df["angle"] = angle

    def calculate_acceleration_module(self):
        """Calculates the acceleration vector module and adds it as a new column to the DataFrame."""

        # Calculated as sqrt(x² + y² + z²)
        module = np.sqrt(self._df["ax"]**2 + self._df["ay"]**2 + self._df["az"]**2)
        self._df["module"] = module

    def remove_edges(self, bottom : int, top : int):
        """Removes a number of samples from the beginning and end of the DataFrame.
        Args:
            bottom (int): Number of samples to remove from the beginning.
            top (int): Number of samples to remove from the end.
        """
        
        self._df = self._df[bottom:]
        self._df = self._df[:-top]

    def generate_acceleration_plot(self):
        """Generates a plot showing acceleration in X, Y, and Z axes over time."""

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
    
    def visualize_3d_vector_plot(self, negative_limit=-20, positive_limit=20):
        """Generates a 3D animated plot showing the acceleration vector over time.
        Args:
            negative_limit (int, optional): Minimum limit for all axes. Defaults to -20.
            positive_limit (int, optional): Maximum limit for all axes. Defaults to 20.
        """

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
                # Start at origin
                0, 0, 0,
                # Direction (unit)              
                dx, dy, dz,
                # Scale by module           
                length=module,        
                color='r'
            )
            return arrow,

        # Build and store animation
        anim = FuncAnimation(figure, update, frames=len(self._df), interval=110)
        self.show_plot()

    def log(self, string : str, display : bool = None):
        """Prints the string into the terminal if display is set to True.
            Args:
                string (str): String to print.
                display (bool, optional): Whether to print the string. Defaults to None, which uses the session's full visualization setting.
        """
        
        if display == None:
            display = self._FULL_VISUALIZATION

        if display:
            print(string)

    def show_plot(self, display : bool = None):
        """Displays the current matplotlib plot if display is set to True.
            Args:
                display (bool, optional): Whether to show the plot. Defaults to None, which uses the session's full visualization setting.
        """
        if display == None:
            display = self._FULL_VISUALIZATION
        
        if display:
            plt.show()
        else:
            plt.close("all")