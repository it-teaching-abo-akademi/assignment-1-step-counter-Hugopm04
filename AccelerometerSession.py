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
        figure, subplots = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

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

        accuracy = 1 - abs(self._estimated_dynamic_steps - self.TAKEN_STEPS) / self.TAKEN_STEPS
        accuracy = accuracy * 100
        self.log(f"{self._name}: Estimated steps (Dynamic threshold): {self._estimated_dynamic_steps}, Real steps: {self._TAKEN_STEPS}, Accuracy: {accuracy}%", True)
        return accuracy

    def _add_threshold_lines(self, subplot, key):
        times = self._dynamic_thresholds["time"]
        thresholds = self._dynamic_thresholds[key]
        maxs = self._dynamic_thresholds[f"{key} max"]
        mins = self._dynamic_thresholds[f"{key} min"]

        for time, threshold, max, min in zip(times, thresholds, maxs, mins):
            # Determine the end of the current horizontal segment.
            # For each chunk, the threshold line should extend from its starting timestamp
            # (the first sample in the chunk) up to the start of the next chunk.
            # If this is the last chunk, extend the line up to the very last timestamp in the dataset.
            next_timestamp = (
                times[np.where(times == time)[0][0] + 1]
                if time != times[-1]
                else self._df["time"].iloc[-1]
            )
            subplot.hlines(threshold, time, next_timestamp, color="red", linestyle="--", label="Threshold")
            subplot.hlines(max, time, next_timestamp, color="purple", linestyle=":", label="Max")
            subplot.hlines(min, time, next_timestamp, color="orange", linestyle=":", label="Min")

    def _calculate_dynamic_thresholds(self):
        hills_module = self._hills["module"]
        hills_angle = self._hills["angle"]
        hills_timestamp = self._hills["timestamp"]
        modules = self._df["module"].values
        angles = self._df["angle"].values
        timestamps = self._df["time"].values
        estimated_steps = 0

        for i in range(0, len(modules), self._DYNAMIC_CHUNK):
            module_chunk = modules[i: i + self._DYNAMIC_CHUNK]
            angle_chunk = angles[i: i + self._DYNAMIC_CHUNK]
            timestamp_chunk = timestamps[i: i + self._DYNAMIC_CHUNK]

            if len(module_chunk) == 0:
                continue
            
            max = np.max(module_chunk)
            min = np.min(module_chunk)
            module_threshold = 0.5 * (max + min)
            self._dynamic_thresholds["module"].append(module_threshold)
            self._dynamic_thresholds["module max"].append(max)
            self._dynamic_thresholds["module min"].append(min)

            max = np.max(angle_chunk)
            min = np.min(angle_chunk)
            angle_threshold = 0.5 * (max + min)
            self._dynamic_thresholds["angle"].append(angle_threshold)
            self._dynamic_thresholds["angle max"].append(max)
            self._dynamic_thresholds["angle min"].append(min)
            
            self._dynamic_thresholds["time"].append(timestamp_chunk[0])

            #estimated_steps += np.sum((module_chunk > module_threshold) & (angle_chunk < angle_threshold))
            are_steps = ((hills_timestamp >= timestamp_chunk[0]) & (hills_timestamp <= timestamp_chunk[-1])) & (hills_module > module_threshold) & (hills_angle < angle_threshold)
            estimated_steps += np.sum(are_steps)

        for key in self._dynamic_thresholds:
            self._dynamic_thresholds[key] = np.array(self._dynamic_thresholds[key])
        self._estimated_dynamic_steps = estimated_steps    

    def create_estimated_steps_plot(self, module_threshold : float, angle_threshold : float):
        fig, subplots = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

        subplots[0].plot(self._df["time"], self._df["az"], label='Z acceleration', color='tab:blue')
        subplots[0].set_ylabel("Z acceleration (m/s²)")
        subplots[0].set_title(f"Estimated Steps for {self._name}")
        
        subplots[1].plot(self._df["time"], self._df["module"], label='|a| (module)', color='tab:orange')
        subplots[1].axhline(module_threshold, color='red', linestyle='--', label=f'Module threshold = {self._ideal_module}')
        subplots[1].set_ylabel("|a| (m/s²)")
        
        subplots[2].plot(self._df["time"], self._df["angle"], label='Alignment angle', color='tab:green')
        subplots[2].axhline(angle_threshold, color='red', linestyle='--', label=f'Angle threshold = {self._ideal_angle}')
        subplots[2].set_xlabel("Time (s)")
        subplots[2].set_ylabel("Angle (°)")
        
        # Add estimated steps lines to all subplots
        for ax in subplots:
            for t_event in self._hills["timestamp"]:
                ax.axvline(t_event, color='gray', alpha=0.7)

        # Legend
        for ax in subplots:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        estimated_steps = self.values_higher(module_threshold, angle_threshold)
        accuracy = 1 - abs(estimated_steps - self.TAKEN_STEPS) / self.TAKEN_STEPS
        accuracy = accuracy * 100
        self.log(f"{self._name}: Estimated steps (Static threshold): {estimated_steps}, Real steps: {self._TAKEN_STEPS}, Accuracy: {accuracy}%", True)
        return accuracy

    def describe_hills(self, display : bool = None):
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
        modules = self._hills["module"]
        angles = self._hills["angle"]

        n = len(modules)
        higher_elements_count = np.zeros(n, dtype=int)
        i = 0

        for module, angle in zip(modules, angles):
            higher_elements_count[i] = self.values_higher(module, angle)
            i += 1

        best_index = np.argmin(np.abs(higher_elements_count - self.TAKEN_STEPS))
        steps = higher_elements_count[best_index]
        self._estmiated_steps = steps
        self._ideal_module = modules[best_index]
        self._ideal_angle = angles[best_index]

    def values_higher(self, module : float, angle : float):
        modules = self._hills["module"]
        angles = self._hills["angle"]
        count = np.sum((modules > module) & (angles < angle))
        return count

    def calculate_hills_averages(self) -> None:
        average_module = np.mean(self._hills["module"])
        average_angle = np.mean(self._hills["angle"])

        self._hills["average module"] = average_module
        self._hills["average angle"] = average_angle

    def calculate_module_hills(self):
        values = self._df["module"].values

        # A hill is defined such as an index i such that v[i-1] <= v[i] >= v[i+1]
        hills = np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1 # type: ignore

        # Getting the values from the indices:
        module_values = values[hills]
        alignment_angles = self._df["angle"].iloc[hills].values
        timestamps = self._df["time"].iloc[hills].values

        self._hills["module"] = np.sort(module_values)
        self._hills["angle"] = alignment_angles
        self._hills["timestamp"] = timestamps

    def generate_module_angle_plot(self):
        figure, plots = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        figure.suptitle(f"Module and Alignment Angle over Time ({self._name})")

        # Acceleration in z axis:
        plots[2].plot(self._df['time'], self._df['az'], color='tab:green')
        plots[2].set_ylabel('Az (m/s²)')
        plots[2].set_title('Z-axis Acceleration')
        plots[2].grid(True, linestyle='--', alpha=0.6)

        plots[0].plot(self._df['time'], self._df['module'], color='tab:blue')
        plots[0].set_ylabel('Acceleration Module (m/s²)')
        plots[0].set_title('Acceleration Module')
        plots[0].grid(True, linestyle='--', alpha=0.6)

        plots[1].plot(self._df['time'], self._df['angle'], color='tab:orange')
        plots[1].set_ylabel('Dominant-axis Angle (°)')
        plots[1].set_title('Alignment Angle')
        plots[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()

    def calculate_alignment_angle(self):
        largest_component = np.maximum.reduce([self._df["ax"].abs(), self._df["ay"].abs(), self._df["az"].abs()])
        
        # Cosine of the angle:
        ratio = largest_component / self._df["module"]
        
        # Angle itself:
        angle = np.arccos(ratio)
        
        # Angle to degrees:
        angle = angle * 180 / np.pi

        self._df["angle"] = angle

    def calculate_acceleration_module(self):
        # Calculated as sqrt(x² + y² + z²)
        module = np.sqrt(self._df["ax"]**2 + self._df["ay"]**2 + self._df["az"]**2)
        self._df["module"] = module

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
    
    def visualize_3d_vector_plot(self, negative_limit=-20, positive_limit=20):
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
        self.show_plot()

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