# For 3D graphs animated over time.
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure

# For displaying graphs.
import matplotlib.pyplot as plt

# For smooth curves in histograms
import seaborn as sns

# For faster data proccesing.
import numpy as np

# For csv retrieving and information proccesing.
import pandas as pd

# For more clarity and readable code.
from typing import Tuple

############# Defining aliases: #################

# For hills and valleys (extrema)
Extrema = Tuple[np.ndarray, np.ndarray]
# For hills and valleys of a full dataframe
ExtremaCollection = Tuple[Extrema, Extrema, Extrema]

#################################################


def main():
    NORMAL_WALKING_FILENAME = "NormalWalking.csv"
    SLOW_WALKING_FILENAME = "SlowWalking.csv"
    CRAZY_JUMPING_FILENAME = "CrazyJumping.csv"

    # Loading the data:
    normal_walking = read_data(NORMAL_WALKING_FILENAME)
    slow_walking = read_data(SLOW_WALKING_FILENAME)
    crazy_jumping = read_data(CRAZY_JUMPING_FILENAME)

    # Performing EDA:
    visualize_data(normal_walking, slow_walking, crazy_jumping, False)


def visualize_data(normal_walking : pd.DataFrame, slow_walking : pd.DataFrame, crazy_jumping : pd.DataFrame, full_visualization : bool = False) -> None:
    """Performs basic EDA.

    Args:
        normal_walking (pd.DataFrame): Walking at a normal pace DataFrame.
        slow_walking (pd.DataFrame): Walking at a slow pace DataFrame.
        crazy_jumping (pd.DataFrame): Jumping at random directions, Jumping in the same place, walking, staying... DataFrame
        full_visualization (bool, optional): When set to "True" all the processed information will be displayed, when False, only the last result will be displayed. Defaults to False.
    """

    NORMAL_WALKING_STEPS = 103
    SLOW_WALKING_STEPS = 109
    CRAZY_JUMPING_STEPS = 16

    # Visualazing acceleration over axis and over time:
    accelerations_over_time(normal_walking, "Normal Walking", full_visualization)
    accelerations_over_time(slow_walking, "Slow Walking", full_visualization)
    accelerations_over_time(crazy_jumping, "Crazy Jumping", full_visualization)

    # Visualazing the 3D motion over time:
    motion_over_time(normal_walking, "Normal Walking", full_visualization)
    motion_over_time(slow_walking, "Slow Walking", full_visualization)
    motion_over_time(crazy_jumping, "Crazy Jumping", full_visualization)

    # Removing start and finnish kick:
    normal_walking = normal_walking[20:]
    normal_walking = normal_walking[:-31]

    slow_walking = slow_walking[17:]
    slow_walking = slow_walking[:-29]

    # Crazy jumping didn't have start and finnish kick.

    # Visualazing acceleration over axis and over time again:
    accelerations_over_time(normal_walking, "Normal Walking", full_visualization)
    accelerations_over_time(slow_walking, "Slow Walking", full_visualization)
    accelerations_over_time(crazy_jumping, "Crazy Jumping", full_visualization)

    '''
    # Extracting hills and valleys into the same array for further exploration:
    normal_walking_extrema = extract_every_extremum(normal_walking)
    slow_walking_extrema = extract_every_extremum(slow_walking)
    crazy_jumping_extrema = extract_every_extremum(crazy_jumping)

    display_extrema(
        [normal_walking_extrema, slow_walking_extrema, crazy_jumping_extrema],
        True
        )'''


    # Calculating acceleration vector module:
    normal_walking["acceleration module"] = acceleration_module(normal_walking)
    slow_walking["acceleration module"] = acceleration_module(slow_walking)
    crazy_jumping["acceleration module"] = acceleration_module(crazy_jumping)

    # Adding the angle between the biggest component and the full vector:
    normal_walking["alignment angle"] = alignment_angle(normal_walking)
    slow_walking["alignment angle"] = alignment_angle(slow_walking)
    crazy_jumping["alignment angle"] = alignment_angle(crazy_jumping)

    visualize_module_with_angle(normal_walking, full_visualization)
    visualize_module_with_angle(slow_walking, full_visualization)
    visualize_module_with_angle(crazy_jumping, full_visualization)

    # Getting the hills of the module:
    normal_walking_modules, normal_walking_angles = module_hills(normal_walking)
    slow_walking_modules, slow_walking_angles = module_hills(slow_walking)
    crazy_jumping_modules, crazy_jumping_angles = module_hills(crazy_jumping)

    # Getting the Average hill value and 20% perentil:
    log(f"Number of hills | Real steps taken:\n\t- Normal Walking: {len(normal_walking_modules)} | {NORMAL_WALKING_STEPS}\n\t- Slow Walking: {len(slow_walking_modules)} | {SLOW_WALKING_STEPS}\n\t- Crazy Jumping: {len(crazy_jumping_modules)} | {CRAZY_JUMPING_STEPS}", True)
    
    # "Perfect percentile" for the given steps:

    normal_walking_module_percentile, normal_walking_angle_percentile = percentil_20_module_angle(normal_walking_modules, normal_walking_angles)
    slow_walking_module_percentile, slow_walking_angle_percentile = percentil_20_module_angle(slow_walking_modules, slow_walking_angles)
    crazy_jumping_module_percentile, crazy_jumping_angle_percentile = percentil_20_module_angle(crazy_jumping_modules, crazy_jumping_angles)
    log(f"20% Percentil (module, angle):\n\t- Normal Walking: {normal_walking_module_percentile, normal_walking_angle_percentile}\n\t- Slow Walking: {slow_walking_module_percentile, slow_walking_angle_percentile}\n\t- Crazy Jumping: {crazy_jumping_module_percentile, crazy_jumping_module_percentile}", True)

def percentil_20_module_angle(module_hills : np.ndarray, angle_value : np.ndarray) -> Tuple[np.floating, np.floating]:
    module_percentile = np.percentile(module_hills, 20)
    angle_percentile = np.percentile(angle_value, 20)

    return module_percentile, angle_percentile

def average_module_angle(module_hills : np.ndarray, angle_value : np.ndarray) -> Tuple[np.floating, np.floating]:
    module_average = np.mean(module_hills)
    angle_average = np.mean(angle_value)

    return module_average, angle_average

def visualize_module_with_angle(df : pd.DataFrame, display : bool) -> None:
    figure, plots = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    figure.suptitle("Module and Alignment Angle over Time")

    # Acceleration in z axis:
    plots[2].plot(df['time'], df['az'], color='tab:green')
    plots[2].set_ylabel('Az (m/s²)')
    plots[2].set_title('Z-axis Acceleration over Time')
    plots[2].grid(True, linestyle='--', alpha=0.6)

    plots[0].plot(df['time'], df['acceleration module'], color='tab:blue')
    plots[0].set_ylabel('Acceleration Module (m/s²)')
    plots[0].set_title('Acceleration Magnitude over Time')
    plots[0].grid(True, linestyle='--', alpha=0.6)

    plots[1].plot(df['time'], df['alignment angle'], color='tab:orange')
    plots[1].set_ylabel('Dominant-axis Angle (°)')
    plots[1].set_title('Alignment Angle over Time')
    plots[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    show_plot(display)

def module_hills(df : pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    values = df["acceleration module"].values

    # A hill is defined such as an index i such that v[i-1] <= v[i] >= v[i+1]
    hills = np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1

    # Getting the values from the indices:
    module_value = values[hills]
    alignment_angle = df["alignment angle"].iloc[hills]
    return module_value, alignment_angle

def alignment_angle(df : pd.DataFrame) -> np.ndarray:
    largest_component = np.maximum.reduce([df["ax"].abs(), df["ay"].abs(), df["az"].abs()])
    
    # Cosine of the angle:
    ratio = largest_component / df["acceleration module"]
    
    # Angle itself:
    angle = np.arccos(ratio)
    
    # Angle to degrees:
    angle = angle * 180 / np.pi

    return angle

def acceleration_module(df : pd.DataFrame) -> np.ndarray:
    # Calculated as sqrt(x² + y² + z²)
    module = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    return module

def display_extrema(extrema_collections : list[ExtremaCollection], display) -> None:
    # Extracting all the hills and valleys values separated by axis
    x_hills = [extremum[0][0] for extremum in extrema_collections]
    x_valleys = [extremum[0][1] for extremum in extrema_collections]

    y_hills = [extremum[1][0] for extremum in extrema_collections]
    y_valleys = [extremum[1][1] for extremum in extrema_collections]

    z_hills = [extremum[2][0] for extremum in extrema_collections]
    z_valleys = [extremum[2][1] for extremum in extrema_collections]

    # Getting all the extrema in flat lists:
    x_hills = np.concatenate(x_hills)
    y_hills = np.concatenate(y_hills)
    z_hills = np.concatenate(z_hills)
    x_valleys = np.concatenate(x_valleys)
    y_valleys = np.concatenate(y_valleys)
    z_valleys = np.concatenate(z_valleys)

    # Setting up the graph information:
    axes = [x_hills, y_hills, z_hills]
    labels = ["X", "Y", "Z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Build the graph with the frequencies of each axis vale:
    plot = plt.figure()
    plot.canvas.manager.set_window_title("X, Y, Z Accelerations Distributions of Hills")
    for axis, label, color in zip(axes, labels, colors):
        sns.kdeplot(axis, fill=True, alpha=0.3, color=color, label=label)
        plt.axvline(np.mean(axis), color=color, linestyle="--", linewidth=2)
    plt.title("X, Y, Z Accelerations Distributions of Hills")
    plt.xlabel("m/s²")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    show_plot(display)

    # Setting up the graph information:
    axes = [x_valleys, y_valleys, z_valleys]
    labels = ["X", "Y", "Z"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Build the graph with the frequencies of each axis vale:
    plot = plt.figure()
    plot.canvas.manager.set_window_title("X, Y, Z Accelerations Distributions of Valleys")
    for axis, label, color in zip(axes, labels, colors):
        sns.kdeplot(axis, fill=True, alpha=0.3, color=color, label=label)
        plt.axvline(np.mean(axis), color=color, linestyle="--", linewidth=2)
    plt.title("X, Y, Z Accelerations Distributions of Valleys")
    plt.xlabel("m/s²")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    show_plot(display)

def extract_every_extremum(df : pd.DataFrame) -> ExtremaCollection:
    """Extracts the indices of every hill and valley for each axis.

    Args:
        df (pd.DataFrame): Dataframe to extract the hills and valleys from. expected collumns:
            - ax : Accerelation in the x axis.
            - ay : Accerelation in the y axis.
            - az : Acceleration in the z axis.

    Returns:
        ExtremaCollection: Tuple with the valleys and hills of each axis:
            - [0] x axis hills and valleys (Extrema)
            - [1] y axis hills and valleys (Extrema)
            - [2] z axis hills and valleys (Extrema)
    """
    x_extrema = extract_hills_valleys(df["ax"])
    y_extrema = extract_hills_valleys(df["ay"])
    z_extrema = extract_hills_valleys(df["az"])

    return x_extrema, y_extrema, z_extrema

def extract_hills_valleys(series : pd.Series) -> Extrema:
    """Extracts the indices of the hills and valleys.

    Args:
        series (pd.Series): Column of the df to extract hills and valleys

    Returns:
        Extrema: Tuple with the the position of:
            - [0] hills index (np.ndarray)
            - [1] valleys index (np.ndarray)
    """

    # Extracting only the values from the column:
    values = np.asarray(series, dtype=float)

    # A hill is defined such as an index i such that v[i-1] <= v[i] >= v[i+1]
    hills = np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1

    # A valley should then be defined as the index i such that v[i-1] >= v[i] <= v[i+1]
    valleys = np.where((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))[0] + 1

    # Getting the values from the indices:
    hills = values[hills]
    valleys = values[valleys]

    return hills, valleys

def motion_over_time(df : pd.DataFrame, title : str, display : bool, positive_limit : float = 25, negative_limit : float = -25) -> tuple[Figure, Axes3D]:
    """Displays the acceleration vector in 3D over time.

    Args:
        df (pd.DataFrame): DataFrame with the following columns:
            - time : Represents the timestamps of the measuring.
            - ax : Accerelation in the x axis.
            - ay : Accerelation in the y axis.
            - az : Acceleration in the z axis.
        title (str): Title of the main graph.
        display (bool): Whether to display or not the resulting graphs.
        positive_limit (float): Max acceleration values shown in the 3D graph. Defaults to 25.
        negative_limit (float): Minimum acceleration values shown in the 3D graph. Defaults to -25
    """

    # Creating the container figure:
    figure = plt.figure()
    # Setting the title:
    figure.suptitle(title)
    # Adding the main plot with 3D capabilities 
    plot = figure.add_subplot(111, projection="3d")

    # Setting the limits and labels:
    plot.set_xlim(negative_limit, positive_limit)
    plot.set_ylim(negative_limit, positive_limit)
    plot.set_zlim(negative_limit, positive_limit)
    plot.set_xlabel("X acceleration")
    plot.set_ylabel("Y acceleration")
    plot.set_zlabel("Z acceleration")

    # Initializing the arrow directed to the coordinates of the accelerations:
    arrow = plot.quiver(0, 0, 0, 0, 0, 0, length=1.0, normalize=True)

    # Defining an animation function:
    def update(time : int):
        # Removing previous arrow.
        plot.cla()

        # Building an arrow that points to the accelerations of that specific time:
        arrow = plot.quiver(0, 0, 0,
                      df['ax'][time], df['ay'][time], df['az'][time],
                      length=1.0, normalize=True, color='r')
    
        return arrow, 

    # Building the FuncAnimation object.
    # A time of 110ms between each frame was set.
    animation = FuncAnimation(figure, update, frames=len(df), interval=110)
    show_plot(display)

    return figure, plot

def accelerations_over_time(df : pd.DataFrame, title : str, display : bool) -> None:
    """Displays the accerelation of the dataframe over each axis over time.

    Args:
        df (pd.DataFrame): DataFrame with the following columns:
            - time : Represents the timestamps of the measuring.
            - ax : Accerelation in the x axis.
            - ay : Accerelation in the y axis.
            - az : Acceleration in the z axis.
        title (str): Title of the main graph.
        display (bool): Whether to display or not the resulting graphs.
    """
    # Setting up a plot with 3 different subplots (one for each axis)
    plot, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    plot.suptitle(title)

    # Defining axis-x subplot
    axes[0].plot(df['time'], df['ax'], color='r')
    # Defining the title
    axes[0].set_title('X-axis Acceleration')
    # Adding the acceleration units to the subplot.
    axes[0].set_ylabel('m/s²')
    # Adding a grid for better visibility.
    axes[0].grid(True)

    # Repeating for y and z axis.
    axes[1].plot(df['time'], df['ay'], color='g')
    axes[1].set_title('Y-axis Acceleration')
    axes[1].set_ylabel('m/s²')
    axes[1].grid(True)

    axes[2].plot(df['time'], df['az'], color='b')
    axes[2].set_title('Z-axis Acceleration')
    axes[2].set_ylabel('m/s²')
    axes[2].grid(True)

    # Automatic adjust to ensure the subplots fit into the main plot.
    plt.tight_layout()

    show_plot(display)

def read_data(filename : str) -> pd.DataFrame:
    """Returns a Pandas DataFrame given a csv file.

    Args:
        filename (str): Name of the csv file.

    Returns:
        pd.DataFrame: Formatted information extracted from the file with the following columns:
            - time : Represents the timestamps of the measuring.
            - ax : Accerelation in the x axis.
            - ay : Accerelation in the y axis.
            - az : Acceleration in the z axis.
    """
    # Loading the file.
    df = pd.read_csv(filename)

    # Assigning name to the columns
    df.columns = ['time', 'ax', 'ay', 'az']

    return df
     
def log(string : str, display : bool):
    """Prints the string into the terminal if display is set to True.

    Args:
        string (str): Information to display.
        display (bool): If it should be displayed.
    """
    if display:
        print(string)

def show_plot(display : bool):
    """Shows the last created plot if display is set to True. Otherwise the plot is discarded.

    Args:
        display (bool): If it should be displayed or discarded.
    """
    if display:
        plt.show()
    
    else:
        plt.close("all")

if __name__ == '__main__':
    main()