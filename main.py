# For 3D graphs animated over time.
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure

# For displaying graphs.
import matplotlib.pyplot as plt

# For csv retrieving and information proccesing.
import pandas as pd

def main():
    NORMAL_WALKING_FILENAME = "NormalWalking.csv"
    SLOW_WALKING_FILENAME = "SlowWalking.csv"
    CRAZY_JUMPING_FILENAME = "CrazyJumping.csv"

    # Loading the data:
    normal_walking = read_data(NORMAL_WALKING_FILENAME)
    slow_walking = read_data(SLOW_WALKING_FILENAME)
    crazy_jumping = read_data(CRAZY_JUMPING_FILENAME)

    # Performing EDA:
    visualize_data(normal_walking, slow_walking, crazy_jumping)

def visualize_data(normal_walking : pd.DataFrame, slow_walking : pd.DataFrame, crazy_jumping : pd.DataFrame, full_visualization : bool = False) -> None:
    """Performs basic EDA.

    Args:
        normal_walking (pd.DataFrame): Walking at a normal pace DataFrame.
        slow_walking (pd.DataFrame): Walking at a slow pace DataFrame.
        crazy_jumping (pd.DataFrame): Jumping at random directions, Jumping in the same place, walking, staying... DataFrame
        full_visualization (bool, optional): When set to "True" all the processed information will be displayed, when False, only the last result will be displayed. Defaults to False.
    """

    # Visualazing acceleration over axis and over time:
    accelerations_over_time(normal_walking, "Normal Walking", full_visualization)
    accelerations_over_time(slow_walking, "Slow Walking", full_visualization)
    accelerations_over_time(crazy_jumping, "Crazy Jumping", full_visualization)

    # Visualazing the 3D motion over time:
    motion_over_time(normal_walking, "Normal Walking", full_visualization)
    motion_over_time(slow_walking, "Slow Walking", True)
    motion_over_time(crazy_jumping, "Crazy Jumping", True)

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