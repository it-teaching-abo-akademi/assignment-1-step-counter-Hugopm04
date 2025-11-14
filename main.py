# For faster data proccesing.
import numpy as np

# For csv retrieving and information proccesing.
import pandas as pd

# For working with the accelerometer data (custom module) 
from AccelerometerSession import AccererometerSession


def main():
    NORMAL_WALKING_FILENAME = "NormalWalking.csv"
    SLOW_WALKING_FILENAME = "SlowWalking.csv"
    CRAZY_JUMPING_FILENAME = "CrazyJumping.csv"

    NORMAL_WALKING_STEPS = 103
    SLOW_WALKING_STEPS = 109
    CRAZY_JUMPING_STEPS = 16

    # Loading the data:
    normal_walking = read_data(NORMAL_WALKING_FILENAME)
    slow_walking = read_data(SLOW_WALKING_FILENAME)
    crazy_jumping = read_data(CRAZY_JUMPING_FILENAME)

    full_visualization = True
    normal_walking = AccererometerSession(
        "Normal Walking",
        normal_walking,
        NORMAL_WALKING_STEPS,
        full_visualization
        )
    
    slow_walking = AccererometerSession(
        "Slow Walking",
        slow_walking,
        SLOW_WALKING_STEPS,
        full_visualization
        )
    
    crazy_jumping = AccererometerSession(
        "Crazy Jumping",
        crazy_jumping,
        CRAZY_JUMPING_STEPS,
        full_visualization
        )

    # Performing EDA:
    basic_module_threshold, basic_angle_threshold = visualize_data(normal_walking, slow_walking, crazy_jumping, full_visualization)
    
    # Testing Basic Step Counter:
    accuracy = 0
    accuracy += count_steps(normal_walking, basic_module_threshold, basic_angle_threshold, full_visualization)
    accuracy += count_steps(slow_walking, basic_module_threshold, basic_angle_threshold, full_visualization)
    accuracy += count_steps(crazy_jumping, basic_module_threshold, basic_angle_threshold, full_visualization)
    log(f"Estatic threshold average accuracy: {accuracy / 3}", full_visualization)

    # Testing Dynamic Step Counter:
    accuracy = 0
    accuracy += count_steps_dynamic_threshold(normal_walking, full_visualization)
    accuracy += count_steps_dynamic_threshold(slow_walking, full_visualization)
    accuracy += count_steps_dynamic_threshold(crazy_jumping, full_visualization)
    log(f"Dynamic threshold average accuracy: {accuracy / 3}", full_visualization)

def count_steps_dynamic_threshold(walking_info : AccererometerSession, display : bool = False):
    accuracy = walking_info.create_dynamic_thresholds_plot()
    walking_info.show_plot(display)
    return accuracy

def count_steps(walking_info : AccererometerSession, module_threshold : float, angle_threshold : float, display : bool = False):
    accuracy = walking_info.create_estimated_steps_plot(module_threshold, angle_threshold)
    walking_info.show_plot(display)
    return accuracy

def visualize_data(normal_walking : AccererometerSession, slow_walking : AccererometerSession, crazy_jumping : AccererometerSession, full_visualization : bool = False) -> tuple[float, float]:
    """Performs basic EDA.

    Args:
        normal_walking (pd.DataFrame): Walking at a normal pace DataFrame.
        slow_walking (pd.DataFrame): Walking at a slow pace DataFrame.
        crazy_jumping (pd.DataFrame): Jumping at random directions, Jumping in the same place, walking, staying... DataFrame
        full_visualization (bool, optional): When set to "True" all the processed information will be displayed, when False, only the last result will be displayed. Defaults to False.
    """


    # Visualazing acceleration over axis and over time:
    normal_walking.generate_acceleration_plot()
    normal_walking.show_plot()

    slow_walking.generate_acceleration_plot()
    slow_walking.show_plot()

    crazy_jumping.generate_acceleration_plot()
    crazy_jumping.show_plot()

    # Visualazing the 3D motion over time:
    normal_walking.visualize_3d_vector_plot()

    slow_walking.visualize_3d_vector_plot()

    crazy_jumping.visualize_3d_vector_plot()

    # Removing start and finnish kick:
    normal_walking.remove_edges(20, 31)
    slow_walking.remove_edges(17, 29)
    # Crazy jumping didn't have start and finnish kick.

    # Now we can calculate all the infered data of the sample:
    normal_walking.setup()
    slow_walking.setup()
    crazy_jumping.setup()

    # Visualazing acceleration over axis and over time again:
    normal_walking.generate_acceleration_plot()
    normal_walking.show_plot()
    
    slow_walking.generate_acceleration_plot()
    slow_walking.show_plot()

    crazy_jumping.generate_acceleration_plot()
    crazy_jumping.show_plot()

    # Visualizing acceleration in Z, the module and the alignment angle:
    normal_walking.generate_module_angle_plot()
    normal_walking.show_plot()

    slow_walking.generate_module_angle_plot()
    slow_walking.show_plot()

    crazy_jumping.generate_module_angle_plot()
    crazy_jumping.show_plot()

    normal_walking.describe_hills(True)
    slow_walking.describe_hills(True)
    crazy_jumping.describe_hills(True)

    NORMAL_WALKING_WEIGHT = 0.45
    SLOW_WALKING_WEIGHT = 0.45
    CRAZY_JUMPING_WEIGHT = 0.1
    ideal_module = normal_walking.ideal_module * NORMAL_WALKING_WEIGHT
    ideal_module += slow_walking.ideal_module * SLOW_WALKING_WEIGHT
    ideal_module += crazy_jumping.ideal_module * CRAZY_JUMPING_WEIGHT
    ideal_module = ideal_module

    ideal_angle = normal_walking.ideal_angle * NORMAL_WALKING_WEIGHT
    ideal_angle += slow_walking.ideal_angle * SLOW_WALKING_WEIGHT
    ideal_angle += crazy_jumping.ideal_angle * CRAZY_JUMPING_WEIGHT
    ideal_angle = ideal_angle

    print(f"""
    Thus, the final selected threshold will be:
        - {ideal_module} m/s² for module.
        - {ideal_angle} degrees for angle.
                """
            )
    return ideal_module, ideal_angle

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


if __name__ == '__main__':
    main()