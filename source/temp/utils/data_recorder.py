import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Dict
import torch


### Store values ###
class DataRecorder:
    """This class handles the temporary and local storage of values from the simulation.
    Values are stored as dictionaries at runtime, transformed into a pd.DataFrame at the end
    of simulation and stored locally as a .csv file.

    Use;
    First call DataRecorder.store() to store values at a given time-step.
    Then you can call DataRecorder.save() to save the stored values as a .csv file.
    If you want to plot your results directly, you can skip the save() method and call plot() directly.
    Keep in mind, calling the plot() method will still save the data as a .csv file locally.

    Attributes:
    - data: Dictionary of dictionaries to store values. First key is time-step in seconds,
            the second key is the name of the value."""

    def __init__(self, record: bool = True):
        self.data = {}
        self.df: pd.DataFrame = None

        self._record = record

    @property
    def get_data(self):
        if self.df is None:
            print("Data has not been transformed to a DataFrame yet. \nCall the save method first.")
        return self.df

    @property
    def record_bool(self):
        return self._record

    def record(self, time_seconds: float, values: dict):
        "Stores values in the data attribute."
        if time_seconds in self.data:
            current_entry = self.data[time_seconds]
            values = {**current_entry, **values}
        self.data[time_seconds] = values
    
    def store(self, time_seconds: float, values: dict):
        self.record(time_seconds, values)
        print(f"Use the 'record' method henceforth.")

    def save(self, path: str):
        "Saves the data attribute as a .csv file at the given path."
        import pandas as pd

        self.df = pd.DataFrame.from_dict(self.data, orient="index")
        self.df.to_csv(path, index_label="time_seconds")

    def plot(self, dictionary: dict = None, save_path: str = "source/temp/default_plot.png"):
        """This function plots the specified columns.
        The input dictionary contains as keys the titles of the subplots
        and the value (list) the columnns to plot within the same subplot."""

        titles = list(dictionary.keys())
        num_subplots = len(titles)

        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3 * num_subplots))

        for i, title in enumerate(titles):
            list_of_columns = dictionary[title]
            cols = [column for column in list_of_columns if column in self.df.columns]

            for col in cols:
                # It is possible that the values are torch tensors, still on the GPU
                if isinstance(self.df[col].iloc[0], torch.Tensor):
                    self.df[col] = self.df[col].apply(lambda x: x.cpu().numpy())

                axes[i].plot(self.df.index, self.df[col], label=col, marker=".", linestyle="None")
                # if 'effort' in col:
                #     axes[i].set_ylabel('Torque [Nm]')
                # elif 'vel' in col:
                #     axes[i].set_ylabel('Velocity [rad/s]')

            if "Velocity" in title:
                axes[i].set_ylabel("Velocity [rad/s]")
            elif "Torque" in title:
                axes[i].set_ylabel("Torque [Nm]")
            elif "Force" in title:
                axes[i].set_ylabel("Force [N]")
            elif "deg" in title:
                axes[i].set_ylabel("Angle [deg]")

            axes[i].set_title(title)
            axes[i].set_xlabel("Time [s]")
            axes[i].legend()
            axes[i].grid()

        # Add spacing between subplots because titles interfere with the plots
        plt.tight_layout()
        # plt.show(block=True)
        plt.savefig(save_path)
