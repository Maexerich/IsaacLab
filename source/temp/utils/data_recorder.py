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
        self.df : pd.DataFrame = None

        self._record = record

    @property
    def get_data(self):
        if self.df is None:
            print("Data has not been transformed to a DataFrame yet. \nCall the save method first.")
        return self.df

    @property
    def record_bool(self):
        return self._record
    
    def store(self, time_seconds: float, values: dict):
        "Stores values in the data attribute."
        self.data[time_seconds] = values
    
    def save(self, path: str):
        "Saves the data attribute as a .csv file at the given path."
        import pandas as pd
        self.df = pd.DataFrame.from_dict(self.data, orient='index')
        self.df.to_csv(path, index_label='time_seconds')

    def plot_old(self, columns_to_plot: Union[list, dict] = None):
        "Creates a matplotlib pop-up plot of the (relevant) stored values. Columns_to_plot is a list of column names."
        
        def add_subplot(fig, ax, title, cols):
            if not isinstance(cols, list):
                cols = [cols]
            
            for col in cols:
                ax.plot(self.df.index, self.df[col])
                ax.set_title(title)
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(col)
        
        if self.df is None:
            print(f"Saving df first...")
            self.save("source/temp/trial_data.csv")
        
        available_columns = self.df.columns
        # Filter columns if columns_to_plot is provided
        if columns_to_plot is None:
            columns_to_plot = available_columns
        elif isinstance(columns_to_plot, dict):
            # Subplots
            titles = list(columns_to_plot.keys())
            # Subplot count
            num_subplots = len(titles)

            fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3 * num_subplots))

        else:
            # Ensure only valid columns are being plotted
            columns_to_plot = [col for col in columns_to_plot if col in available_columns]
        
        print(f"Columns being plotted: {columns_to_plot}")
        
        # Create a figure and set of subplots
        num_columns = len(columns_to_plot)
        fig, axes = plt.subplots(num_columns, 1, figsize=(10, 3 * num_columns))
        
        # Ensure axes is always iterable
        if num_columns == 1:
            axes = [axes]
        
        # Plot each column in a separate subplot
        for i, col in enumerate(columns_to_plot):
            ax = axes[i]
            ax.plot(self.df.index, self.df[col])
            ax.set_title(col)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(col)
            # ax.set_ylim(bottom=0)  # Limit y-axis to positive values only
        
        plt.suptitle(title)        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show(block=True)

    def plot(self, dictionary: dict = None):
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
                if isinstance(self.df[col].iloc[0], torch.Tensor):
                    self.df[col] = self.df[col].apply(lambda x: x.cpu().numpy())
                axes[i].plot(self.df.index, self.df[col], label=col, marker='.', linestyle='None')
                axes[i].set_title(title)
                axes[i].set_xlabel('Time [s]')
                
                if 'effort' in col:
                    axes[i].set_ylabel('Torque [Nm]')
                elif 'vel' in col:
                    axes[i].set_ylabel('Velocity [rad/s]')
                
                axes[i].legend()

        # Add spacing between subplots because titles interfere with the plots
        plt.tight_layout()
        plt.show(block=True)
