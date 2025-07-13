import numpy as np
import pandas as pd
import tdt
import os
import matplotlib.pyplot as plt

from trial_class import Trial

class Hello:
    def __init__(self, experiment_folder_path, behavior_folder_path):
        self.experiment_folder_path = experiment_folder_path
        self.behavior_folder_path = behavior_folder_path
        self.trials = {}

        self.load_trials()
    

    '''********************************** GROUP PROCESSING **********************************'''
    def load_trials(self):
        """
        Loads each trial folder (block) as a TDTData object and extracts manual annotation behaviors.
        """
        trial_folders = [folder for folder in os.listdir(self.experiment_folder_path)
                        if os.path.isdir(os.path.join(self.experiment_folder_path, folder))]

        for trial_folder in trial_folders:
            trial_path = os.path.join(self.experiment_folder_path, trial_folder)
            trial_obj = Trial(trial_path)

            self.trials[trial_folder] = trial_obj


    def default_batch_process(self, time_segments_to_remove=None):
        """
        Batch processes the TDT data by extracting behaviors, removing LED artifacts, and computing z-score.
        """
        for trial_folder, trial in self.trials.items():
            # Check if the subject name is in the time_segments_to_remove dictionary
            if time_segments_to_remove and trial.subject_name in time_segments_to_remove:
                self.remove_time_segments_from_block(trial_folder, time_segments_to_remove[trial.subject_name])

            print(f"Processing {trial_folder}...")
            trial.remove_initial_LED_artifact(t=30)
            # trial.remove_final_data_segment(t = 10)
            
            trial.smooth_and_apply(window_len=int(trial.fs)*2)
            trial.apply_ma_baseline_drift()
            trial.align_channels()
            trial.compute_dFF()
            #baseline_start, baseline_end = trial.find_baseline_period()  
            #trial.compute_zscore(method = 'baseline', baseline_start = baseline_start, baseline_end = baseline_end)
            trial.compute_zscore(method = 'standard')

            trial.verify_signal()


    def group_extract_manual_annotations(self, bout_definitions, first_only=True):
        """
        Extracts behavior bouts and annotations for all trials in the experiment.

        This function:
        1. Iterates through self.trials, looking for behavior CSV files in self.behavior_folder_path.
        2. Calls extract_bouts_and_behaviors for each trial.
        3. Stores the behavior data inside each Trial object.

        Parameters:
        - bout_definitions (list of dict): List defining each bout with:
            - 'prefix': Label used for the bout (e.g., "s1", "s2", "x").
            - 'introduced': Name of the behavior marking the start of the bout.
            - 'removed': Name of the behavior marking the end of the bout.
        - first_only (bool): If True, only the first event in each bout is kept;
                            if False, all events within each bout are retained.
        """
        for trial_name, trial in self.trials.items():
            csv_path = os.path.join(self.behavior_folder_path, f"{trial_name}.csv")
            if os.path.exists(csv_path):
                print(f"Processing behaviors for {trial_name}...")
                trial.extract_bouts_and_behaviors(csv_path, bout_definitions, first_only=first_only)
                # trial.combine_consecutive_behaviors(behavior_name='all', bout_time_threshold=0.3)
                # Optionally, you can remove short behaviors:
                # trial.remove_short_behaviors(behavior_name='all', min_duration=0.3)
            else:
                print(f"Warning: No CSV found for {trial_name} in {self.behavior_folder_path}. Skipping.")

    '''********************************** PLOTTING **********************************'''
    def plot_all_traces(experiment, behavior_name='all'): 
        """
        Plots behavior events for all trials in separate subplots within the same figure.
        """
        num_trials = len(experiment.trials)

        if num_trials == 0:
            print("No trials found in the experiment.")
            return
        
        fig, axes = plt.subplots(nrows=num_trials, figsize=(12, 3 * num_trials), linewidth =0.1, sharex=True)

        # Ensure axes is iterable (in case of a single subplot)
        if num_trials == 1:
            axes = [axes]

        # Loop through trials and plot behavior events in each subplot
        for ax, (trial_name, trial) in zip(axes, experiment.trials.items()):
            trial.plot_behavior_event(behavior_name, ax=ax) 
            ax.set_title(trial_name)

        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout()
        plt.show()


    def plot_all_behavior_PETHs(self, selected_bouts=None):
        """
        Plots ALL Investigation PETHs for every trial in the experiment, with separate subplots.

        For each trial, the method:
        - Filters rows where 'Behavior' equals 'Investigation'.
        - (Optionally) filters the data to only include specific bouts (if selected_bouts is given).
        - Sorts by [Bout, Event_Start] so events appear in chronological order.
        - Plots each Investigation trace in its own subplot (Relative_Time_Axis vs. Relative_Zscore).
            * a dashed black line at x=0 (Investigation Start),
            * a dashed red line at x=Time of Max Peak.
        - Color-codes lines by the 'Subject' column (e.g., "Novel" vs. "Subject").

        The y-axis limits are determined dynamically based on the global minimum and maximum values 
        across all plotted bouts (with an extra margin of 1 added to each end). Each subplot displays 
        its own y-axis tick numbers.

        Parameters:
        selected_bouts (list, optional): A list of bout identifiers to include. If None, all bouts are plotted.
        """
        # Lists to store data per trial.
        trial_data = []
        trial_names = []
        max_events = 0  # Track maximum number of Investigation events across all trials

        # -- 1) Gather Investigation data for each trial --
        for trial_name, trial in self.trials.items():
            # Skip trials that don't have the 'behaviors' attribute
            if not hasattr(trial, 'behaviors'):
                continue
            
            # Filter for Investigation events
            df_invest = trial.behaviors[trial.behaviors["Behavior"] == "Investigation"].copy()
            
            # (Optionally) filter by selected bouts
            if selected_bouts is not None:
                df_invest = df_invest[df_invest["Bout"].isin(selected_bouts)]
            
            # Sort by [Bout, Event_Start] so events are plotted in chronological order
            df_invest = df_invest.sort_values(["Bout", "Event_Start"], ascending=True)
            
            # Keep track of how many total Investigation events this trial has
            n_events = len(df_invest)
            if n_events > 0:
                trial_data.append(df_invest)
                trial_names.append(trial_name)
                if n_events > max_events:
                    max_events = n_events

        # If there's no data at all, exit.
        if len(trial_data) == 0:
            print("No trial data available for plotting Investigation PETHs.")
            return

        # -- 2) Determine global y-axis limits (across all events) --
        global_min = np.inf
        global_max = -np.inf

        for df_invest in trial_data:
            for _, row in df_invest.iterrows():
                # row["Relative_Zscore"] should be an array or list of y-values
                y_data = row["Relative_Zscore"]
                if y_data is not None and len(y_data) > 0:
                    current_min = np.min(y_data)
                    current_max = np.max(y_data)
                    if current_min < global_min:
                        global_min = current_min
                    if current_max > global_max:
                        global_max = current_max

        ymin = global_min - 1
        ymax = global_max + 1

        # -- 3) Create subplots: one row per trial, one column per Investigation event --
        n_rows = len(trial_data)
        n_cols = max_events
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Ensure axes is a 2D array for consistent indexing.
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = np.array([[ax] for ax in axes])

        # -- 4) Plot each Investigation event in its own subplot --
        for row_idx, (df_invest, trial_name) in enumerate(zip(trial_data, trial_names)):
            for col_idx in range(n_cols):
                ax = axes[row_idx][col_idx]
                
                # Check if this trial actually has an event at this index
                if col_idx < len(df_invest):
                    data_row = df_invest.iloc[col_idx]

                    # Extract arrays for time and Z-score.
                    x = data_row["Relative_Time_Axis"]
                    y = data_row["Relative_Zscore"]

                    # Decide on line color based on whether Subject == 'Novel' or something else
                    if data_row["Subject"] == "Novel":
                        line_color = "green"
                    else:
                        line_color = "blue"

                    # Plot the Investigation trace.
                    ax.plot(x, y, color=line_color,
                            label=f"{data_row['Subject']} - {data_row['Bout']}")

                    # Plot vertical dashed lines for Start (x=0) and Time of Max Peak.
                    ax.axvline(x=0, color='black', linestyle='--', label="Start")
                    ax.axvline(x=data_row["Time of Max Peak"], color='red', linestyle='--', label="Max Peak")
                    
                    # Set y-axis limits based on global min/max.
                    ax.set_ylim([ymin, ymax])
                    ax.set_xlabel("Relative Time (s)")
                    ax.set_title(f"Trial {trial_name}\nBout: {data_row['Bout']}")

                    # Add y-label and legend for the first column to reduce clutter.
                    if col_idx == 0:
                        ax.set_ylabel("Z-score")
                        ax.legend()
                else:
                    # No event for this trial in this column => turn off the axis.
                    ax.axis('off')

        plt.tight_layout()
        plt.show()

    def plot_first_behavior_PETHs(self, selected_bouts=None):
        """
        Plots the first Investigation PETHs for all trials in the experiment.
        
        For each trial, the method:
          - Filters rows where 'Behavior' equals 'Investigation'.
          - (Optionally) Further filters the data to only include bouts listed in 'selected_bouts'.
          - Groups the data by 'Bout' and selects the first event in each bout.
          - Plots the investigation trace (Relative_Time_Axis vs. Relative_Zscore) with:
              • a dashed black line at x = 0 (Investigation Start),
              • a dashed blue line at x = Duration (s) (Investigation End),
              • a dashed red line at x = Time of Max Peak.
              
        The y-axis limits are determined dynamically based on the global minimum and maximum values 
        across all plotted bouts (with an extra margin of 1 added to each end). Each subplot displays 
        its own y-axis tick numbers.
        
        Parameters:
          selected_bouts (list, optional): A list of bout identifiers to include. If None, all bouts are plotted.
        """
        trial_first_data = []  # list to store each trial's first investigation DataFrame
        trial_names = []       # list to track trial names
        max_bouts = 0          # maximum number of bouts across trials

        # Loop over each trial and extract first-investigation events.
        for trial_name, trial in self.trials.items():
            if not hasattr(trial, 'behaviors'):
                continue
            # Filter for Investigation events.
            df_invest = trial.behaviors[trial.behaviors["Behavior"] == "Investigation"].copy()
            
            # If a selection of bouts is provided, filter to include only those bouts.
            if selected_bouts is not None:
                df_invest = df_invest[df_invest["Bout"].isin(selected_bouts)]
            
            # Group by 'Bout' and take the first event in each group.
            df_first_invest = df_invest.groupby("Bout", as_index=False).first()
            trial_first_data.append(df_first_invest)
            trial_names.append(trial_name)
            if len(df_first_invest) > max_bouts:
                max_bouts = len(df_first_invest)

        if len(trial_first_data) == 0:
            print("No trial data available for plotting first investigation behaviors.")
            return

        # Determine global y-axis limits from all Relative_Zscore data.
        global_min = np.inf
        global_max = -np.inf
        for df_first in trial_first_data:
            for _, row in df_first.iterrows():
                y_data = row["Relative_Zscore"]
                current_min = np.min(y_data)
                current_max = np.max(y_data)
                if current_min < global_min:
                    global_min = current_min
                if current_max > global_max:
                    global_max = current_max

        ymin = global_min - 1
        ymax = global_max + 1

        n_rows = len(trial_first_data)
        n_cols = max_bouts

        # Create a grid of subplots without sharing y axes so each shows its own y-axis numbers.
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        # Ensure axes is a 2D array for consistent indexing.
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        # Loop over each trial (row) and each bout (column) to plot.
        for row_idx, (df_first, trial_name) in enumerate(zip(trial_first_data, trial_names)):
            for col_idx in range(n_cols):
                ax = axes[row_idx][col_idx]
                # Only plot if data for this bout exists.
                if col_idx < len(df_first):
                    data_row = df_first.iloc[col_idx]

                    # Extract time and Z-score arrays.
                    x = data_row["Relative_Time_Axis"]
                    y = data_row["Relative_Zscore"]

                    # Plot the investigation trace.
                    ax.plot(x, y, label=f"Bout: {data_row['Bout']}")
                    # Plot vertical dashed lines:
                    # Start (x = 0)
                    ax.axvline(x=0, color='black', linestyle='--', label="Start")
                    # End (x = Duration (s))
                    ax.axvline(x=data_row["Duration (s)"], color='blue', linestyle='--', label="End")
                    # Time of Max Peak (x = Time of Max Peak)
                    ax.axvline(x=data_row["Time of Max Peak"], color='red', linestyle='--', label="Max Peak")
                    
                    # Set y-axis limits based on computed global min and max.
                    ax.set_ylim([ymin, ymax])
                    ax.set_xlabel("Relative Time (s)")
                    ax.set_title(f"Trial {trial_name} - Bout {data_row['Bout']}")

                    # Ensure y-axis tick labels are visible on every subplot.
                    ax.tick_params(axis='y', labelleft=True)
                    # Add y-label and legend for the first column.
                    if col_idx == 0:
                        ax.set_ylabel("Z-score")
                        ax.legend()
                else:
                    ax.axis('off')

        plt.tight_layout()
        plt.show()


    def plot_average_PETHs_per_trial(self, selected_bouts=None):
        """
        For each trial, compute the average Investigation PETH (with SEM) for each subject group 
        ("Novel" and "Subject") and plot them in separate figures.

        Parameters:
        selected_bouts (list, optional): A list of bout identifiers to include. If None, all bouts are used.
        """
        # Iterate over each trial.
        for trial_name, trial in self.trials.items():
            if not hasattr(trial, 'behaviors'):
                continue
            
            # Filter Investigation events.
            df_invest = trial.behaviors[trial.behaviors["Behavior"] == "Investigation"].copy()
            if selected_bouts is not None:
                df_invest = df_invest[df_invest["Bout"].isin(selected_bouts)]
            
            if df_invest.empty:
                continue
            
            # Group events by subject.
            subject_groups = {}
            for idx, row in df_invest.iterrows():
                subj = row["Subject"]
                # Ensure the time axis and Z-score trace are numpy arrays.
                x = np.array(row["Relative_Time_Axis"])
                y = np.array(row["Relative_Zscore"])
                if subj not in subject_groups:
                    subject_groups[subj] = {"x": x, "y_list": [y]}
                else:
                    subject_groups[subj]["y_list"].append(y)
            
            # For each subject group, compute the average and SEM, and plot in a separate figure.
            for subj, data in subject_groups.items():
                y_stack = np.vstack(data["y_list"])
                avg_y = np.mean(y_stack, axis=0)
                # Compute SEM (using ddof=1 for an unbiased estimator if more than one bout is available)
                sem_y = np.std(y_stack, axis=0, ddof=1) / np.sqrt(y_stack.shape[0]) if y_stack.shape[0] > 1 else np.zeros_like(avg_y)
                
                # Select a color based on the subject.
                color = "green" if subj == "Novel" else "blue"
                
                plt.figure(figsize=(8, 6))
                plt.plot(data["x"], avg_y, label=f"{subj} Average", color=color)
                plt.fill_between(data["x"], avg_y - sem_y, avg_y + sem_y, color=color, alpha=0.3)
                plt.axvline(x=0, color="black", linestyle="--", label="Onset")
                plt.title(f"Average Investigation PETH - Trial {trial_name} ({subj})")
                plt.xlabel("Relative Time (s)")
                plt.ylabel("Average Z-score")
                plt.legend()
                plt.tight_layout()
                plt.show()


    def plot_average_PETHs_experiment(self, selected_bouts=None):
        """
        Compute and plot the average Investigation PETH (with SEM) for the entire experiment,
        separately for each subject group ("Novel" and "Subject"). Each subject group is plotted in its
        own figure.

        Parameters:
        selected_bouts (list, optional): A list of bout identifiers to include. If None, all bouts are used.
        """
        # Gather Investigation events from all trials.
        subject_groups = {}
        for trial_name, trial in self.trials.items():
            if not hasattr(trial, 'behaviors'):
                continue
            df_invest = trial.behaviors[trial.behaviors["Behavior"] == "Investigation"].copy()
            if selected_bouts is not None:
                df_invest = df_invest[df_invest["Bout"].isin(selected_bouts)]
            if df_invest.empty:
                continue
            
            for idx, row in df_invest.iterrows():
                subj = row["Subject"]
                x = np.array(row["Relative_Time_Axis"])
                y = np.array(row["Relative_Zscore"])
                if subj not in subject_groups:
                    subject_groups[subj] = {"x": x, "y_list": [y]}
                else:
                    subject_groups[subj]["y_list"].append(y)
        
        if not subject_groups:
            print("No data available for experiment average PETH plotting.")
            return
        
        # For each subject group, compute the average and SEM and plot in a separate figure.
        for subj, data in subject_groups.items():
            y_stack = np.vstack(data["y_list"])
            avg_y = np.mean(y_stack, axis=0)
            sem_y = np.std(y_stack, axis=0, ddof=1) / np.sqrt(y_stack.shape[0]) if y_stack.shape[0] > 1 else np.zeros_like(avg_y)
            
            color = "green" if subj == "Novel" else "blue"
            
            plt.figure(figsize=(8, 6))
            plt.plot(data["x"], avg_y, label=f"{subj} Average", color=color)
            plt.fill_between(data["x"], avg_y - sem_y, avg_y + sem_y, color=color, alpha=0.3)
            plt.axvline(x=0, color="black", linestyle="--", label="Onset")
            plt.title(f"Average Investigation PETH - Experiment ({subj})")
            plt.xlabel("Relative Time (s)")
            plt.ylabel("Average Z-score")
            plt.legend()
            plt.tight_layout()
            plt.show()

    '''********************************** DOPAMINE SHIZ **********************************'''
    def compute_all_da_metrics(self, use_max_length=False, max_bout_duration=10, 
                            use_adaptive=False, allow_bout_extension=False, mode='standard'):
        """
        Iterates over all trials in the experiment and computes DA metrics with the specified windowing options.
        
        Parameters:
        - use_max_length (bool): Whether to limit the window to a maximum duration.
        - max_bout_duration (int): Maximum allowed window duration (in seconds) if fractional analysis is applied.
        - use_adaptive (bool): Whether to adjust the window using adaptive windowing (via local minimum detection).
        - allow_bout_extension (bool): Whether to extend the window if no local minimum is found.
        - mode (str): Either 'standard' to compute metrics using the full standard DA signal, or 'EI' to compute metrics
                        using the event-induced data (i.e. the precomputed 'Event_Time_Axis' and 'Event_Zscore' columns).
        """
        
        for trial_name, trial in self.trials.items():
            if hasattr(trial, 'compute_da_metrics'):
                print(f"Computing DA metrics for {trial_name} ...")
                trial.compute_da_metrics(use_max_length=use_max_length,
                                        max_bout_duration=max_bout_duration,
                                        use_adaptive=use_adaptive,
                                        allow_bout_extension=allow_bout_extension,
                                        mode=mode)
            else:
                print(f"Warning: Trial '{trial_name}' does not have compute_da_metrics method.")



    def compute_all_event_induced_DA(self, pre_time=4, post_time=15):
        """
        Iterates over all trials in the experiment and computes the event-induced DA signals
        for each trial by calling each Trial's compute_event_induced_DA() method.
        
        Parameters:
        - pre_time (float): Seconds to include before event onset.
        - post_time (float): Seconds to include after event onset.
        """
        for trial_name, trial in self.trials.items():
            print(f"Computing event-induced DA for trial {trial_name} ...")
            trial.compute_event_induced_DA(pre_time=pre_time, post_time=post_time)



    '''********************************** MISC **********************************'''
    def reset_all_behaviors(self):
        """
        Sets each trial's 'behaviors' DataFrame to empty, so you can re-run
        group_extract_manual_annotations with different parameters.
        """
        for trial in self.trials.values():
            trial.behaviors = pd.DataFrame()