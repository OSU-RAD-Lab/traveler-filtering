
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

class Curves:

    def __init__(self, data_src_folder_path:str, plot_dst_folder_path:str, plot_color:str = 'black'):
        self.data_src_folder_path = data_src_folder_path
        self.plot_dst_folder_path = plot_dst_folder_path
        self.plot_color = plot_color
        self.filenames= []
        self.curve_data = []

    def get_curve_data(self):
        for filename in os.listdir(self.data_src_folder_path):
            df = pd.read_csv(f"{self.data_src_folder_path}/{filename}", skiprows=2)
            df = df[['toeforce_y', 'toe_position_y']] # takes just the two important columns
            df.columns = ["resistance", "depth"] # rename columns
            self.curve_data.append(df)
            self.filenames.append(filename)
    
    def flip_curve_over_yaxis(self):
        cleaned_df_list = []
        for df in self.curve_data:
            copy_df = df.copy()
            copy_df['depth'] = -copy_df['depth']
            cleaned_df_list.append(copy_df)
        self.curve_data = cleaned_df_list

    def flip_over_x_axis(self):
        cleaned_df_list = []
        for df in self.curve_data:
            copy_df = df.copy()
            copy_df['resistance'] = -copy_df['resistance']
            cleaned_df_list.append(copy_df)
        self.curve_data = cleaned_df_list

    def remove_points_after_max_depth(self):
        cleaned_list = []
        for i, df in enumerate(self.curve_data):
            end_idx = df[df["depth"] == df["depth"].max()].index[0]
            cleaned_df = df.iloc[:end_idx+1]
            cleaned_list.append(cleaned_df)
        self.curve_data = cleaned_list
    
    def remove_points_before_min_depth(self):
        cleaned_list = []
        for i, df in enumerate(self.curve_data):
            min_idx = df[df["depth"] == df["depth"].min()].index[0]
            cleaned_df = df.iloc[min_idx:]
            cleaned_list.append(cleaned_df)
        self.curve_data = cleaned_list
        
    def make_resistance_min_equal_zero(self):
        cleaned_df_list = []
        for df in self.curve_data:
            copy_df = df.copy()
            copy_df["resistance"] = copy_df["resistance"].clip(lower=0)
            cleaned_df_list.append(copy_df)
        self.curve_data = cleaned_df_list

    def find_positive_subranges_of_resistance(self, df: pd.DataFrame):
        ranges_above_zero_list = []
        range_max_height_list = []
    
        in_range = False
        range_start_idx = None
        range_max_resistance = 0
    
        for i, res in enumerate(df["resistance"]):
            if res > 0:
                if not in_range:
                    # starting a new range
                    in_range = True
                    if i > 0: range_start_idx = i - 1
                    else: range_start_idx = 0
                    range_max_resistance = res
                else:
                    range_max_resistance = max(range_max_resistance, res)
            elif in_range:
                # end of a positive range
                ranges_above_zero_list.append((range_start_idx, i))
                range_max_height_list.append(range_max_resistance)
                in_range = False
    
        # handle if last element was part of a range
        if in_range:
            ranges_above_zero_list.append((range_start_idx, len(df["resistance"]) - 1))
            range_max_height_list.append(range_max_resistance)
    
        return ranges_above_zero_list, range_max_height_list

    def filter_subranges(self, subrange_list, subrange_max_resistance_list, subrange_max_resistance):
        max_resistance_overall = max(subrange_max_resistance_list)
        filtered_subranges = []
        for i, pos_range in enumerate(subrange_list):
            if subrange_max_resistance_list[i] > max_resistance_overall * subrange_max_resistance:
                filtered_subranges.append(pos_range)
        return filtered_subranges
    
    def get_ground_start_idx(self, df, subrange_max_resistance, spacing_between_ranges, idx):
        subrange_list, subrange_max_resistance_list = self.find_positive_subranges_of_resistance(df)

        if len(subrange_list) < 1: return 0
        
        # removes subranes below subrange_max_resistance threshold
        filtered_subranges = self.filter_subranges(subrange_list, subrange_max_resistance_list, subrange_max_resistance)
        ground_start_idx = filtered_subranges[-1][0] # init ground_start_idx with start of largest curve (last subrange in range_list)
        if len(filtered_subranges) < 2: return ground_start_idx


        # reverse iterate over the filtered subranges and stop when the distance from subrange i to j is too high
        for i in range(len(filtered_subranges)-2, -1, -1): 
            subrange_i_start = df["depth"].iloc[filtered_subranges[i][1]]
            subrange_j_end = df["depth"].iloc[filtered_subranges[i+1][0]]
            if subrange_j_end - subrange_i_start > spacing_between_ranges * (df['depth'].iloc[-1] - df['depth'].iloc[0]):
                ground_start_idx = filtered_subranges[i+1][0]
                break # found our final ground_start_idx
            else:
                ground_start_idx = filtered_subranges[i][0]
        return ground_start_idx

    def remove_data_prior_to_ground(self, subrange_max_resistance, spacing_between_ranges):
        cleaned_df_list = []
        for idx, df in enumerate(self.curve_data):
            copy_df = df.copy()
            start_idx = self.get_ground_start_idx(copy_df, subrange_max_resistance, spacing_between_ranges, idx)
            copy_df = copy_df.iloc[start_idx:]
            # copy_df = start_curve_at_ground(copy_df, subrange_max_resistance, spacing_between_ranges)
            copy_df["depth"] = copy_df["depth"] - copy_df['depth'].iloc[0]
            cleaned_df_list.append(copy_df)
        self.curve_data = cleaned_df_list
    
    def interpolate(self, num_points):
        interp_df_list = []
        for df in self.curve_data:
            x_intervals = np.linspace(0, df['depth'].max(), num_points, endpoint=True) # 100 points between 0 and trunc_level
            y_new = np.interp(x_intervals, df["depth"], df["resistance"])
            new_df = pd.DataFrame({'depth': x_intervals, 'resistance': y_new})
            interp_df_list.append(new_df)
        self.curve_data = interp_df_list

    def func(self, x, a):
        return a * x

    def plot(self):
        for i in range(len(self.curve_data)):
            plt.figure(figsize=(4,4))
            plt.xlabel('Depth (m)')
            plt.ylabel('Resistance (N)')
            plt.title(self.filenames[i], fontsize=8)
# 
            resist = self.curve_data[i]["resistance"]
            depth = self.curve_data[i]["depth"]

            plt.plot(depth, resist, c=self.plot_color, linewidth=2)
            opt_slope, _ = curve_fit(self.func, depth, resist)
            plt.plot(depth, self.func(depth, opt_slope), 'r--', label=f"slope: {round(opt_slope[0], 2)}")
            plt.legend()
            
            plt.savefig(f'{self.plot_dst_folder_path}/{self.filenames[i]}.png')
            # plt.show()

def main():
    if len(sys.argv) != 4:
        print(f'incorrect number of arguments given')
        print('Correct Format:\n\tpython3 curve_slope.py data_src_folder_path plot_dst_folder_path color')
        sys.exit()

    # Create the data object
    curves = Curves(sys.argv[1], sys.argv[2], sys.argv[3])
    
    # load the data in 
    curves.get_curve_data()

    # clean the data
    curves.flip_curve_over_yaxis()
    # curves.flip_over_x_axis() # this is needed depending on how data is formatted
    curves.remove_points_after_max_depth()
    curves.remove_points_before_min_depth()
    curves.make_resistance_min_equal_zero()
    curves.remove_data_prior_to_ground(0.1, 0.05)
    curves.interpolate(500)

    # plot the data
    curves.plot()


if __name__ == "__main__":
    main()