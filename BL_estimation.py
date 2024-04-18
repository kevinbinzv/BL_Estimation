
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import random


#------------------------------------------------------------#FUNCTIONS------------------------------------------------------------------------#

def calculate_aggregated_baseline_rowise(dataframe, start_row, end_row, new_row_name):
    """
    Sums values across specified columns, adds this sum as a new column, and calculates the total sum of this new column.

    Parameters:
    - dataframe: Pandas DataFrame with the data.
    - start_col: str, the name of the first column in the range to be summed.
    - end_col: str, the name of the last column in the range to be summed.
    - new_col_name: str, the name of the new column to store the sum across specified columns.

    Returns:
    - The modified DataFrame with the new column.
    - The total sum of the new column.
    """
    
    dataframe[new_row_name] = dataframe.loc[start_row:end_row, :].sum(axis=1)
    total_sum_row = dataframe[new_row_name].sum()
    return dataframe, f"Aggregated Baseline Load for the 32 customers: {total_sum_row}"

def select_and_rename_columns(data, indices, prefix='BL_'):
    """
    Randomly selects 20 columns based on the provided indices, which are assumed to already correspond to the full column names with a prefix, and returns a new DataFrame with these selected columns.
    
    Parameters:
        data (pd.DataFrame): The DataFrame from which columns are to be selected.
        indices (list): A list of column indices to select from. These are not zero-based indices, but direct identifiers included in the column names.
        prefix (str, optional): The prefix included in the column names. Defaults to 'BL_'.
    
    Returns:
        pd.DataFrame: A new DataFrame with 20 randomly selected columns.
    """
    # Generate full column names based on the indices and prefix
    full_column_names = [prefix + str(index) for index in indices]
    
    # Check if the specified column names exist in the DataFrame
    missing_columns = [name for name in full_column_names if name not in data.columns]
    if missing_columns:
        raise ValueError(f"Some column names are missing in the DataFrame: {missing_columns}")
    
    # Randomly select 20 column names from the list of full column names
    selected_column_names = random.sample(full_column_names, 20)
    
    # Select the columns by name
    selected_df = data[selected_column_names].copy()
    
    # Return the newly created DataFrame
    return selected_df

def randomise_abl(data, indices, start_row, end_row, new_col_name, prefix='BL_'):
    """
    Selects specified columns using select_and_rename_columns function,
    and then aggregates the data row-wise over a specified row range using calculate_aggregated_baseline_rowise function.

    Parameters:
    - data: Pandas DataFrame from which to select columns and calculate sums.
    - indices: list of column indices (1-based) to include in the selection.
    - start_row: int, the index of the first row for summation.
    - end_row: int, the index of the last row for summation.
    - new_col_name: str, the name for the new column storing the row-wise sum.
    - prefix: str, optional, the prefix for column names. Defaults to 'BL_'.

    Returns:
    - The modified DataFrame with aggregated data and the total sum of the new column.
    """
    # Select and rename columns from the DataFrame
    selected_df = select_and_rename_columns(data, indices, prefix)
    
    # Calculate row-wise sums and add as a new column
    aggregated_df, total_sum = calculate_aggregated_baseline_rowise(selected_df, start_row, end_row, new_col_name)
    
    return aggregated_df, total_sum


def calculate_baseline(df, selected_date, x, periods=10, freq='B', acceptance_ratio=0.75, method='high', ):
    if selected_date not in df.index:
        latest_date = df.index[df.index < selected_date].max()
        print(f"Selected date is not in the data range. Using {latest_date} instead.")
        selected_date = latest_date

    # Filter to get data only up to the day before the selected date
    df_prior = df[df.index < selected_date]

    # Collect the last business days data
    business_days = pd.bdate_range(end=selected_date - pd.Timedelta(days=1), periods=periods, freq=freq)
    df_business_days = df_prior[df_prior.index.normalize().isin(business_days)]

    # Calculate ADEPU for each day
    ADEPU = df_business_days.resample('D').sum()

    # Calculate ADEPL
    ADEPL = ADEPU.mean()

    # Calculate the ratio for selection criteria and filter days
    selection = ADEPU / ADEPL
    acceptable_days = selection[selection >= acceptance_ratio].index

    if len(acceptable_days) < 5:
        print("Not enough acceptable days to calculate the baseline.")
        return None, None
    else:
        # Select the top days by ADEPU based on the method
        if method == 'high':
            top_days = ADEPU.loc[acceptable_days].nlargest(x).index
        elif method == 'mid':
            # Calculate median index and select days around the median
            sorted_days = ADEPU.loc[acceptable_days].sort_values()
            mid_index = len(sorted_days) // 2
            top_days = sorted_days.iloc[mid_index-2:mid_index+3].index
        elif method == 'low':
            top_days = ADEPU.loc[acceptable_days].nsmallest(x).index

        # Calculate the average for each half-hour period over the selected top days
        subset_top_days = df[df.index.normalize().isin(top_days)]
        baseline = subset_top_days.groupby(subset_top_days.index.time).mean()

        return baseline, top_days


def calculate_weekend_baseline(df, selected_date, num_weekends, x=5, acceptance_ratio=0.75, method='high'):

    if selected_date.dayofweek not in [5, 6]:  # 5=Saturday, 6=Sunday
        print("The selected date is not on a weekend. Choose a Saturday or Sunday.")
        return None
       
    weekend_dates = collect_weekends(selected_date, num_weekends)
    df_weekends = df[df.index.normalize().isin(weekend_dates)]

    # Calculate ADEPU for each weekend day
    ADEPU = df_weekends.resample('D').sum()

    # Calculate ADEPL
    ADEPL = ADEPU.mean()

    # Calculate the ratio for selection criteria and filter days
    selection = ADEPU / ADEPL
    acceptable_days = selection[selection >= acceptance_ratio].index

    if len(acceptable_days) < x:
        print("Not enough acceptable days to calculate the baseline.")
        return None

    # Select the top days by ADEPU based on the method
    if method == 'high':
        top_days = ADEPU.loc[acceptable_days].nlargest(x).index
    elif method == 'mid':
        sorted_days = ADEPU.loc[acceptable_days].sort_values()
        mid_index = len(sorted_days) // 2
        top_days = sorted_days.iloc[max(0, mid_index - x//2):min(len(sorted_days), mid_index + x//2 + 1)].index
    elif method == 'low':
        top_days = ADEPU.loc[acceptable_days].nsmallest(x).index

    # Calculate the average for each half-hour period over the selected top days
    subset_top_days = df[df.index.normalize().isin(top_days)]
    baseline = subset_top_days.groupby(subset_top_days.index.time).mean()

    return baseline

def add_date_to_time(baseline_series, date):
    # Convert the time index to string and prepend the date string
    date_str = date.strftime('%Y-%m-%d')
    datetime_index = pd.to_datetime([date_str + ' ' + str(time) for time in baseline_series.index])
    
    # Create a new series with the new datetime index
    new_baseline = pd.Series(baseline_series.values, index=datetime_index)
    return new_baseline

def collect_weekends(start_date, num_weekends):
    weekends = []
    current_date = start_date
    if current_date.dayofweek < 5:  # Adjust to the closest previous Friday if it's a weekday
        current_date -= pd.Timedelta(days=(current_date.dayofweek + 1))
    for _ in range(num_weekends):
        saturday = current_date - pd.Timedelta(days=1)  # Saturday
        sunday = current_date  # Sunday (current_date is adjusted to Friday)
        weekends.extend([saturday, sunday])
        current_date -= pd.Timedelta(weeks=1)  # Move to the previous weekend
    return weekends

def check_day_type(date):
    if not isinstance(date, pd.Timestamp):
        date = pd.to_datetime(date)

    # Check the day of the week; .dayofweek returns an integer, where Monday is 0 and Sunday is 6
    if date.dayofweek < 5:  # From Monday (0) to Friday (4)
        return 'Weekday'
    else:  # Saturday (5) and Sunday (6)
        return 'Weekend'
    
def adjust_date_for_weekend(selected_date):
    if selected_date.dayofweek >= 5:  # 5 for Saturday, 6 for Sunday
        return pd.offsets.BDay(1).rollforward(selected_date)
    return selected_date

def plot_energy_consumption(baseline, actual_data, selected_date):
    plt.figure(figsize=(12, 6))
    
    # Setup for plotting
    times = pd.date_range(selected_date, periods=len(baseline), freq='30T').to_numpy()
    baseline_values = baseline.values
    actual_times = actual_data.index.to_numpy()
    actual_values = actual_data.values
    
    plt.plot(times, baseline_values, label='Baseline', color='blue')
    plt.plot(actual_times, actual_values, label='Recorded Consumption', color='red', linestyle='--')
    
    plt.title('Baseline vs Actual Energy Consumption on ' + str(selected_date.date()))
    plt.xlabel('Time of Day')
    plt.ylabel('Energy Consumption')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_baseline_peak_hours(baseline, actual_data, selected_date):
    plt.figure(figsize=(12, 6))
    
    # Since baseline is already a Series with a DateTimeIndex, we just filter it directly
    # Define the time range for displaying baseline data
    time_start = pd.Timestamp(selected_date).replace(hour=6, minute=0, second=0)
    time_end = pd.Timestamp(selected_date).replace(hour=12, minute=30, second=0)

    # Filter the baseline data within the specified time range using .loc[]
    filtered_baseline = baseline.loc[time_start:time_end]

    # Plot baseline for the specified time range
    plt.plot(filtered_baseline.index.to_numpy(), filtered_baseline.values, label='Baseline [06:00 to 12:30]]', color='blue')

    # Assuming actual_data is also a DataFrame with a datetime index and it contains a specific column for values
    # Ensure to specify the correct column name in actual_data for plotting
    plt.plot(actual_data.index.to_numpy(), actual_data.values, label='Recorded Consumption', color='red', linestyle='--')  # Replace 'column_name' with the actual column name

    # Set plot titles and labels
    plt.title('Baseline vs Actual Energy Consumption on Peak Hours')
    #' + str(selected_date.date())
    plt.xlabel('Time of Day')
    plt.ylabel('Energy Consumption')
    plt.xticks(rotation=45)
    
    # Additional plot settings
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def cvrmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mean_observed = np.mean(y_true)
    cvrmse_value = (rmse / mean_observed) * 100 if mean_observed != 0 else float('inf')
    return cvrmse_value

def mrbe(y_true, y_pred):
    epsilon = np.finfo(float).eps
    abs_error = np.abs(y_true - y_pred)
    mean_relative = np.mean(abs_error / (y_true + epsilon))  # Add epsilon to avoid division by zero
    return mean_relative * 100 

def calculate_error_metrics(baseline, actual_data):
    if baseline is not None and not actual_data.empty:
        # Ensure both Series are non-empty and have an intersection
        common_index = baseline.index.intersection(actual_data.index)
        if not common_index.empty:
            baseline_aligned = baseline.loc[common_index].dropna()
            actual_data_aligned = actual_data.loc[common_index].dropna()

            # Calculate error metrics
            cvrmse_value = cvrmse(actual_data_aligned.values, baseline_aligned.values)
            mrbe_value = mrbe(actual_data_aligned.values, baseline_aligned.values)

            return cvrmse_value, mrbe_value
        else:
            print(f"Warning: No common timestamps between baseline and actual data for method.")
            return float('nan'), float('nan')
    else:
        print(f"Error: Baseline is None or actual data is empty for method.")
        return float('nan'), float('nan')


#------------------------------------------------------------#MAIN WORKSPACE------------------------------------------------------------------------#

abl  = pd.read_excel('customers_53.xlsx')
abl.set_index('datetime', inplace = True )

households = [1, 5, 7, 9, 13, 19, 20, 28, 29, 30, 38, 39, 49, 50, 58, 61, 64, 67, 69, 70, 72, 84, 86, 90, 106, 127, 155, 158, 160, 165, 167, 169, 171, 177, 178, 184, 186, 202, 206, 215, 223, 224, 227, 232, 245, 246, 266, 267, 276, 286, 292, 297, 299]

random_abl, total_sum = randomise_abl(abl, households, '2012-07-01', '2013-06-30', 'ABL')

selected_date = pd.to_datetime('2013-05-11')

df = random_abl['ABL']  

# Adjust date for weekends
selected_date = adjust_date_for_weekend(selected_date)

# Calculate the baseline
baselineh5of10, top_days = calculate_baseline(df, selected_date, 5,  periods = 10, method = 'high' )
baselineh5of10 = add_date_to_time(baselineh5of10,  selected_date)
if baselineh5of10 is not None:
    # Retrieve actual data if it exists
    if selected_date in df.index:
        actual_data = df[df.index.date == selected_date.date()]
        plot_energy_consumption(baselineh5of10, actual_data, selected_date)
    else:
        print("No actual data for the selected date.")

cvrmse_h5of10, mrbe_h5of10 = calculate_error_metrics(baselineh5of10, actual_data)
print(f"High (5 of 10) - CVRMSE: {cvrmse_h5of10}, MRBE: {mrbe_h5of10}")