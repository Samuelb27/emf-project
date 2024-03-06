import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# Read the Excel file into a DataFrame
df = pd.read_excel('Data_Project1.xlsx', sheet_name='Entertainment', header=1)

# Drop any rows with missing values
df.dropna(inplace=True)

# Convert the desired columns to numeric type
df.iloc[2, 1:5] = pd.to_numeric(df.iloc[2, 1:5], errors='coerce')

# Drop any rows with missing or invalid values after conversion
df.dropna(inplace=True)

#1)

# Calculate daily simple returns for each column
daily_simple_returns_list = []
for column in df.columns[1:]:
    daily_simple_returns = df[column].pct_change()
    daily_simple_returns_list.append(daily_simple_returns)

# Concatenate daily simple returns for all columns
all_daily_simple_returns = pd.concat(daily_simple_returns_list, axis=1)
#print(all_daily_simple_returns.head())

# Calculate summary statistics for all daily simple returns
summary_daily_simple_stats = all_daily_simple_returns.describe()
summary_daily_simple_stats.loc['skewness'] = all_daily_simple_returns.skew()
summary_daily_simple_stats.loc['kurtosis'] = all_daily_simple_returns.kurtosis()

# Print summary statistics for daily simple returns
print("Daily simple return summary statistics :\n", summary_daily_simple_stats)

# Calculate daily log returns for each column
daily_log_returns_list = []
for column in df.columns[1:]:
    daily_log_returns = df[column].pct_change().apply(lambda x: np.log(1+x))  # Calculate daily log returns
    daily_log_returns_list.append(daily_log_returns)

# Concatenate daily log returns for all columns
all_log_returns = pd.concat(daily_log_returns_list, axis=1)

# Calculate summary statistics for all daily log returns
summary_daily_log_stats = all_log_returns.describe()
summary_daily_log_stats.loc['skewness'] = all_log_returns.skew()
summary_daily_log_stats.loc['kurtosis'] = all_log_returns.kurtosis()

# Print summary statistics for daily log returns
print("Daily log return summary statistics :\n", summary_daily_log_stats)

# Weekly
df_weekly = pd.DataFrame()

# Iterate over each column
for column in df.columns[1:]:  # Start from index 1 to skip the first column which might be the date column
    # Select every 5th value starting from index 3
    selected_values = df[column][3::5]
    # Add the selected values to the result DataFrame
    df_weekly[column] = selected_values

# Calculate weekly returns for each column
weekly_returns_list = []
for column in df_weekly.columns[0:]:
    weekly_returns = df_weekly[column].pct_change()  # Calculate weekly percentage change
    weekly_returns_list.append(weekly_returns)

# Concatenate weekly returns for all columns
all_weekly_returns = pd.concat(weekly_returns_list, axis=1)
print(all_weekly_returns)

# Calculate summary statistics for all weekly returns
summary_weekly_stats = all_weekly_returns.describe()
summary_weekly_stats.loc['skewness'] = all_weekly_returns.skew()
summary_weekly_stats.loc['kurtosis'] = all_weekly_returns.kurtosis()

# Print summary statistics for weekly simple returns
print("Weekly simple return summary statistics:\n", summary_weekly_stats)

# Calculate weekly log returns for each column
weekly_log_returns_list = []
for column in df_weekly.columns[0:]:
    weekly_log_returns = df_weekly[column].pct_change().apply(lambda x: np.log(1+x))  # Calculate weekly log returns
    weekly_log_returns_list.append(weekly_log_returns)

# Concatenate weekly log returns for all columns
all_weekly_log_returns = pd.concat(weekly_log_returns_list, axis=1)

# Calculate summary statistics for all weekly log returns
summary_weekly_log_stats = all_weekly_log_returns.describe()
summary_weekly_log_stats.loc['skewness'] = all_weekly_log_returns.skew()
summary_weekly_log_stats.loc['kurtosis'] = all_weekly_log_returns.kurtosis()

# Print summary statistics for weekly log returns
print("Weekly log return summary statistics:\n", summary_weekly_log_stats)

#2)

# compute log price
df_log = pd.DataFrame()

# Iterate over each column except the first one (assuming the first column is the date column)
for column in df.columns[1:]:
    # Compute the log price for the current column
    log_price = np.log(df[column])
    # Assign the computed log price to the new DataFrame df_log
    df_log[f'log_{column}'] = log_price

def run_dickey_fuller_test(column):
    result = adfuller(column, autolag='AIC')
    return result

for column in df_log.columns:  # Iterate over columns of df_log
    df_test_result = run_dickey_fuller_test(df_log[column])
    print("\nDickey-Fuller test for", column, "is:")
    print(f'Test Statistic: {df_test_result[0]}')
    print(f'p-value: {df_test_result[1]}')
    print(f'Critical Values:')
    for key, value in df_test_result[4].items():
        print(f'\t{key}: {value}')

    # Check if the test is accepted or rejected at different significance levels
    if df_test_result[1] < 0.01:
        print("Reject the null hypothesis; the data is stationary at the 1% level.")
    elif df_test_result[1] < 0.05:
        print("Reject the null hypothesis; the data is stationary at the 5% level.")
    elif df_test_result[1] < 0.1:
        print("Reject the null hypothesis; the data is stationary at the 10% level.")
    else:
        print("Failed to reject the null hypothesis; the data is non-stationary.")

#2.1)

