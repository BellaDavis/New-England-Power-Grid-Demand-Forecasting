from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date # For adding holiday information.
import holidays

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# TODO: No extra features yet, such as weather information and calendar events. JUST ISO NEW England Demand Reports

###################### DATA PRE-PROCESSING ######################

# Load Data files
files = glob("Data/Hourly/*xlsx")  # adjust path if needed
# Dataframe to store data
df_list = []

# List of the 8 SMD Load Zones in New England
Load_Zones = ['CT', 'ME', 'NEMA', 'NH', 'RI', 'SEMA', 'VT', 'WCMA']

for f in files:

    try:
        # Read all sheets
        df = pd.read_excel(f, sheet_name=None)
    except Exception as e:
        print(f"Error reading file {f}: {e}.")
        break

    # Combine data from all the load zones, adding the sheet name as new feature 'Zone'
    for zone_name, data in df.items():

        if zone_name in Load_Zones:

            # Clean columns before creating datetime.
            # Replace non int values with NaN and then set a default value. Ran into errors as some of the data is not correct.
            data['Hr_End'] = pd.to_numeric(data['Hr_End'], errors='coerce')
            data = data.dropna(subset=['Hr_End'])

            # Made into list in case we need to clean more than one col.
            cols = ['Hr_End']
            data.loc[:, cols] = data[cols].astype(int)


            # Create the DateTime Index
            def create_datetime(row):

                dt = pd.to_datetime(row['Date']) + pd.Timedelta(hours=row['Hr_End'])
                # Correct for hour ending 24
                if row['Hr_End'] == 24:
                    return dt - pd.Timedelta(hours=1)
                return dt


            data.loc[:, 'DateTime'] = data.apply(create_datetime, axis=1)

            # Select all other cols
            cols_to_keep = [col for col in data.columns if col not in ['Date', 'Hr_End', 'DateTime']]
            hourly_load = data.set_index('DateTime')[cols_to_keep]

            # Select the real-time demand and set the index.
            # hourly_load = data.set_index('DateTime')[['RT_Demand']]
            hourly_load = hourly_load.rename(columns={'RT_Demand': 'Load_MW'})

            # Add the zone name as a feature
            hourly_load['Zone'] = zone_name

            df_list.append(hourly_load)

# Merge all the zone data into a single dataframe
load_zone_df = pd.concat(df_list)
load_zone_df.sort_index(inplace=True)

# Estimate missing values
load_zone_df['Load_MW'] = load_zone_df['Load_MW'].interpolate(method='linear')

#print(f"Load Data Shape: {load_zone_df.shape}")
#print(load_zone_df)

# Plot power demand from each zone on a graph

# for zone, df_zone in load_zone_df.groupby('Zone'):
#     plt.plot(df_zone.index, df_zone['Load_MW'], label=zone)
#
# plt.title("ISO-New England Load by Zone")
# plt.xlabel("DateTime")
# plt.ylabel("Load (MW)")
# plt.legend()
# plt.show()
#
# # Plot one zone
# zone = 'VT' # Adjust if needed
#
# ct_df = load_zone_df[load_zone_df['Zone'] == zone]
# plt.plot(ct_df.index, ct_df['Load_MW'], label=zone)
# plt.title(f"{zone} Load MW ")
# plt.xlabel("DateTime")
# plt.ylabel("Load (MW)")
# plt.legend()
# plt.show()


###################### FEATURE ENGINEERING ######################

df_processed = load_zone_df.copy()

# Adding extra features such as if the day is a day of the week.
df_processed['Hour'] = df_processed.index.hour
df_processed['Day_of_Week'] = df_processed.index.weekday  # Monday=0 - Sunday=6

# TODO: Add Lag for features continuous features?

# One-Hot encoding.
cat_cols = ['Zone', 'Hour', 'Day_of_Week']
df_encoded = pd.get_dummies(df_processed, columns=cat_cols, drop_first=False)

# Add holiday columns where 0 is not a holiday, and 1 is a holiday

df_processed['Is_Holiday'] = 0

start_year = 2020 # Adjust years if needed
end_year = 2025

us_holidays = [] # Dict of US holidays

for year in range(start_year, end_year+1):
    for date, name in sorted(holidays.US(years=year).items()):
        us_holidays.append(f'{date}:{name}')

us_holidays = set(holidays.US(years=range(start_year, end_year + 1)).keys())

# Check for holidays in the data and set them to 1
df_processed.loc[df_processed.index.normalize().isin(us_holidays), 'Is_Holiday'] = 1

print(df_processed)

# Add the cooling degree day and heating degree days from the monthly data.
# TODO: Unable to find hourly weather data for the years and zones we need.
#   Maybe in another life it would have been recorded with the hourly demand data.

files = glob("Data/Monthly/*xlsx")  # adjust path if needed
# Dataframe to store data
monthly_data = []

# List of the 8 SMD Load Zones in New England
Load_Zones = ['CT', 'ME', 'NEMA', 'NH', 'RI', 'SEMA', 'VT', 'WCMA']

for f in files:
    try:
        # Read all sheets
        new_df = pd.read_excel(f, sheet_name=None)
    except Exception as e:
        print(f"Error reading file {f}: {e}.")
        break

    for zone_name, data in new_df.items():
            data = data.copy()
            if zone_name in Load_Zones:
                df_list.append(data)

new_df = pd.concat(df_list, ignore_index=True)

# Select relevant columns
features = ['Year', 'Month', 'CDD', 'HDD']

new_df = new_df[features]

# 3. Handle missing values (drop rows with NaN)
# TODO: Still a work in progress.

print(new_df)


















