import numpy as np
import matplotlib.pyplot as plt


from glob import glob
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Excel file
files_daily = glob("Data_daily/*xlsx")
df_list = []
for f in files_daily:
    df = pd.read_excel(f, sheet_name = None)
    print(f"File: {f}")
    # Combine all sheets, adding the sheet name as new feature 'Zone'
    for zone_name, data in df.items():
        data = data.copy()
        df_list.append(data)
# Merge all zones into one dataframe
df = pd.concat(df_list, ignore_index=True)



# 2. Select relevant columns
features = ['Date', 'DT', 'Energy', 'Peak_Demand', 'Peak_Hour', 'Peak_DB', 'Peak_DP', 'Peak_WTHI', 'Min_Demand']
df = df[features]


# 3. Handle missing values (drop rows with NaN)
df = df.dropna()

# 4. Encode categorical column 'Zone' if it's not numeric
    
target = 'Peak_Demand'
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

df = df.dropna(subset=['Date'])

df['Year']  = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day']   = df['Date'].dt.day

df = df.drop(columns=['Date'])
X = df.drop(columns=[target])
y = df[target]

# 6. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Initialize and train model
scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

svm_regressor = SVR(kernel='linear', C=1.0)
svm_regressor.fit(X_train_scaled, y_train)



# 8. Make predictions
y_pred = svm_regressor.predict(X_test_scaled)

# 9. Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

# 10. Optionally, show sample predictions
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'error': y_test - y_pred})
print(results.head(10))
