import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# TODO: get data from DataProcessing and assign an X and y

# Split data either 80/20?
# Currently data is split 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the Features
scaler = StandardScaler()
#X_train_scaled =
#X_test_scaled =

# Train the SVM classifier

