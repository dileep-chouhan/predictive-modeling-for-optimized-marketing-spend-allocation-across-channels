import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_campaigns = 100
data = {
    'Channel': np.random.choice(['Facebook', 'Google', 'Email', 'TV'], size=num_campaigns),
    'Spend': np.random.uniform(100, 1000, size=num_campaigns),
    'Impressions': np.random.randint(1000, 100000, size=num_campaigns),
    'Clicks': np.random.randint(10, 1000, size=num_campaigns),
    'Conversions': np.random.randint(1, 100, size=num_campaigns),
    'Revenue': np.random.randint(100, 10000, size=num_campaigns)
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering ---
# Calculate ROI (Return on Investment)
df['ROI'] = (df['Revenue'] - df['Spend']) / df['Spend']
# One-hot encode categorical variable 'Channel'
df = pd.get_dummies(df, columns=['Channel'], drop_first=True)
# --- 3. Predictive Modeling ---
# Define features (X) and target (y)
X = df.drop(['Revenue', 'ROI'], axis=1)
y = df['Revenue']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# --- 4. Visualization ---
#Plot the actual vs predicted values.
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)], color='red') #Line of perfect prediction
plt.savefig('actual_vs_predicted.png')
print("Plot saved to actual_vs_predicted.png")
# Visualize feature importances (if applicable for the chosen model)
# This part is model-specific;  Linear Regression uses coefficients
feature_importances = pd.Series(model.coef_, index=X.columns)
plt.figure(figsize=(10,6))
feature_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.ylabel('Coefficient Magnitude')
plt.savefig('feature_importances.png')
print("Plot saved to feature_importances.png")