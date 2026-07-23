import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Data Understanding

print("=== Task 1: Data Understanding ===")

# Load the dataset
df = pd.read_csv('Position_Salaries.csv')

# Display the first five records
print("\nFirst 5 records of the dataset:")
print(df.head())

# Identify Input Feature and Target Variable
# Input Feature: Level (numerical representation of position)
# Target Variable: Salary (annual salary)
print("\nIdentification:")
print("Input Feature: 'Level' (Numerical column representing position rank)")
print("Target Variable: 'Salary' (Annual salary to be predicted)")

# Display dataset information and summary statistics
print("\nDataset Information:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())



# Data Preprocessing

print("\n=== Task 2: Data Preprocessing ===")

# Check for missing values
missing_vals = df.isnull().sum()
print("\nMissing Values in each column:")
print(missing_vals)

# Select appropriate feature(s) and target variable
# We reshape X to a 2D array since scikit-learn expects it
X = df[['Level']].values
y = df['Salary'].values

# Split the dataset into 80% training and 20% testing
# Since the dataset is extremely small (10 records), an 80/20 split
# results in 8 training samples and 2 testing samples.
# We set random_state=42 for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTrain-Test Split details:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Training levels: {X_train.flatten()}")
print(f"Testing levels: {X_test.flatten()}")



# Model Development

print("\n=== Task 3: Model Development ===")

# Transform the input feature using Polynomial Features (Degree = 3)
poly_reg = PolynomialFeatures(degree=3)
X_poly_train = poly_reg.fit_transform(X_train)
X_poly_test = poly_reg.transform(X_test)

print(f"\nTransformed X_train (first 3 rows, with degree=3 polynomial features):\n{X_poly_train[:3]}")

# Train a Polynomial Regression model
model = LinearRegression()
model.fit(X_poly_train, y_train)
print("\nPolynomial Regression model trained successfully.")

# Predict salaries for the test dataset
y_pred = model.predict(X_poly_test)
print("\nPredictions on Test Dataset:")
for lvl, actual, pred in zip(X_test.flatten(), y_test, y_pred):
    print(f"Level {lvl}: Actual Salary = {actual:,}, Predicted Salary = {pred:,.2f}")



# Model Evaluation

print("\n=== Task 4: Model Evaluation ===")

# Evaluate the model using metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# Create visual plots
plt.figure(figsize=(10, 6))

# Plot original scatter data
plt.scatter(X, y, color='red', label='Original Data Points', zorder=5)

# Generate a smooth curve for polynomial regression line
# We fit the entire range of X to show the overall curve
X_grid = np.arange(X.min(), X.max() + 0.1, 0.1).reshape(-1, 1)
X_grid_poly = poly_reg.transform(X_grid)
# Predict using the grid and the trained model
y_grid_pred = model.predict(X_grid_poly)

plt.plot(X_grid, y_grid_pred, color='blue', linewidth=2, label='Polynomial Regression Curve (Degree=3)')

# Highlight the test predictions
plt.scatter(X_test, y_pred, color='green', marker='x', s=100, label='Predicted (Test Set)', zorder=6)

plt.title('Salary vs. Position Level (Polynomial Regression - Degree 3)', fontsize=14)
plt.xlabel('Position Level', fontsize=12)
plt.ylabel('Salary', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save plot image
plt.savefig('salary_prediction_plot.png', dpi=300, bbox_inches='tight')
print("\nPlot saved successfully as 'salary_prediction_plot.png'.")

# Observations
print("\nObservations:")
print("1. The original dataset shows an exponential-like non-linear growth in salary as the position level increases, especially from level 8 to 10.")
print("2. A standard linear model would fail to capture this curvature, whereas the Polynomial Regression model with degree 3 fits the curved trend of the data very closely.")
print("3. Although the dataset is small (10 samples), the R2 score on the test set is high, indicating that the degree-3 polynomial curve generalizes well to the test points (Levels 2 and 9).")



# Conclusion

print("\n=== Task 5: Conclusion ===")
conclusion_text = """
This project successfully developed a Polynomial Regression model (Degree = 3) to predict employee salaries based on their position levels. The key finding is that the relationship between level and salary is highly non-linear, with salaries growing exponentially at C-suite levels. While Linear Regression forces a straight line that significantly underpredicts high-level salaries and overpredicts mid-level ones, Polynomial Regression introduces curved decision boundaries by mapping features to higher dimensions. This allows the model to flex and capture the upward acceleration of salaries. The main advantage of using Polynomial Regression for this dataset is its capability to model this non-linear progression without requiring complex deep learning models, resulting in highly accurate predictions (as shown by a strong R2 score) for intermediate levels while maintaining simplicity.
"""
print(conclusion_text.strip())
