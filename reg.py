#linear regression
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 data points with a single feature
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with some noise

# Add a constant (intercept) to the model
X_b = sm.add_constant(X)  # Adds a column of ones to X

# Fit the model
model = sm.OLS(y, X_b)
results = model.fit()

# Print the model summary
print(results.summary())

# Make predictions
X_new = np.array([[0], [2]])
X_new_b = sm.add_constant(X_new)
y_predict = results.predict(X_new_b)

# Plot the results
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_new, y_predict, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with statsmodels')
plt.legend()
plt.show()
