'''
However, this could lead to data breaches, and it is hard to maintain

So, I decide to use pipeline for model tuning.

Not only that, I want to add hyerparameters to optimize my model, 
and find out which parameters is the best, such as Ridge alpha and poly degree.
'''


# pipe
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures()),
    ("ridge", Ridge())
])

# Set the default
param_grid = {
    "poly__degree": [1, 2],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100]
}

# Use grid search the best parameter
grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1
)

# Here is same as what I did before.
grid.fit(X_train, Y_train)

best_model = grid.best_estimator_

val_pred = best_model.predict(X_val)
mse = mean_squared_error(Y_val, val_pred)
r2 = r2_score(Y_val, val_pred)

print("greatest hypa：", grid.best_params_)
print(f"After Optimize MSE: {mse:.3f}, R2: {r2:.3f}")
'''
greatest hypa： {'poly__degree': 2, 'ridge__alpha': 10}
After Optimize MSE: 3.746, R2: 0.708
'''


# Base on the best parameters, we can do our final test validation.
y_test_pred = best_model.predict(X_test)

MSE_best_test = mean_squared_error(Y_test, y_test_pred)
R2_best_test = r2_score(Y_test, y_test_pred)

print("Best Hyperparameters:", grid.best_params_)
print(f"Test MSE: {MSE_best_test:.4f}")
print(f"Test R2: {R2_best_test:.4f}")
'''
Best Hyperparameters: {'poly__degree': 2, 'ridge__alpha': 10}
Test MSE: 4.0817
Test R2: 0.6966
'''


'''
Why it is like what I did on step by step?
  the features are not that much
  mismatch of model complexity and true data pattern
  Maybe just this dataset shows the strong linear relationship
'''
















