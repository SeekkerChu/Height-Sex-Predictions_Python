# First, I will try to make the linear regression.
'''
Let monther, father and midparentHegiht be the X
Let childHeight be the Y
'''
X = numerical_df[['father', 'mother', 'sex_num', 'childNum']]
Y = numerical_df['childHeight']

# random seed
np.random.seed(7)

scaler = StandardScaler()

# set the data
X_train, X_rest, Y_train, Y_rest = train_test_split(X,Y, test_size = 0.4,
                                                    random_state = 42)

X_val, X_test, Y_val, Y_test = train_test_split(X_rest, Y_rest,
                                                test_size=0.5,
                                                random_state=42)

# scale the data
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# check the shape of them
print(f"THIS IS TRAIN: {X_train_scaled.shape}")
print(f"THIS IS VAL: {X_val_scaled.shape}")
print(f"THIS IS TEST: {X_test_scaled.shape}")
'''
THIS IS TRAIN: (560, 4)
THIS IS VAL: (187, 4)
THIS IS TEST: (187, 4)
'''

# Use train data set to fit the model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, Y_train)

'''
Here is just a linear model, but we should have another model to compare with
So, I am trying to make a polynomial model
'''
# set the degree = 2
poly = PolynomialFeatures(degree = 2)
'''
Interpreatation of degree: 
Let's say, if we have two features as our X: x1, x0
After degree = 2, it will be: a + x0^2 + x1^2 + x1x0
'''

# scale the data
X_train_poly = poly.fit_transform(X_train_scaled)
X_val_poly = poly.transform(X_val_scaled)
X_test_poly = poly.transform(X_test_scaled)

# fit the model
poly_model = LinearRegression()
poly_model.fit(X_train_poly,Y_train)

# Now, use it to validation data set, predict the value
linear_pred = linear_model.predict(X_val_scaled)
linear_pred_poly = poly_model.predict(X_val_poly)

# use MSE and R-squred to check
# MSE: Test the different mean value of  y_hat to y_true, smaller it has, the better the model 
mse_linear = mean_squared_error(Y_val, linear_pred)
mse_linear_poly = mean_squared_error(Y_val, linear_pred_poly)

# R2(Z-SOCRE): Between 0 to 1, means the model explains how much variaion in y
r2_linear = r2_score(Y_val, linear_pred)
r2_linear_poly = r2_score(Y_val, linear_pred_poly)

print(f"linear model MSE is {mse_linear:.3f}, and R2 is {r2_linear:.3f}")
print(f"poly model MSE is {mse_linear_poly:.3f}, and R2 is {r2_linear_poly:.3f}")
'''
linear model MSE is 3.775, and R2 is 0.706
poly model MSE is 3.746, and R2 is 0.708
'''
# which shows poly model is better to test

# Final test by Polynomial model
poly_predict = poly_model.predict(X_test_poly)

MSE_test = mean_squared_error(Y_test, poly_predict)
R2_test = r2_score(Y_test, poly_predict)

print(f"MSE IS: {MSE_test: .4f}, and R2 is: {R2_test: .4f}")
'''
MSE IS:  4.0943, and R2 is:  0.6957
'''

