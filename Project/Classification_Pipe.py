# Classification Modeling
# Base on KNN and logistic

# Set a new X and Y
X_class = classification_df.drop(columns=['sex'])
Y_class = classification_df['sex']

# Split the data set
X_train_c, X_res_c, Y_train_c, Y_res_c = train_test_split(X_class, Y_class,
                                                         test_size=0.3,random_state=42)

X_val_c, X_test_c, Y_val_c, Y_test_c = train_test_split(X_res_c, Y_res_c,
                                                         test_size=0.5,random_state=42)

# pipe
knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# grid param
param_grid_knn = {
    'knn__n_neighbors': np.arange(1,21,1),
    'knn__weights': ['uniform', 'distance'] # this is using distance to evaluate the weights
}

# grid search to find the best param
knn_grid = GridSearchCV(
    knn_pipe,
    param_grid_knn,
    cv = 5,
    scoring = 'accuracy',
    n_jobs = -1
)

# fit the model
knn_grid.fit(X_train_c, Y_train_c)

print("the best param",knn_grid.best_params_)

# evaluate
y_pred_knn = knn_grid.predict(X_val_c)

print("Accuarcy:\n", accuracy_score(Y_val_c, y_pred_knn))
print("Confusion_matrix:\n", confusion_matrix(Y_val_c, y_pred_knn))
print("Report:\n", classification_report(Y_val_c, y_pred_knn))
'''
the best param {'knn__n_neighbors': np.int64(16), 'knn__weights': 'distance'}
Accuarcy:
 0.9357142857142857
Confusion_matrix:
 [[58  5]
 [ 4 73]]
Report:
               precision    recall  f1-score   support

      female       0.94      0.92      0.93        63
        male       0.94      0.95      0.94        77

    accuracy                           0.94       140
   macro avg       0.94      0.93      0.93       140
weighted avg       0.94      0.94      0.94       140
'''

# Now, lets testing the logistic
# pipe
log_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('log_model', LogisticRegression(max_iter=2000))
])

# Setting params
param_grid_log = {
    'log_model__C': np.logspace(-3,2,num=6),
    'log_model__penalty':['l2'],
    'log_model__solver': ['lbfgs'] # it is useful on small data set and binary problem
}

# grid search the best param
log_grid = GridSearchCV(
    log_pipe,
    param_grid_log,
    cv = 5,
    scoring = 'accuracy',
    n_jobs = 1
)

log_grid.fit(X_train_c, Y_train_c)

print("the best param",log_grid.best_params_)

# evaluate
y_pred_log = log_grid.predict(X_val_c)

print("Accuarcy:\n", accuracy_score(Y_val_c, y_pred_log))
print("Confusion_matrix:\n", confusion_matrix(Y_val_c, y_pred_log))
print("Report:\n", classification_report(Y_val_c, y_pred_log))
'''
the best param {'log_model__C': np.float64(0.01), 'log_model__penalty': 'l2', 'log_model__solver': 'lbfgs'}
Accuarcy:
 0.9
Confusion_matrix:
 [[56  7]
 [ 7 70]]
Report:
               precision    recall  f1-score   support

      female       0.89      0.89      0.89        63
        male       0.91      0.91      0.91        77

    accuracy                           0.90       140
   macro avg       0.90      0.90      0.90       140
weighted avg       0.90      0.90      0.90       140
'''
# So, KNN is the best for testing.
knn_test_c = knn_grid.predict(X_test_c)

print("Accuarcy:\n", accuracy_score(Y_test_c, knn_test_c))
print("Confusion_matrix:\n", confusion_matrix(Y_test_c, knn_test_c))
print("Report:\n", classification_report(Y_test_c, knn_test_c))
'''
Accuarcy:
 0.8723404255319149
Confusion_matrix:
 [[62 10]
 [ 8 61]]
Report:
               precision    recall  f1-score   support

      female       0.89      0.86      0.87        72
        male       0.86      0.88      0.87        69

    accuracy                           0.87       141
   macro avg       0.87      0.87      0.87       141
weighted avg       0.87      0.87      0.87       141
'''

