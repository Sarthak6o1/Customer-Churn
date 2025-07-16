# Evaluate GridSearchCV best model
best_grid_model = grid_result.best_estimator_
test_grid_accuracy = best_grid_model.score(X_test, y_test)
print("Test set accuracy (GridSearchCV): {:.3f}".format(test_grid_accuracy))

# Evaluate RandomizedSearchCV best model
best_random_model = random_result.best_estimator_
test_random_accuracy = best_random_model.score(X_test, y_test)
print("Test set accuracy (RandomizedSearchCV): {:.3f}".format(test_random_accuracy))
