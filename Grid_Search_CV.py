keras_model = KerasClassifier(build_fn=build_model, verbose=0)

param_grid = {
    'batch_size': [32, 64],
    'epochs': [50, 100],
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'neurons': [8, 16],
    'dropout_rate': [0.0, 0.2],
    'layers': [1, 2],
    'learning_rate': [0.001, 0.0005]
}

grid_search = GridSearchCV(
    estimator=keras_model,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring='accuracy'
)
grid_result = grid_search.fit(X_train, y_train)

print("GridSearchCV Best Accuracy: {:.4f}".format(grid_result.best_score_))
print("GridSearchCV Best Params:", grid_result.best_params_)
