param_dist = {
    'batch_size': [32, 64, 128],
    'epochs': [50, 100, 150],
    'optimizer': ['adam', 'rmsprop'],
    'learning_rate': uniform(0.0005, 0.002),
    'activation': ['relu', 'tanh'],
    'neurons': [8, 16, 32],
    'dropout_rate': uniform(0.0, 0.4),
    'layers': [1, 2, 3]
}

random_search = RandomizedSearchCV(
    estimator=keras_model,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    random_state=42,
    n_jobs=-1
)
random_result = random_search.fit(X_train, y_train)

print("RandomizedSearchCV Best Accuracy: {:.4f}".format(random_result.best_score_))
print("RandomizedSearchCV Best Params:", random_result.best_params_)
