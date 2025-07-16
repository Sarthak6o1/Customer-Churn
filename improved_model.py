def build_model(optimizer='adam', activation='relu', neurons=16, dropout_rate=0.0, layers=1, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
  
    for _ in range(layers-1):
        model.add(Dense(neurons, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
            
    model.add(Dense(1, activation='sigmoid'))

  
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = optimizer
        
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model(optimizer='adam', activation='relu', neurons=16, dropout_rate=0.1, layers=2, learning_rate=0.001)
model.fit(X_train, y_train, batch_size=32, epochs=50)
