from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Bidirectional, Embedding
from keras.layers import LSTM
from matplotlib import pyplot

def runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix, num_neurons_lstm, num_categories, batch_size, early_stopping_patience, save_model_name):
    # create the model
    model = Sequential()
    e = Embedding(len(embedding_matrix), 129, weights=[embedding_matrix], trainable=False)
    model.add(e)
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(num_neurons_lstm)))
    if num_categories == 2:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense(num_categories, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=early_stopping_patience)
    mc = ModelCheckpoint(save_model_name, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=1500, batch_size=batch_size,
                        callbacks=[es, mc])
    # plot training history
    pyplot.title(save_model_name)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()
    return es.stopped_epoch - early_stopping_patience