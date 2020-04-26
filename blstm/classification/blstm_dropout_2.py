from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Bidirectional
from keras.layers import LSTM
from matplotlib import pyplot

def runTraining(X_train, y_train, X_validation, y_validation, num_neurons_lstm, early_stopping_patience, save_model_name):
    # create the model
    model = Sequential()
    model.add(Bidirectional(LSTM(num_neurons_lstm, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=early_stopping_patience)
    mc = ModelCheckpoint(save_model_name, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=150, batch_size=64,
                        callbacks=[es, mc])
    # plot training history
    pyplot.title(save_model_name)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.legend()
    pyplot.show()
    return es.stopped_epoch - early_stopping_patience