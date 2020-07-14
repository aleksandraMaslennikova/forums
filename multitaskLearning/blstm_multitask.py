from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from matplotlib import pyplot


def runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix,
                num_neurons, batch_size, early_stopping_patience, early_stopping, save_model_name):
    model_input = Input(shape=X_train[0].shape)
    # divided layers
    main_branch = Embedding(len(embedding_matrix), 129, weights=[embedding_matrix], trainable=False)(model_input)
    main_branch = Dense(64, activation='relu')(main_branch)
    main_branch = Dropout(0.2)(main_branch)
    main_branch = Dense(64, activation='relu')(main_branch)
    main_branch = Dropout(0.2)(main_branch)

    # separated layers
    y1 = Bidirectional(LSTM(num_neurons))(main_branch)
    y1 = Dense(1, activation='softmax', name='age')(y1)

    y2 = Bidirectional(LSTM(num_neurons))(main_branch)
    y2 = Dense(5, activation='softmax', name='country_part')(y2)

    model = Model(inputs=model_input, outputs=[y1, y2])
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])

    if early_stopping == "age":
        es = EarlyStopping(monitor='val_age_accuracy', mode='max', verbose=1, patience=early_stopping_patience)
        mc = ModelCheckpoint(save_model_name, monitor='val_age_accuracy', mode='max', save_best_only=True,
                             verbose=1)
    else:
        es = EarlyStopping(monitor='val_country_part_accuracy', mode='max', verbose=1, patience=early_stopping_patience)
        mc = ModelCheckpoint(save_model_name, monitor='val_country_part_accuracy', mode='max', save_best_only=True,
                             verbose=1)

    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=1500,
                        batch_size=batch_size, callbacks=[es, mc])

    # plot training history
    pyplot.title(save_model_name)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.legend()
    pyplot.savefig(save_model_name + ".png")
    pyplot.show()
    return es.stopped_epoch - early_stopping_patience


def runTraining_k_fold(X_train, y_train, embedding_matrix, num_neurons, batch_size,
                       num_epochs, save_model_name):
    model_input = Input(shape=X_train[0].shape)
    # divided layers
    main_branch = Embedding(len(embedding_matrix), 129, weights=[embedding_matrix], trainable=False)(model_input)
    main_branch = Dense(64, activation='relu')(main_branch)
    main_branch = Dropout(0.2)(main_branch)
    main_branch = Dense(64, activation='relu')(main_branch)
    main_branch = Dropout(0.2)(main_branch)

    # separated layers
    y1 = Bidirectional(LSTM(num_neurons))(main_branch)
    y1 = Dense(1, activation='softmax', name='age')(y1)

    y2 = Bidirectional(LSTM(num_neurons))(main_branch)
    y2 = Dense(5, activation='softmax', name='country_part')(y2)

    model = Model(inputs=model_input, outputs=[y1, y2])
    model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_split=0, epochs=num_epochs, batch_size=batch_size)

    model.save(save_model_name)
    # plot training history
    pyplot.title(save_model_name)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.legend()
    pyplot.savefig(save_model_name + ".png")
    pyplot.show()
