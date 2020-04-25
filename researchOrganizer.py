import pickle
from random import shuffle
import numpy
from keras.engine.saving import load_model
from keras.preprocessing import sequence
import lstm.lstm_simple
import lstm.lstm_dropout_1
import lstm.lstm_dropout_2
import lstm.lstm_with_cnn
import lstm.blstm_simple
import lstm.blstm_dropout_1
import lstm.blstm_dropout_2
import lstm.blstm_with_cnn


def divide_into_training_validation_test(corpus, training_percentage, validation_percentage):
    # creating a list of all possible user_id-s
    user_id_list = []
    for message_dict in corpus:
        user_id_list.append(int(message_dict["user_id"]))
    user_id_list = list(set(user_id_list))
    shuffle(user_id_list)

    # dividing users into users that go to training set, validation set and test set
    training_division_point = round(training_percentage * len(user_id_list))
    validation_division_point = training_division_point + round(validation_percentage * len(user_id_list))
    training_user_id = user_id_list[:training_division_point]
    validation_used_id = user_id_list[training_division_point:validation_division_point]

    # dividing messages into training, validation and test (based on user_id)
    training = []
    validation = []
    test = []
    for message_dict in corpus:
        user_id = int(message_dict["user_id"])
        if user_id in training_user_id:
            training.append(message_dict)
        elif user_id in validation_used_id:
            validation.append(message_dict)
        else:
            test.append(message_dict)
    return training, validation, test


def create_x_and_y(corpus, task):
    X = []
    y = []
    for message in corpus:
        X.append(message["word_embedding"])
        y.append(message[task])
    return X, numpy.array(y)

def resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, X_test, y_test, save_model_name, actual_epochs):
    # load the saved model
    saved_model = load_model(save_model_name)
    _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
    _, valid_acc = saved_model.evaluate(X_validation, y_validation, verbose=0)
    _, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    result_str = '\t\tAttempt ' + str(save_model_name[-1]) + ", Epochs " + str(actual_epochs) + "\t\t\t"
    result_str += 'Train: %.3f%%, Validation: %.3f%%, Test: %.3f%%' % (train_acc*100, valid_acc*100, test_acc*100)
    result_str += '\n'
    return result_str

# main variables notation
max_review_length = 200
task = "country_part"
training_set_percentage = 0.5
validation_set_percentage = 0.25
filePathMainInfo = "results_" + str(max_review_length) + "_words.txt"
numNeurons = [5, 10, 25, 50, 100]

# data preparation
# download dictionary of 200 word length messages
with open('data/dict/final_corpus_max_length_' + str(max_review_length) + '.pickle', 'rb') as handle:
    corpus = pickle.load(handle)

training, validation, test = divide_into_training_validation_test(corpus, training_set_percentage, validation_set_percentage)

try:
    del(corpus)
except Exception as e:
    print("Not deleted")
    pass

X_train, y_train = create_x_and_y(training, task)
X_validation, y_validation = create_x_and_y(validation, task)
X_test, y_test = create_x_and_y(test, task)
# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_validation = sequence.pad_sequences(X_validation, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

with open(filePathMainInfo, "w+") as f:
    f.write("Max " + str(max_review_length) + " word posts\n")

    f.write("\nSimple LSTM\n")
    for n in numNeurons:
        save_model_name = "models/ripartizione_geografica_lstm_simple_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(5):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.lstm_simple.runTraining(X_train, y_train, X_validation, y_validation, n, 25, save_model_name)
            f.write(resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, X_test, y_test, save_model_name, actual_epochs))
    f.write("\n")

    f.write("\nLSTM Dropout 1\n")
    for n in numNeurons:
        save_model_name = "models/ripartizione_geografica_lstm_dropout_1_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(5):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.lstm_dropout_1.runTraining(X_train, y_train, X_validation,
                                                            y_validation, n, 25, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, X_test, y_test, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nLSTM Dropout 2\n")
    for n in numNeurons:
        save_model_name = "models/ripartizione_geografica_lstm_dropout_2_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(5):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.lstm_dropout_2.runTraining(X_train, y_train, X_validation,
                                                            y_validation, n, 25, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, X_test, y_test, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nLSTM with CNN\n")
    for n in numNeurons:
        save_model_name = "models/ripartizione_geografica_lstm_with_cnn_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(5):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.lstm_with_cnn.runTraining(X_train, y_train, X_validation,
                                                           y_validation, n, 25, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, X_test, y_test, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nSimple BLSTM\n")
    for n in numNeurons:
        save_model_name = "models/ripartizione_geografica_blstm_simple_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(5):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.blstm_simple.runTraining(X_train, y_train, X_validation,
                                                                                 y_validation, n, 25, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, X_test, y_test, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nBLSTM Dropout 1\n")
    for n in numNeurons:
        save_model_name = "models/ripartizione_geografica_blstm_dropout_1_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(5):
            save_model_name= save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.blstm_dropout_1.runTraining(X_train, y_train, X_validation,
                                                                                    y_validation, n, 25,
                                                                                    save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, X_test, y_test, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nBLSTM Dropout 2\n")
    for n in numNeurons:
        save_model_name = "models/ripartizione_geografica_blstm_dropout_2_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(5):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.blstm_dropout_2.runTraining(X_train, y_train, X_validation,
                                                                                    y_validation, n, 25,
                                                                                    save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, X_test, y_test, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nBLSTM with CNN\n")
    for n in numNeurons:
        save_model_name = "models/ripartizione_geografica_blstm_with_cnn_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(5):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.blstm_with_cnn.runTraining(X_train, y_train, X_validation,
                                                                                   y_validation, n, 25, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, X_test, y_test, save_model_name,
                                        actual_epochs))
    f.write("\n")

