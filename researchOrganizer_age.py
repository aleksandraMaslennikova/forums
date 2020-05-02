import pickle
import numpy
from keras.engine.saving import load_model
from keras.preprocessing import sequence
import lstm.classification.lstm_simple
import lstm.classification.lstm_dropout_1
import lstm.classification.lstm_dropout_2
import lstm.classification.lstm_with_cnn
import blstm.classification.blstm_simple
import blstm.classification.blstm_dropout_1
import blstm.classification.blstm_dropout_2
import blstm.classification.blstm_with_cnn


def transform_age_category(corpus):
    for post in corpus:
        age = int(post["age"])
        if age < 20:
            post["age"] = "<20"
        elif age < 30:
            post["age"] = "20-29"
        elif age < 40:
            post["age"] = "30-39"
        elif age < 50:
            post["age"] = "40-49"
        elif age < 60:
            post["age"] = "50-59"
        elif age < 70:
            post["age"] = "60-69"
        else:
            post["age"] = ">69"
    return corpus

def create_x_and_y(corpus, task):
    X = []
    y = []
    for message in corpus:
        X.append(message["word_embedding"])
        y.append(message[task])
    return X, numpy.array(y)


def resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name, actual_epochs):
    # load the saved model
    saved_model = load_model(save_model_name)
    _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
    _, valid_acc = saved_model.evaluate(X_validation, y_validation, verbose=0)
    result_str = '\t\tAttempt ' + str(save_model_name[-1]) + ", Epochs " + str(actual_epochs) + "\t\t\t"
    result_str += 'Train: %.3f%%, Validation: %.3f%%' % (train_acc*100, valid_acc*100)
    result_str += '\n'
    return result_str


def resultsOfTest(X_test, y_test, save_model_name):
    # load the saved model
    saved_model = load_model(save_model_name)
    _, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    result_str = '\t\tAttempt ' + str(save_model_name[-1]) + "\t\t\t"
    result_str += 'Test: %.3f%%' % (test_acc*100)
    result_str += '\n'
    return result_str

# main variables notation
max_review_length = 100
word_embedding_dict = "twitter"
task = "age"
#numNeurons = [5, 10, 25, 50, 100]
numNeurons = [100]
early_stopping_wait = 25
repeat = 1
num_categories = 7
filePathMainInfoTrain = "results/results_age_training_" + str(word_embedding_dict) + "_max_length_" + str(word_embedding_dict)
filePathMainInfoTest = "results/results_age_test_" + str(word_embedding_dict) + "_max_length_" + str(word_embedding_dict)

# data preparation
with open('data/dict/' + str(word_embedding_dict) + '/' + str(max_review_length) + '/final_corpus_training_' + str(word_embedding_dict) + '_max_length_' + str(max_review_length) + '.pickle', 'rb') as handle:
    training = pickle.load(handle)
training = training
X_train, y_train = create_x_and_y(training, task)
try:
    del(training)
except Exception as e:
    print("Not deleted")
    pass

with open('data/dict/' + str(word_embedding_dict) + '/' + str(max_review_length) + '/final_corpus_validation_' + str(word_embedding_dict) + '_max_length_' + str(max_review_length) + '.pickle', 'rb') as handle:
    validation = pickle.load(handle)
validation = transform_age_category(validation)
X_validation, y_validation = create_x_and_y(validation, task)
try:
    del(validation)
except Exception as e:
    print("Not deleted")
    pass

# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_validation = sequence.pad_sequences(X_validation, maxlen=max_review_length)

with open(filePathMainInfoTrain, "w") as f:
    f.write("\nSimple LSTM\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_simple_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.classification.lstm_simple.runTraining(X_train, y_train, X_validation, y_validation, n, num_categories, early_stopping_wait, save_model_name)
            f.write(resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name, actual_epochs))
    f.write("\n")

    f.write("\nLSTM Dropout 1\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_dropout_1_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.classification.lstm_dropout_1.runTraining(X_train, y_train, X_validation,
                                                                           y_validation, n, num_categories, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nLSTM Dropout 2\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_dropout_2_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.classification.lstm_dropout_2.runTraining(X_train, y_train, X_validation,
                                                                           y_validation, n, num_categories, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nLSTM with CNN\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_with_cnn_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.classification.lstm_with_cnn.runTraining(X_train, y_train, X_validation,
                                                                          y_validation, n, num_categories, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nSimple BLSTM\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_simple_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = blstm.classification.blstm_simple.runTraining(X_train, y_train, X_validation,
                                                                          y_validation, n, num_categories, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nBLSTM Dropout 1\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_dropout_1_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = blstm.classification.blstm_dropout_1.runTraining(X_train, y_train, X_validation,
                                                                             y_validation, n, num_categories, early_stopping_wait,
                                                                             save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nBLSTM Dropout 2\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_dropout_2_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = blstm.classification.blstm_dropout_2.runTraining(X_train, y_train, X_validation,
                                                                             y_validation, n, num_categories, early_stopping_wait,
                                                                             save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
    f.write("\n")

    f.write("\nBLSTM with CNN\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_with_cnn_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = blstm.classification.blstm_with_cnn.runTraining(X_train, y_train, X_validation,
                                                                            y_validation, n, num_categories, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
    f.write("\n")

try:
    del(X_train)
    del(y_train)
    del(X_validation)
    del(y_validation)
except Exception as e:
    print("Not deleted")
    pass

with open('data/dict/' + str(word_embedding_dict) + '/' + str(max_review_length) + '/final_corpus_test_' + str(word_embedding_dict) + '_max_length_' + str(max_review_length) + '.pickle', 'rb') as handle:
    test = pickle.load(handle)
test = transform_age_category(test)
X_test, y_test = create_x_and_y(test, task)
try:
    del(test)
except Exception as e:
    print("Not deleted")
    pass

X_test = sequence.pad_sequences(X_train, maxlen=max_review_length)

with open(filePathMainInfoTest, "w") as f:
    f.write("\nSimple LSTM\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_simple_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            f.write(resultsOfTest(X_test, y_test, save_model_name))
    f.write("\n")

    f.write("\nLSTM Dropout 1\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_dropout_1_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            f.write(resultsOfTest(X_test, y_test, save_model_name))
    f.write("\n")

    f.write("\nLSTM Dropout 2\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_dropout_2_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            f.write(resultsOfTest(X_test, y_test, save_model_name))
    f.write("\n")

    f.write("\nLSTM with CNN\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_with_cnn_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            f.write(resultsOfTest(X_test, y_test, save_model_name))
    f.write("\n")

    f.write("\nSimple BLSTM\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_simple_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            f.write(resultsOfTest(X_test, y_test, save_model_name))
    f.write("\n")

    f.write("\nBLSTM Dropout 1\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_dropout_1_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            f.write(resultsOfTest(X_test, y_test, save_model_name))
    f.write("\n")

    f.write("\nBLSTM Dropout 2\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_dropout_2_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            f.write(resultsOfTest(X_test, y_test, save_model_name))
    f.write("\n")

    f.write("\nBLSTM with CNN\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_with_cnn_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            f.write(resultsOfTest(X_test, y_test, save_model_name))
    f.write("\n")