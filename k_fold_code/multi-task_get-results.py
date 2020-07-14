import pickle
import shutil
from pathlib import Path
import numpy
from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences
from multiprocessing import Process, Array

import multitaskLearning.lstm_multitask
import multitaskLearning.lstm_dropout_1_multitask
import multitaskLearning.lstm_dropout_2_multitask
import multitaskLearning.lstm_with_cnn_multitask
import multitaskLearning.blstm_multitask
import multitaskLearning.blstm_dropout_1_multitask
import multitaskLearning.blstm_dropout_2_multitask
import multitaskLearning.blstm_with_cnn_multitask


def transform_age_category(corpus, topic, reversed):
    for post in corpus:
        age = int(post["age"])
        forums_thematic = post["forums_thematic"]
        if reversed:
            if age < 30 and forums_thematic != topic:
                post["age"] = numpy.int64(0)
            elif 49 < age < 70 and forums_thematic != topic:
                post["age"] = numpy.int64(1)
            else:
                post["age"] = "to_del"
        else:
            if age < 30 and forums_thematic == topic:
                post["age"] = numpy.int64(0)
            elif 49 < age < 70 and forums_thematic == topic:
                post["age"] = numpy.int64(1)
            else:
                post["age"] = "to_del"
    return corpus


def create_x_y(corpus, task):
    x = []
    y = []
    for message in corpus:
        x.append(message["text_sequence"])
        y.append(message[task])
    return numpy.array(x), numpy.array(y)


def f(name):
    print('hello', name)


def output_in_file(filePathMainInfo, writeType, text):
    with open(filePathMainInfo, writeType) as f:
        f.write(text)
        f.flush()
        f.close()


def run_train_one_k_fold(nn_type, task, filePathMainInfo, max_len_post, num_neurons, attempt_i, validation_k,
                         X_train, y_train, X_validation, y_validation,
                         embedding_matrix, batch_size, early_stopping_wait, early_stopping,
                         validation_results, actual_epochs_results):
    path_nn = None
    if nn_type == "Simple LSTM":
        path_nn = multitaskLearning.lstm_multitask
    elif nn_type == "LSTM Dropout 1":
        path_nn = multitaskLearning.lstm_dropout_1_multitask
    elif nn_type == "LSTM Dropout 2":
        path_nn = multitaskLearning.lstm_dropout_2_multitask
    elif nn_type == "LSTM with CNN":
        path_nn = multitaskLearning.lstm_with_cnn_multitask
    elif nn_type == "Simple BLSTM":
        path_nn = multitaskLearning.blstm_multitask
    elif nn_type == "BLSTM Dropout 1":
        path_nn = multitaskLearning.blstm_dropout_1_multitask
    elif nn_type == "BLSTM Dropout 2":
        path_nn = multitaskLearning.blstm_dropout_2_multitask
    elif nn_type == "BLSTM with CNN":
        path_nn = multitaskLearning.blstm_with_cnn_multitask
    path = "../models/" + str(task) + "/K-fold/" + str(max_len_post) + "/neurons_" + str(num_neurons) + "/" + str(nn_type) +"/Attempt " + str(attempt_i + 1) + "/"
    Path(path).mkdir(parents=True, exist_ok=True)
    save_model_name = path + "attempt_" + str(attempt_i + 1) + "_k-fold_" + str(validation_k + 1)

    actual_epochs = path_nn.runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix,
                num_neurons, batch_size, early_stopping_wait, early_stopping, save_model_name)

    saved_model = load_model(save_model_name)
    res_train = saved_model.evaluate(X_train, y_train, verbose=1)
    res_valid = saved_model.evaluate(X_validation, y_validation, verbose=0)
    result_str = '\t\t\tK-Fold ' + str(validation_k + 1) + ", Epochs " + str(actual_epochs) + "\t\t\t"
    result_str += '\n\t\t\t\tAge\t\t\t\tTrain: %.3f%%, Validation: %.3f%%' % (res_train[3] * 100, res_valid[3] * 100)
    result_str += '\n\t\t\t\tCountry part\tTrain: %.3f%%, Validation: %.3f%%' % (res_train[4] * 100, res_valid[4] * 100)
    result_str += '\n'
    output_in_file(filePathMainInfo, "a", result_str)

    i = 0
    while validation_results[i] != -1:
        i +=1
    validation_results[i] = round(res_valid[3], 5)
    validation_results[i+1] = round(res_valid[4], 5)
    i = 0
    while actual_epochs_results[i] != -1:
        i += 1
    actual_epochs_results[i] = actual_epochs


def run_train_final(nn_type, filePathMainInfo, task, max_len_post, num_neurons,
                    X_train, y_train, embedding_matrix, batch_size, attempt_i,
                    result_acc, num_epochs):
    path_nn = None
    if nn_type == "Simple LSTM":
        path_nn = multitaskLearning.lstm_multitask
    elif nn_type == "LSTM Dropout 1":
        path_nn = multitaskLearning.lstm_dropout_1_multitask
    elif nn_type == "LSTM Dropout 2":
        path_nn = multitaskLearning.lstm_dropout_2_multitask
    elif nn_type == "LSTM with CNN":
        path_nn = multitaskLearning.lstm_with_cnn_multitask
    elif nn_type == "Simple BLSTM":
        path_nn = multitaskLearning.blstm_multitask
    elif nn_type == "BLSTM Dropout 1":
        path_nn = multitaskLearning.blstm_dropout_1_multitask
    elif nn_type == "BLSTM Dropout 2":
        path_nn = multitaskLearning.blstm_dropout_2_multitask
    elif nn_type == "BLSTM with CNN":
        path_nn = multitaskLearning.blstm_with_cnn_multitask
    path = "../models/" + str(task) + "/K-fold/" + str(max_len_post) + "/neurons_" + str(max_len_post) + "/" + str(nn_type) + "/"
    Path(path).mkdir(parents=True, exist_ok=True)
    save_model_name = path + "attempt_" + str(attempt_i + 1)
    actual_epochs = path_nn.runTraining_k_fold(X_train, y_train, embedding_matrix, num_neurons, batch_size,
                       num_epochs, save_model_name)
    saved_model = load_model(save_model_name)
    res_train = saved_model.evaluate(X_train, y_train, verbose=0)
    result_str = '\t\tRESULTS ' + str(5) + ", Epochs " + str(num_epochs) + "\t\t\t"
    result_str += '\n\t\t\tAge\t\t\t\tTrain: %.3f%%, Validation: %.3f%%' % (res_train[3] * 100, result_acc[0] * 100)
    result_str += '\n\t\t\tCountry part\tTrain: %.3f%%, Validation: %.3f%%' % (res_train[4] * 100, result_acc[1] * 100)
    result_str += '\n'
    output_in_file(filePathMainInfo, "a", result_str + "\n")


def train(nn_type, num_neurons):
    output_in_file(filePathMainInfo, "a", "\n" + "Multi-task" + "\n")
    for i in range(repeat):
        output_in_file(filePathMainInfo, "a", "\tAttempt " + str(i+1) + "\n")

        validation_final_results_age = []
        validation_final_results_country_part = []
        actual_epochs_final_results = []
        for validation_k in range(k):
            training = []
            validation = []
            for j in range(k):
                if j == validation_k:
                    validation += posts_k_fold[j]
                else:
                    training += posts_k_fold[j]
            X_train, y1_train = create_x_y(training, "age")
            X_train, y2_train = create_x_y(training, "country_part")
            X_train = pad_sequences(X_train, maxlen=max_len_post, padding='post')
            X_validation, y1_validation = create_x_y(validation, "age")
            X_validation, y2_validation = create_x_y(validation, "country_part")
            X_validation = pad_sequences(X_validation, maxlen=max_len_post, padding='post')
            validation_results = Array('d', [-1] * 4)
            actual_epochs_results = Array('i', [-1] * 2)
            for early_stopping in ["age", "country_part"]:
                output_in_file(filePathMainInfo, "a", "\t\tEarly-stopping " + early_stopping + "\n")
                p = Process(target=run_train_one_k_fold, args=(nn_type, task, filePathMainInfo, max_len_post, num_neurons, i, validation_k,
                         X_train, [y1_train, y2_train], X_validation, [y1_validation, y2_validation],
                         embedding_matrix, batch_size, early_stopping_wait, early_stopping,
                         validation_results, actual_epochs_results,))
                p.start()
                p.join()
            for_age = 0
            for_country_part = 0
            if validation_results[:][0] > validation_results[:][2]:
                for_age += 1
            elif validation_results[:][0] < validation_results[:][2]:
                for_country_part += 1
            if validation_results[:][1] > validation_results[:][3]:
                for_age += 1
            elif validation_results[:][1] > validation_results[:][3]:
                for_country_part += 1
            if for_age > for_country_part:
                output_in_file(filePathMainInfo, "a", "\t\tEarly-stopping is Age\n\n")
                validation_final_results_age.append(validation_results[:][0])
                validation_final_results_country_part.append(validation_results[:][1])
                actual_epochs_final_results.append(actual_epochs_results[:][0])
            else:
                output_in_file(filePathMainInfo, "a", "\t\tEarly-stopping is Country-part\n\n")
                validation_final_results_age.append(validation_results[:][2])
                validation_final_results_country_part.append(validation_results[:][3])
                actual_epochs_final_results.append(actual_epochs_results[:][1])

        path = "../models/" + str(task) + "/K-fold/" + str(max_len_post) + "/neurons_" + str(num_neurons) + "/" + str(nn_type) + "/Attempt " + str(i + 1) + "/"
        shutil.rmtree(path)
        result_acc_age = numpy.mean(validation_final_results_age)
        result_acc_country_part = numpy.mean(validation_final_results_country_part)
        better_results_acc = []
        for j in range(len(validation_final_results_age)):
            better_results_acc.append(validation_final_results_age[j] + validation_final_results_country_part[j])
        num_epochs = actual_epochs_final_results[better_results_acc.index(max(better_results_acc))] + 1
        training = []
        for j in range(k):
            training += posts_k_fold[j]
        X_train, y1_train = create_x_y(training, "age")
        X_train, y2_train = create_x_y(training, "country_part")
        X_train = pad_sequences(X_train, maxlen=max_len_post, padding='post')
        p = Process(target=run_train_final, args=(nn_type, filePathMainInfo, task, max_len_post, num_neurons,
                    X_train, [y1_train, y2_train], embedding_matrix, batch_size, i,
                    [result_acc_age, result_acc_country_part], num_epochs,))
        p.start()
        p.join()



if __name__ == '__main__':
    max_len_post = 1000
    task = "multi-task"
    topic = "Watches"
    word_embedding_dictionary = "itwac"
    number_of_neurons = [25]
    k = 5
    batch_size = 250
    early_stopping_wait = 50
    repeat = 5

    if word_embedding_dictionary == "itwac":
        with open('../data/itwac_word_embedding_matrix.pickle', 'rb') as handle:
            embedding_matrix = pickle.load(handle)
    else:
        with open('../data/twitter_word_embedding_matrix.pickle', 'rb') as handle:
            embedding_matrix = pickle.load(handle)

    developing_user_id_watches_two_groups = [524, 754, 537, 854, 529, 520, 605, 848, 525, 719, 825, 809, 510, 606,
                                             687, 806, 452, 607, 493, 588, 823, 623, 762, 536, 612, 603, 517, 744, 563,
                                             581, 624, 371, 530, 650, 370, 498, 713, 721, 857, 618, 671, 831, 540, 822,
                                             590, 679, 582, 579, 760, 689, 658, 850, 633, 655, 697, 765, 495, 816, 804,
                                             637, 682, 635, 599, 630, 701, 805, 672, 598, 832, 162, 617, 545, 675, 827,
                                             768, 654, 800, 601, 778, 759, 586, 508, 653, 80]

    # divide user_ids in k-folds
    user_id_k_fold = [None] * k
    div_point = round(len(developing_user_id_watches_two_groups) / k)
    for i in range(k):
        user_id_k_fold[i] = developing_user_id_watches_two_groups[i * div_point:(i+1)*div_point]
        if i == k-1:
            user_id_k_fold[i] = developing_user_id_watches_two_groups[i * div_point:]

    # load data for topic
    with open('../data/final_corpus_dictionary_max_length_' + str(max_len_post) + '.pickle', 'rb') as handle:
        corpus_watches = pickle.load(handle)

    corpus_watches = transform_age_category(corpus_watches, topic, False)

    posts_id_to_del = []
    for i in range(len(corpus_watches)):
        if corpus_watches[i]["age"] == "to_del":
            posts_id_to_del.append(i)
        elif len(corpus_watches[i]["text"]) == 0:
            posts_id_to_del.append(i)
    for index in sorted(posts_id_to_del, reverse=True):
        del corpus_watches[index]

    # divide posts in k-folds
    posts_k_fold = [None] * k
    for message_dict in corpus_watches:
        user_id = int(message_dict["user_id"])
        for i in range(k):
            if user_id in user_id_k_fold[i]:
                if posts_k_fold[i] is None:
                    posts_k_fold[i] = [message_dict]
                else:
                    posts_k_fold[i].append(message_dict)

    for num_neurons in number_of_neurons:
        filePathMainInfo = "../results/results_" + task + "_k-fold_itwac_max_length_" + str(max_len_post) + "_num_neurons_" + str(num_neurons) + ".txt"
        train("Simple LSTM", num_neurons)
        train("LSTM Dropout 1", num_neurons)
        train("LSTM Dropout 2", num_neurons)
        train("LSTM with CNN", num_neurons)
        train("Simple BLSTM", num_neurons)
        train("BLSTM Dropout 1", num_neurons)
        train("BLSTM Dropout 2", num_neurons)
        train("BLSTM with CNN", num_neurons)
