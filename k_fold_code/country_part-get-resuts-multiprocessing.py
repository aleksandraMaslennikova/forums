import pickle
import shutil
from pathlib import Path
import numpy
from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences
from multiprocessing import Process, freeze_support, Array

import lstm.classification.lstm_simple
import lstm.classification.lstm_dropout_1
import lstm.classification.lstm_dropout_2
import lstm.classification.lstm_with_cnn
import blstm.classification.blstm_simple
import blstm.classification.blstm_dropout_1
import blstm.classification.blstm_dropout_2
import blstm.classification.blstm_with_cnn


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


def run_train_one_k_fold(filePathMainInfo, task, max_len_post, num_neurons, nn_type, attempt_i, validation_k,
                         X_train, y_train, X_validation, y_validation,
                         embedding_matrix, num_categories, batch_size, early_stopping_wait,
                         validation_results, actual_epochs_results):
    path_nn = None
    if nn_type == "Simple LSTM":
        path_nn = lstm.classification.lstm_simple
    elif nn_type == "LSTM Dropout 1":
        path_nn = lstm.classification.lstm_dropout_1
    elif nn_type == "LSTM Dropout 2":
        path_nn = lstm.classification.lstm_dropout_2
    elif nn_type == "LSTM with CNN":
        path_nn = lstm.classification.lstm_with_cnn
    elif nn_type == "Simple BLSTM":
        path_nn = blstm.classification.blstm_simple
    elif nn_type == "BLSTM Dropout 1":
        path_nn = blstm.classification.blstm_dropout_1
    elif nn_type == "BLSTM Dropout 2":
        path_nn = blstm.classification.blstm_dropout_2
    elif nn_type == "BLSTM with CNN":
        path_nn = blstm.classification.blstm_with_cnn
    path = "../models/" + str(task) + "/K-fold/" + str(max_len_post) + "/neurons_" + str(num_neurons) + "/" + nn_type + "/Attempt " + str(attempt_i + 1) + "/"
    Path(path).mkdir(parents=True, exist_ok=True)
    save_model_name = path + "attempt_" + str(attempt_i + 1) + "_k-fold_" + str(validation_k + 1)

    actual_epochs = path_nn.runTraining(X_train, y_train, X_validation,
                                        y_validation, embedding_matrix, num_neurons,
                                        num_categories, batch_size,
                                        early_stopping_wait, save_model_name)

    saved_model = load_model(save_model_name)
    _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
    _, valid_acc = saved_model.evaluate(X_validation, y_validation, verbose=0)
    result_str = '\t\tK-Fold ' + str(validation_k + 1) + ", Epochs " + str(actual_epochs) + "\t\t\t"
    result_str += 'Train: %.3f%%, Validation: %.3f%%' % (train_acc * 100, valid_acc * 100)
    result_str += '\n'
    output_in_file(filePathMainInfo, "a", result_str)

    validation_results[validation_k] = round(valid_acc, 5)
    actual_epochs_results[validation_k] = actual_epochs


def run_train_final(filePathMainInfo, task, max_len_post, num_neurons, nn_type,
                    X_train, y_train, embedding_matrix, num_categories, batch_size,
                    result_acc, num_epochs):
    path_nn = None
    if nn_type == "Simple LSTM":
        path_nn = lstm.classification.lstm_simple
    elif nn_type == "LSTM Dropout 1":
        path_nn = lstm.classification.lstm_dropout_1
    elif nn_type == "LSTM Dropout 2":
        path_nn = lstm.classification.lstm_dropout_2
    elif nn_type == "LSTM with CNN":
        path_nn = lstm.classification.lstm_with_cnn
    elif nn_type == "Simple BLSTM":
        path_nn = blstm.classification.blstm_simple
    elif nn_type == "BLSTM Dropout 1":
        path_nn = blstm.classification.blstm_dropout_1
    elif nn_type == "BLSTM Dropout 2":
        path_nn = blstm.classification.blstm_dropout_2
    elif nn_type == "BLSTM with CNN":
        path_nn = blstm.classification.blstm_with_cnn
    path = "../models/" + str(task) + "/K-fold/" + str(max_len_post) + "/neurons_" + str(
        num_neurons) + "/" + nn_type + "/"
    Path(path).mkdir(parents=True, exist_ok=True)
    save_model_name = path + "attempt_" + str(i + 1)
    actual_epochs = path_nn.runTraining_k_fold(X_train, y_train, embedding_matrix,
                                               num_neurons, num_categories, batch_size,
                                               num_epochs, save_model_name)
    saved_model = load_model(save_model_name)
    _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
    result_str = '\t\tRESULTS ' + str(5) + ", Epochs " + str(num_epochs) + "\t\t\t"
    result_str += 'Train: %.3f%%, Validation: %.3f%%' % (train_acc * 100, result_acc * 100)
    result_str += '\n'
    output_in_file(filePathMainInfo, "a", result_str + "\n")


def train(nn_type):
    output_in_file(filePathMainInfo, "a", "\n" + nn_type + "\n")
    for i in range(repeat):
        output_in_file(filePathMainInfo, "a", "\tAttempt " + str(i+1) + "\n")

        validation_results = Array('d', [0] * k)
        actual_epochs_results = Array('i', [0] * k)
        for validation_k in range(k):
            training = []
            validation = []
            for j in range(k):
                if j == validation_k:
                    validation += posts_k_fold[j]
                else:
                    training += posts_k_fold[j]
            X_train, y_train = create_x_y(training, task)
            X_train = pad_sequences(X_train, maxlen=max_len_post, padding='post')
            X_validation, y_validation = create_x_y(validation, task)
            X_validation = pad_sequences(X_validation, maxlen=max_len_post, padding='post')

            p = Process(target=run_train_one_k_fold, args=(filePathMainInfo, task, max_len_post, num_neurons, nn_type, i, validation_k, X_train, y_train, X_validation, y_validation, embedding_matrix, num_categories, batch_size, early_stopping_wait, validation_results, actual_epochs_results,))
            p.start()
            p.join()

        path = "../models/" + str(task) + "/K-fold/" + str(max_len_post) + "/neurons_" + str(
            num_neurons) + "/" + nn_type + "/Attempt " + str(i + 1)
        shutil.rmtree(path)
        result_acc = numpy.mean(validation_results[:])
        num_epochs = numpy.max(actual_epochs_results[:])
        training = []
        for j in range(k):
            training += posts_k_fold[j]
        X_train, y_train = create_x_y(training, task)
        X_train = pad_sequences(X_train, maxlen=max_len_post, padding='post')

        p = Process(target=run_train_final, args=(filePathMainInfo, task, max_len_post, num_neurons, nn_type,
                    X_train, y_train, embedding_matrix, num_categories, batch_size,
                    result_acc, num_epochs,))
        p.start()
        p.join()



if __name__ == '__main__':
    max_len_post = 1000
    task = "country_part"
    topic = "Watches"
    word_embedding_dictionary = "itwac"
    number_of_neurons = [100]
    k = 5
    batch_size = 250
    early_stopping_wait = 50
    repeat = 5
    num_categories = 5

    if word_embedding_dictionary == "itwac":
        with open('../data/itwac_word_embedding_matrix.pickle', 'rb') as handle:
            embedding_matrix = pickle.load(handle)
    else:
        with open('../data/twitter_word_embedding_matrix.pickle', 'rb') as handle:
            embedding_matrix = pickle.load(handle)

    developing_user_id_watches_country_part = [667, 599, 598, 499, 669, 755, 704, 552, 617, 821, 671, 620, 584, 834, 636, 676,
                                         292, 594, 841, 754, 547, 521, 828, 777, 824, 745, 595, 650, 843, 781, 518, 527,
                                         735, 597, 722, 590, 780, 639, 798, 728, 500, 678, 717, 774, 532, 577, 514, 742,
                                         682, 537, 680, 538, 504, 832, 505, 548, 661, 853, 763, 531, 668, 606, 800, 806,
                                         610, 829, 601, 797, 688, 784, 724, 398, 370, 649, 789, 818, 600, 516, 616, 775,
                                         725, 567, 665, 700, 807, 580, 604, 793, 697, 384, 820, 563, 842, 686, 510, 810,
                                         638, 568, 586, 855, 759, 758, 776, 677, 790, 607, 791, 859, 557, 515, 713, 507,
                                         684, 613, 766, 672, 666, 628, 619, 801, 710, 673, 503, 566, 799, 764, 573, 703,
                                         762, 747, 695, 848, 565, 740, 681, 587, 802, 623, 819, 645, 689, 522, 809, 135,
                                         80, 367, 632, 727, 750, 702, 757, 825, 714, 605, 679, 642, 752, 581, 687, 767,
                                         553, 501, 706, 723, 696, 813, 732, 511, 657, 517, 849, 751, 715, 394, 575, 792,
                                         812, 212, 615, 493, 525, 778, 827, 574, 452, 737, 519, 733, 508, 699, 38, 545,
                                         630, 592, 815, 839, 560, 794, 243, 662, 578, 551, 721, 701, 475, 760, 814, 624,
                                         564, 705, 637, 698, 805, 838, 651, 659, 720, 555, 528, 711, 583, 534, 582, 852,
                                         753, 822, 787, 830, 352, 611, 529, 631, 719, 494, 562, 520, 561, 641, 663, 614,
                                         502, 770, 543, 773, 738, 512, 643, 835, 854, 761, 795, 851, 536, 691, 674, 524,
                                         782, 495, 541, 850, 48, 833, 748, 635, 804, 78, 602, 779, 729, 858, 443, 540, 743,
                                         400, 664, 535, 731, 756, 654, 593, 350, 744, 633, 741, 544, 685, 692, 660, 846,
                                         709, 396, 585, 542, 513, 749, 591]

    # divide user_ids in k-folds
    user_id_k_fold = [None] * k
    div_point = round(len(developing_user_id_watches_country_part) / k)
    for i in range(k):
        user_id_k_fold[i] = developing_user_id_watches_country_part[i * div_point:(i+1)*div_point]
        if i == k-1:
            user_id_k_fold[i] = developing_user_id_watches_country_part[i * div_point:]

    # load data for topic
    with open('../data/final_corpus_dictionary_max_length_' + str(max_len_post) + '.pickle', 'rb') as handle:
        corpus_watches = pickle.load(handle)

    for post in corpus_watches:
        forums_thematic = post["forums_thematic"]
        if forums_thematic != topic:
             post[task] = "to_del"

    posts_id_to_del = []
    for i in range(len(corpus_watches)):
        if corpus_watches[i][task] == "to_del":
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

        task_options_list = []
        for message_dict in corpus_watches:
            option = message_dict[task]
            if type(option) is list:
                option = option.index(1)
            task_options_list.append(option)
        task_options_list = list(set(task_options_list))

        dict = {}
        for option in task_options_list:
            dict[option] = 0
        for post in corpus_watches:
            option = post[task]
            if type(option) is list:
                option = option.index(1)
            dict[option] += 1

        text = topic + ":\n"
        for option in task_options_list:
            text += ("\t" + str(option) + ": " + str(dict[option]) + "; percent: " + str(round(dict[option] * 100.0 / len(corpus_watches))) + "%\n")
        output_in_file(filePathMainInfo, "w", text)
        train("Simple LSTM")
        train("LSTM Dropout 1")
        train("LSTM Dropout 2")
        train("LSTM with CNN")
        #train("Simple BLSTM")
        #train("BLSTM Dropout 1")
        #train("BLSTM Dropout 2")
        #train("BLSTM with CNN")