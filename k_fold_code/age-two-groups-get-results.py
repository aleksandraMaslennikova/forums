import os
import pickle
import shutil
from pathlib import Path
import numpy
from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences

import lstm.classification.lstm_simple
import lstm.classification.lstm_dropout_1
import lstm.classification.lstm_dropout_2
import lstm.classification.lstm_with_cnn
import blstm.classification.blstm_simple
import blstm.classification.blstm_dropout_1
import blstm.classification.blstm_dropout_2
import blstm.classification.blstm_with_cnn


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


def train(nn_type):
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
    with open(filePathMainInfo, "a") as f:
        f.write("\n" + nn_type + "\n")
        for i in range(repeat):
            f.write("\tAttempt " + str(i+1) + "\n")
            validation_results = [0] * k
            actual_epochs_results = [0] * k
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

                path = "models/" + str(task) + "/K-fold/" + str(max_len_post) + "/two-groups/neurons_" + str(num_neurons) + "/" + nn_type + "/Attempt " + str(i+1) + "/"
                Path(path).mkdir(parents=True, exist_ok=True)
                save_model_name = path + "attempt_" + str(i+1) + "_k-fold_" + str(validation_k+1)
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
                f.write(result_str)
                f.flush()

                validation_results[validation_k] = round(valid_acc, 5)
                actual_epochs_results[validation_k] = actual_epochs

            path = "models/" + str(task) + "/K-fold/" + str(max_len_post) + "/two-groups/neurons_" + str(
                num_neurons) + "/" + nn_type + "/Attempt " + str(i + 1)
            shutil.rmtree(path)
            result_acc = numpy.mean(validation_results)
            num_epochs = numpy.max(actual_epochs_results)
            training = []
            for j in range(k):
                training += posts_k_fold[j]
            X_train, y_train = create_x_y(training, task)
            X_train = pad_sequences(X_train, maxlen=max_len_post, padding='post')

            path = "models/" + str(task) + "/K-fold/" + str(max_len_post) + "/two-groups/neurons_" + str(
                num_neurons) + "/" + nn_type + "/"
            Path(path).mkdir(parents=True, exist_ok=True)
            save_model_name = path + "attempt_" + str(i + 1)
            actual_epochs = path_nn.runTraining_k_fold(X_train, y_train, embedding_matrix,
                                                                               num_neurons, num_categories, batch_size,
                                                                               num_epochs, save_model_name)
            saved_model = load_model(save_model_name)
            _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
            result_str = '\t\tRESULTS ' + str(validation_k + 1) + ", Epochs " + str(num_epochs) + "\t\t\t"
            result_str += 'Train: %.3f%%, Validation: %.3f%%' % (train_acc * 100, result_acc * 100)
            result_str += '\n'
            f.write(result_str)
            f.flush()
        f.write("\n")
        f.close()


max_len_post = 1000
task = "age"
topic = "Watches"
word_embedding_dictionary = "itwac"
number_of_neurons = [100]
k = 5
batch_size = 500
early_stopping_wait = 50
repeat = 1
num_categories = 2

if word_embedding_dictionary == "itwac":
    with open('../data/itwac_word_embedding_matrix.pickle', 'rb') as handle:
        embedding_matrix = pickle.load(handle)
else:
    with open('../data/twitter_word_embedding_matrix.pickle', 'rb') as handle:
        embedding_matrix = pickle.load(handle)

developing_user_id_watches_two_groups = [524, 754, 537, 854, 529, 520, 605, 848, 525, 719, 825, 809, 510, 606, 687, 806,
                                         452, 607, 493, 588, 823, 623, 762, 536, 612, 603, 517, 744, 563, 581, 624, 371,
                                         530, 650, 370, 498, 713, 721, 857, 618, 671, 831, 540, 822, 590, 679, 582, 579,
                                         760, 689, 658, 850, 633, 655, 697, 765, 495, 816, 804, 637, 682, 635, 599, 630,
                                         701, 805, 672, 598, 832, 162, 617, 545, 675, 827, 768, 654, 800, 601, 778, 759,
                                         586, 508, 653, 80]
test_user_id_watches_two_groups = [753, 670, 509, 532, 496, 758, 840, 48, 851, 813, 841, 711, 584, 788, 856, 795, 833,
                                   589, 292, 712, 803, 737, 506, 627, 363, 702, 640, 834, 539, 665, 621, 731, 796, 839,
                                   695, 638, 787, 79, 398, 514, 764, 538]


# divide user_ids in k-folds
user_id_k_fold = [None] * k
div_point = round(len(developing_user_id_watches_two_groups) / k)
for i in range(k):
    user_id_k_fold[i] = developing_user_id_watches_two_groups[i * div_point:(i+1)*div_point]
    if i == k-1:
        user_id_k_fold[i] = developing_user_id_watches_two_groups[i * div_point:]

# load data for topic
with open('data/final_corpus_dictionary_max_length_' + str(max_len_post) + '.pickle', 'rb') as handle:
    corpus_watches = pickle.load(handle)
corpus_watches = transform_age_category(corpus_watches, topic, False)
posts_id_to_del = []
for i in range(len(corpus_watches)):
    if corpus_watches[i]["age"] == "to_del":
        posts_id_to_del.append(i)
    elif len(corpus_watches[i]["text_sequence"]) == 0:
        posts_id_to_del.append(i)
for index in sorted(posts_id_to_del, reverse=True):
    del corpus_watches[index]

# load data for the rest of the corpus
with open('data/final_corpus_dictionary_max_length_' + str(max_len_post) + '.pickle', 'rb') as handle:
    corpus_rest = pickle.load(handle)
corpus_rest = transform_age_category(corpus_rest, topic, True)
posts_id_to_del = []
for i in range(len(corpus_rest)):
    if corpus_rest[i]["age"] == "to_del":
        posts_id_to_del.append(i)
    elif len(corpus_rest[i]["text_sequence"]) == 0:
        posts_id_to_del.append(i)
for index in sorted(posts_id_to_del, reverse=True):
    del corpus_rest[index]

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
    filePathMainInfo = "results/results_age_k-fold_itwac_max_length_" + str(max_len_post) + "_num_neurons_" + str(num_neurons) + ".txt"
    """
    with open(filePathMainInfo, "w") as f:
        num_0 = 0
        num_1 = 0
        for post in corpus_watches:
            if post["age"] == 0:
                num_0 += 1
            if post["age"] == 1:
                num_1 += 1
        f.write("Watches:\n")
        f.write("\t<30    : " + str(num_0) + "; percent: " + str(round(num_0 * 100.0 / len(corpus_watches))) + "%\n")
        f.write("\t>49,<70: " + str(num_1) + "; percent: " + str(round(num_1 * 100.0 / len(corpus_watches))) + "%\n")

        num_0 = 0
        num_1 = 0
        for post in corpus_rest:
            if post["age"] == 0:
                num_0 += 1
            if post["age"] == 1:
                num_1 += 1
        f.write("Rest of the corpus:\n")
        f.write("\t<30    : " + str(num_0) + "; percent: " + str(round(num_0 * 100.0 / len(corpus_rest))) + "%\n")
        f.write("\t>49,<70: " + str(num_1) + "; percent: " + str(round(num_1 * 100.0 / len(corpus_rest))) + "%\n")
    """
    #train("Simple LSTM")
    train("LSTM Dropout 1")
    #train("LSTM Dropout 2")
    #train("LSTM with CNN")
    #train("Simple BLSTM")
    #train("BLSTM Dropout 1")
    #train("BLSTM Dropout 2")
    #train("BLSTM with CNN")