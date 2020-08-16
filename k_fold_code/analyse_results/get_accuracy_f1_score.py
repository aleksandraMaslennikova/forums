import os
import pickle

import numpy
from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences
from sklearn import metrics
from multiprocessing import Process, Array


def transform_age_category(corpus, topic, reversed):
    for post in corpus:
        age = int(post["age"])
        forums_thematic = post["forums_thematic"]
        if reversed:
            if age < 30 and forums_thematic != topic:
                post["age"] = "age < 30"
            elif 49 < age < 70 and forums_thematic != topic:
                post["age"] = "49 < age < 70"
            else:
                post["age"] = "to_del"
        else:
            if age < 30 and forums_thematic == topic:
                post["age"] = "age < 30"
            elif 49 < age < 70 and forums_thematic == topic:
                post["age"] = "49 < age < 70"
            else:
                post["age"] = "to_del"
    return corpus


def transform_country_part_category(corpus, topic, reversed):
    for post in corpus:
        country_part = numpy.array(post["country_part"])
        forums_thematic = post["forums_thematic"]
        if (reversed and forums_thematic == topic) or (not reversed and forums_thematic != topic):
                post[task] = "to_del"
        else:
            if (country_part == numpy.array([1, 0, 0, 0, 0])).all():
                post["country_part"] = "Nord-ovest"
            elif (country_part == numpy.array([0, 1, 0, 0, 0])).all():
                post["country_part"] = "Nord-est"
            elif (country_part == numpy.array([0, 0, 1, 0, 0])).all():
                post["country_part"] = "Centro"
            elif (country_part == numpy.array([0, 0, 0, 1, 0])).all():
                post["country_part"] = "Sud"
            elif (country_part == numpy.array([0, 0, 0, 0, 1])).all():
                post["country_part"] = "Isole"
    return corpus


def create_x_y(corpus, task):
    x = []
    y = []
    for message in corpus:
        x.append(message["text_sequence"])
        y.append(message[task])
    return numpy.array(x), numpy.array(y)

def output_in_file(filePathMainInfo, writeType, text):
    with open(filePathMainInfo, writeType) as f:
        f.write(text)
        f.flush()
        f.close()


def load_data(topic, max_len_post):
    # load data for topic
    with open('../../data/final_corpus_dictionary_max_length_' + str(max_len_post) + '.pickle', 'rb') as handle:
        corpus_topic = pickle.load(handle)

    if task == "age":
        corpus_topic = transform_age_category(corpus_topic, topic, False)
    if task == "country_part":
        corpus_topic = transform_country_part_category(corpus_topic, topic, False)

    posts_id_to_del = []
    for i in range(len(corpus_topic)):
        if corpus_topic[i][task] == "to_del":
            posts_id_to_del.append(i)
        elif len(corpus_topic[i]["text"]) == 0:
            posts_id_to_del.append(i)
    for index in sorted(posts_id_to_del, reverse=True):
        del corpus_topic[index]

    test_user_id = []
    if task == "age":
        # test_user_id_watches_two_groups
        test_user_id = [753, 670, 509, 532, 496, 758, 840, 48, 851, 813, 841, 711, 584, 788, 856, 795,
                        833, 589, 292, 712, 803, 737, 506, 627, 363, 702, 640, 834, 539, 665, 621, 731,
                        796, 839, 695, 638, 787, 79, 398, 514, 764, 538]
    if task == "country_part":
        test_user_id = [523, 526, 530, 533, 539, 546, 549, 550, 554, 556, 558, 559, 569, 570, 571, 572,
                        576, 579, 588, 589, 79, 596, 603, 608, 609, 612, 618, 621, 622, 625, 626, 627,
                        629, 634, 640, 644, 646, 647, 648, 652, 653, 655, 656, 658, 670, 162, 675, 683,
                        690, 693, 694, 707, 708, 712, 716, 718, 726, 730, 734, 736, 739, 746, 244, 765,
                        768, 769, 771, 772, 783, 785, 786, 788, 796, 803, 808, 811, 816, 817, 823, 826,
                        831, 836, 837, 840, 844, 845, 847, 856, 857, 860, 861, 363, 371, 392, 496, 497,
                        498, 506, 509]

    in_domain = []
    for message_dict in corpus_topic:
        user_id = int(message_dict["user_id"])
        if user_id in test_user_id:
            in_domain.append(message_dict)

    # load data for the rest of the corpus
    with open('../../data/final_corpus_dictionary_max_length_' + str(max_len_post) + '.pickle', 'rb') as handle:
        out_domain = pickle.load(handle)

    if task == "age":
        out_domain = transform_age_category(out_domain, topic, True)
    if task == "country_part":
        out_domain = transform_country_part_category(out_domain, topic, True)

    posts_id_to_del = []
    for i in range(len(out_domain)):
        if out_domain[i][task] == "to_del":
            posts_id_to_del.append(i)
        elif len(out_domain[i]["text"]) == 0:
            posts_id_to_del.append(i)
    for index in sorted(posts_id_to_del, reverse=True):
        del out_domain[index]
    return in_domain, out_domain


def most_common(lst):
    return max(set(lst), key=lst.count)


def analyzing_predictions(predictions, right_y, filePath, task, description):
    results = []
    if task == "age":
        for i in range(len(predictions[0])):
            results_x = []
            for j in range(len(predictions)):
                results_x.append(predictions[j][i])
            for j in range(len(results_x)):
                results_x[j] = round(results_x[j][0])
            y_predicted = most_common(results_x)
            if y_predicted == 0:
                y_predicted = "age < 30"
            else:
                y_predicted = "49 < age < 70"
            results.append(y_predicted)
    if task == "country_part":
        for i in range(len(predictions[0])):
            results_x = []
            for j in range(len(predictions)):
                results_x.append(list(predictions[j][i]).index(max(predictions[j][i])))
            y_predicted = most_common(results_x)
            if y_predicted == 0:
                y_predicted = "Nord-ovest"
            elif y_predicted == 1:
                y_predicted = "Nord-est"
            elif y_predicted == 2:
                y_predicted = "Centro"
            elif y_predicted == 3:
                y_predicted = "Sud"
            elif y_predicted == 4:
                y_predicted = "Isole"
            results.append(y_predicted)

    output_in_file(filePath, "a", "\n\t\t" + description + "\n")
    output_in_file(filePath, "a", str(metrics.classification_report(right_y, results, digits=3)))


def predict_and_analyze(filePath, task, path, model, X_in_domain, y_in_domain, X_out_domain, y_out_domain):
    saved_model_list = []
    for i in range(1, 6):
        path_model = path + model + "/attempt_" + str(i)
        saved_model = load_model(path_model)
        if task == "age":
            saved_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        if task == "country_part":
            saved_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        saved_model_list.append(saved_model)

    predictions = []
    for saved_model in saved_model_list:
        predictions.append(saved_model.predict(X_in_domain))
    analyzing_predictions(predictions, y_in_domain, filePath, task, "in-domain")

    predictions = []
    for saved_model in saved_model_list:
        predictions.append(saved_model.predict(X_out_domain))
    analyzing_predictions(predictions, y_out_domain, filePath, task, "out-domain")


if __name__ == '__main__':
    task = "country_part"
    sub_category = ""
    max_length = 100
    num_neurons_list = [25, 50, 100]
    topic = "Watches"
    filePath = "../../results/f-score_analysis_" + task + "_k-fold_itwac_max_length_" + str(max_length) + ".txt"

    in_domain, out_domain = load_data(topic, max_length)
    X_in_domain, y_in_domain = create_x_y(in_domain, task)
    X_in_domain = pad_sequences(X_in_domain, maxlen=max_length, padding='post')
    X_out_domain, y_out_domain = create_x_y(out_domain, task)
    X_out_domain = pad_sequences(X_out_domain, maxlen=max_length, padding='post')

    #with open(filePath, "w") as f:
    baseline_in_domain = [most_common(list(y_in_domain))] * len(y_in_domain)
    output_in_file(filePath, "w", "\nIn-domain Baseline\n")
    output_in_file(filePath, "a", str(metrics.classification_report(y_in_domain, baseline_in_domain, digits=3)))
    output_in_file(filePath, "a", "\n")
    baseline_out_domain = [most_common(list(y_out_domain))] * len(y_out_domain)
    output_in_file(filePath, "a", "\nOut-domain Baseline\n")
    output_in_file(filePath, "a", str(metrics.classification_report(y_out_domain, baseline_out_domain, digits=3)))
    output_in_file(filePath, "a", "\n")

    for num_neurons in num_neurons_list:
        print("Number of neurons " + str(num_neurons))
        output_in_file(filePath, "a", "\nNumber of neurons " + str(num_neurons) + "\n")

        if task == "age":
            path = "../../models/" + task + "/K-fold/" + str(max_length) + "/" + sub_category + "/" + "neurons_" + str(
                num_neurons) + "/"
        else:
            path = "../../models/" + task + "/K-fold/" + str(max_length) + "/" + "neurons_" + str(
                num_neurons) + "/"

        list_models = os.listdir(path)
        for model in list_models:
            print(model)
            output_in_file(filePath, "a", "\n\t" + model + "\n")
            p = Process(target=predict_and_analyze,
                        args=(filePath, task, path, model, X_in_domain, y_in_domain, X_out_domain, y_out_domain,))
            p.start()
            p.join()
