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

    corpus_topic = transform_age_category(corpus_topic, topic, False)
    corpus_topic = transform_country_part_category(corpus_topic, topic, False)

    posts_id_to_del = []
    for i in range(len(corpus_topic)):
        if corpus_topic[i]["age"] == "to_del":
            posts_id_to_del.append(i)
        elif len(corpus_topic[i]["text"]) == 0:
            posts_id_to_del.append(i)
    for index in sorted(posts_id_to_del, reverse=True):
        del corpus_topic[index]

    test_user_id = [753, 670, 509, 532, 496, 758, 840, 48, 851, 813, 841, 711, 584, 788, 856, 795,
                        833, 589, 292, 712, 803, 737, 506, 627, 363, 702, 640, 834, 539, 665, 621, 731,
                        796, 839, 695, 638, 787, 79, 398, 514, 764, 538]

    in_domain = []
    for message_dict in corpus_topic:
        user_id = int(message_dict["user_id"])
        if user_id in test_user_id:
            in_domain.append(message_dict)

    # load data for the rest of the corpus
    with open('../../data/final_corpus_dictionary_max_length_' + str(max_len_post) + '.pickle', 'rb') as handle:
        out_domain = pickle.load(handle)

    out_domain = transform_age_category(out_domain, topic, True)
    out_domain = transform_country_part_category(out_domain, topic, True)

    posts_id_to_del = []
    for i in range(len(out_domain)):
        if out_domain[i]["age"] == "to_del":
            posts_id_to_del.append(i)
        elif len(out_domain[i]["text"]) == 0:
            posts_id_to_del.append(i)
    for index in sorted(posts_id_to_del, reverse=True):
        del out_domain[index]
    return in_domain, out_domain


def most_common(lst):
    return max(set(lst), key=lst.count)


def analyzing_predictions(predictions, right_y, filePath, task, description):
    results_age = []
    predictions_age = predictions[0]
    right_y_age = right_y[0]
    for i in range(len(predictions_age[0])):
        results_x = []
        for j in range(len(predictions_age)):
            results_x.append(predictions_age[j][i])
        for j in range(len(results_x)):
            results_x[j] = round(results_x[j][0])
        y_predicted = most_common(results_x)
        if y_predicted == 0:
            y_predicted = "age < 30"
        else:
            y_predicted = "49 < age < 70"
        results_age.append(y_predicted)
    output_in_file(filePath, "a", "\n\t\tAge - " + description + "\n")
    output_in_file(filePath, "a", str(metrics.classification_report(right_y_age, results_age, digits=3)))
    output_in_file(filePath, "a", "\n")

    results_country_part = []
    predictions_country_part = predictions[1]
    right_y_country_part = right_y[1]
    for i in range(len(predictions_country_part[0])):
        results_x = []
        for j in range(len(predictions_country_part)):
            results_x.append(list(predictions_country_part[j][i]).index(max(predictions_country_part[j][i])))
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
        results_country_part.append(y_predicted)

    output_in_file(filePath, "a", "\n\t\tCountry-part - " + description + "\n")
    output_in_file(filePath, "a", str(metrics.classification_report(right_y_country_part, results_country_part, digits=3)))


def predict_and_analyze(filePath, task, path, model, X_in_domain, y_in_domain, X_out_domain, y_out_domain):
    saved_model_list = []
    for i in range(1, 6):
        path_model = path + model + "/attempt_" + str(i)
        saved_model = load_model(path_model)
        saved_model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])
        saved_model_list.append(saved_model)

    predictions_age = []
    predictions_country_part = []
    for saved_model in saved_model_list:
        predictions = saved_model.predict(X_in_domain)
        predictions_age.append(predictions[0])
        predictions_country_part.append(predictions[1])
    analyzing_predictions([predictions_age, predictions_country_part], y_in_domain, filePath, task, "in-domain")

    predictions_age = []
    predictions_country_part = []
    for saved_model in saved_model_list:
        predictions = saved_model.predict(X_out_domain)
        predictions_age.append(predictions[0])
        predictions_country_part.append(predictions[1])
    analyzing_predictions([predictions_age, predictions_country_part], y_out_domain, filePath, task, "out-domain")


if __name__ == '__main__':
    task = 'multi-task'
    max_length_list = [500]
    num_neurons_list = [25, 50, 100]
    topic = "Watches"
    for max_length in max_length_list:
        filePath = "../../results/f-score_analysis_" + task + "_k-fold_itwac_max_length_" + str(max_length) + ".txt"

        in_domain, out_domain = load_data(topic, max_length)
        X_in_domain, y_in_domain_age = create_x_y(in_domain, "age")
        X_in_domain, y_in_domain_country_part = create_x_y(in_domain, "country_part")
        X_in_domain = pad_sequences(X_in_domain, maxlen=max_length, padding='post')
        X_out_domain, y_out_domain_age = create_x_y(out_domain, "age")
        X_out_domain, y_out_domain_country_part = create_x_y(out_domain, "country_part")
        X_out_domain = pad_sequences(X_out_domain, maxlen=max_length, padding='post')

        baseline_in_domain_age = [most_common(list(y_in_domain_age))] * len(y_in_domain_age)
        output_in_file(filePath, "w", "\nIn-domain Baseline - Age\n")
        output_in_file(filePath, "a", str(metrics.classification_report(y_in_domain_age, baseline_in_domain_age, digits=3)))
        output_in_file(filePath, "a", "\n")
        baseline_in_domain_country_part = [most_common(list(y_in_domain_country_part))] * len(y_in_domain_country_part)
        output_in_file(filePath, "a", "\nIn-domain Baseline - Country-part\n")
        output_in_file(filePath, "a", str(metrics.classification_report(y_in_domain_country_part, baseline_in_domain_country_part, digits=3)))
        output_in_file(filePath, "a", "\n")
        baseline_out_domain_age = [most_common(list(y_out_domain_age))] * len(y_out_domain_age)
        output_in_file(filePath, "a", "\nOut-domain Baseline - Age\n")
        output_in_file(filePath, "a", str(metrics.classification_report(y_out_domain_age, baseline_out_domain_age, digits=3)))
        output_in_file(filePath, "a", "\n")
        baseline_out_domain_country_part = [most_common(list(y_out_domain_country_part))] * len(y_out_domain_country_part)
        output_in_file(filePath, "a", "\nOut-domain Baseline - Country-part\n")
        output_in_file(filePath, "a", str(metrics.classification_report(y_out_domain_country_part, baseline_out_domain_country_part, digits=3)))
        output_in_file(filePath, "a", "\n")

        for num_neurons in num_neurons_list:
            print("Number of neurons " + str(num_neurons))
            output_in_file(filePath, "a", "\nNumber of neurons " + str(num_neurons) + "\n")

            path = "../../models/" + task + "/K-fold/" + str(max_length) + "/" + "neurons_" + str(num_neurons) + "/"

            list_models = os.listdir(path)
            for model in list_models:
                print(model)
                output_in_file(filePath, "a", "\n\t" + model + "\n")
                p = Process(target=predict_and_analyze,
                            args=(filePath, task, path, model, X_in_domain, [y_in_domain_age, y_in_domain_country_part], X_out_domain, [y_out_domain_age, y_out_domain_country_part],))
                p.start()
                p.join()
