import os
import pickle

import numpy
from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences


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


def load_data(topic, max_len_post):
    # load data for topic
    with open('data/final_corpus_dictionary_max_length_' + str(max_len_post) + '.pickle', 'rb') as handle:
        corpus_topic = pickle.load(handle)
    corpus_topic = transform_age_category(corpus_topic, topic, False)
    posts_id_to_del = []
    for i in range(len(corpus_topic)):
        if corpus_topic[i]["age"] == "to_del":
            posts_id_to_del.append(i)
        elif len(corpus_topic[i]["text_sequence"]) == 0:
            posts_id_to_del.append(i)
    for index in sorted(posts_id_to_del, reverse=True):
        del corpus_topic[index]

    test_user_id_watches_two_groups = [753, 670, 509, 532, 496, 758, 840, 48, 851, 813, 841, 711, 584, 788, 856, 795,
                                       833, 589, 292, 712, 803, 737, 506, 627, 363, 702, 640, 834, 539, 665, 621, 731,
                                       796, 839, 695, 638, 787, 79, 398, 514, 764, 538]

    in_domain = []
    for message_dict in corpus_topic:
        user_id = int(message_dict["user_id"])
        if user_id in test_user_id_watches_two_groups:
            in_domain.append(message_dict)

    # load data for the rest of the corpus
    with open('data/final_corpus_dictionary_max_length_' + str(max_len_post) + '.pickle', 'rb') as handle:
        out_domain = pickle.load(handle)
    out_domain = transform_age_category(out_domain, topic, True)
    posts_id_to_del = []
    for i in range(len(out_domain)):
        if out_domain[i]["age"] == "to_del":
            posts_id_to_del.append(i)
        elif len(out_domain[i]["text_sequence"]) == 0:
            posts_id_to_del.append(i)
    for index in sorted(posts_id_to_del, reverse=True):
        del out_domain[index]

    return in_domain, out_domain


def most_common(lst):
    return max(set(lst), key=lst.count)


def analyzing_predictions(predictions, right_y, filePath, description):
    results = []
    for i in range(len(predictions[0])):
        results_x = []
        for j in range(len(predictions)):
            results_x.append(predictions[j][i])

        y_confidence = 0
        for x in results_x:
            y_confidence += abs((x[0] - 0.5)) * 2
        y_confidence = str(int(round((y_confidence / 5) * 100))) + "%"

        for j in range(len(results_x)):
            results_x[j] = round(results_x[j][0])
        y_predicted = most_common(results_x)
        y_probability = str(int(results_x.count(y_predicted) * 100 / 5)) + "%"
        results.append((y_predicted, y_probability, y_confidence))

    results_y = []
    for i in range(len(results)):
        results_y.append(results[i][0])
    num_1 = results_y.count(1.0)
    num_0 = results_y.count(0.0)

    sum_prediction_right = 0
    sum_probability_right = 0
    sum_probability_wrong = 0
    sum_confidence_right = 0
    sum_confidence_wrong = 0
    guessed_right_1 = 0
    guessed_right_0 = 0
    guessed_wrong_1 = 0
    guessed_wrong_0 = 0
    for i in range(len(results)):
        if int(results[i][0]) == int(right_y[i]):
            sum_prediction_right += 1
            sum_probability_right += int(results[i][1][:-1])
            sum_confidence_right += int(results[i][2][:-1])
            if int(right_y[i] == 1):
                guessed_right_1 += 1
            else:
                guessed_right_0 += 1
        else:
            if int(right_y[i] == 1):
                guessed_wrong_1 += 1
            else:
                guessed_wrong_0 += 1
            sum_probability_wrong += int(results[i][1][:-1])
            sum_confidence_wrong += int(results[i][2][:-1])
    sum_prediction_wrong = len(results) - sum_prediction_right
    with open(filePath, "a") as f:
        f.write("\n\t\t" + description + "\n")
        f.write("\t\t\tAccuracy: " + str(round(sum_prediction_right * 10000.0 / len(results)) / 100) + "%")
        f.write("\n")
        f.write("\t\t\tNN predict 0: " + str(num_0) + " times (" + str(round(num_0*10000.0/len(results_y))/100) + "%); NN predict 1: " + str(num_1) + " times (" + str(round(num_1*10000.0/len(results_y))/100) + "%)")
        f.write("\n")
        f.write("\t\t\tGuessed right: " + str(sum_prediction_right) + " times; Mean probability: " + str(
        round(sum_probability_right * 100.0 / sum_prediction_right)/100) + "%; Mean confidence: " + str(
        round(sum_confidence_right * 100.0 / sum_prediction_right)/100) + "%")
        f.write("\n")
        f.write("\t\t\tNN rightfully predicted 0: " + str(guessed_right_0) + " times (" + str(round(guessed_right_0*10000.0/sum_prediction_right)/100) + "%); NN rightfully predicted 1: " + str(guessed_right_1) + " times (" + str(round(guessed_right_1*10000.0/sum_prediction_right)/100) + "%)")
        f.write("\n")
        f.write("\t\t\tGuessed wrong: " + str(sum_prediction_wrong) + " times; Mean probability: " + str(
        round(sum_probability_wrong * 100.0 / sum_prediction_wrong)/100) + "%; Mean confidence: " + str(
        round(sum_confidence_wrong * 100.0 / sum_prediction_wrong)/100) + "%")
        f.write("\n")
        f.write("\t\t\tNN predict 1 instead of 0: " + str(guessed_wrong_0) + " times (" + str(round(guessed_wrong_0 * 10000.0 / sum_prediction_wrong) / 100) + "% of errors); NN predict 0 instead of 1: " + str(guessed_wrong_1) + " times (" + str(round(guessed_wrong_1 * 10000.0 / sum_prediction_wrong) / 100) + "% of errors)")
        f.write("\n")


task = "age"
sub_category = "two-groups"
max_length = 100
num_neurons_list = [25, 50, 100]
topic = "Watches"
filePath = "results/analysis_" + task + "_k-fold_itwac_max_length_" + str(max_length) + ".txt"

in_domain, out_domain = load_data(topic, max_length)
X_in_domain, y_in_domain = create_x_y(in_domain, task)
X_in_domain = pad_sequences(X_in_domain, maxlen=max_length, padding='post')
X_out_domain, y_out_domain = create_x_y(out_domain, task)
X_out_domain = pad_sequences(X_out_domain, maxlen=max_length, padding='post')

with open(filePath, "w") as f:
    num_0 = 0
    num_1 = 0
    for post in in_domain:
        if post["age"] == 0:
            num_0 += 1
        if post["age"] == 1:
            num_1 += 1
    f.write("Watches:\n")
    f.write("\t<30    : " + str(num_0) + "; percent: " + str(round(num_0 * 100.0 / len(in_domain))) + "%\n")
    f.write("\t>49,<70: " + str(num_1) + "; percent: " + str(round(num_1 * 100.0 / len(in_domain))) + "%\n")

    num_0 = 0
    num_1 = 0
    for post in out_domain:
        if post["age"] == 0:
            num_0 += 1
        if post["age"] == 1:
            num_1 += 1
    f.write("Rest of the corpus:\n")
    f.write("\t<30    : " + str(num_0) + "; percent: " + str(round(num_0 * 100.0 / len(out_domain))) + "%\n")
    f.write("\t>49,<70: " + str(num_1) + "; percent: " + str(round(num_1 * 100.0 / len(out_domain))) + "%\n")

for num_neurons in num_neurons_list:
    print("Number of neurons " + str(num_neurons))
    with open(filePath, "a") as f:
        f.write("\nNumber of neurons " + str(num_neurons) + "\n")

    path = "models/" + task + "/K-fold/" + str(max_length) + "/" + sub_category + "/" + "neurons_" + str(
            num_neurons) + "/"
    list_models = os.listdir(path)
    for model in list_models:
        print(model)
        with open(filePath, "a") as f:
            f.write("\n\t" + model + "\n")

        saved_model_list = []
        for i in range(1, 6):
            path_model = path + model + "/attempt_" + str(i)
            saved_model = load_model(path_model)
            saved_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            saved_model_list.append(saved_model)

        predictions = []
        for saved_model in saved_model_list:
            predictions.append(saved_model.predict(X_in_domain))
        analyzing_predictions(predictions, y_in_domain, filePath, "in-domain")

        predictions = []
        for saved_model in saved_model_list:
            predictions.append(saved_model.predict(X_out_domain))
        analyzing_predictions(predictions, y_out_domain, filePath, "out-domain")




