from random import shuffle

import readCorpusData


def get_statistics(corpus, task, task_options_list):
    dict = {}
    for option in task_options_list:
        dict[option] = 0
    for post in corpus:
        dict[post[task]] += 1
    return dict

thematic = "Watches"
task = "country_part"
development_percentage = 0.75

corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 100)

for post in corpus:
    forums_thematic = post["forums_thematic"]
    if forums_thematic != thematic:
        post[task] = "to_del"
    else:
        if type(post[task]) is list:
            post[task] = post[task].index(1)
posts_id_to_del = []

for i in range(len(corpus)):
    if corpus[i][task] == "to_del":
        posts_id_to_del.append(i)
    elif len(corpus[i]["text"]) == 0:
        posts_id_to_del.append(i)
for index in sorted(posts_id_to_del, reverse=True):
    del corpus[index]

user_id_list = []
for message_dict in corpus:
    user_id_list.append(int(message_dict["user_id"]))
user_id_list = list(set(user_id_list))

task_options_list = []
for message_dict in corpus:
    task_options_list.append(message_dict[task])
task_options_list = list(set(task_options_list))

corpus_statistics = get_statistics(corpus, task, task_options_list)
print("Corpus")
for option in corpus_statistics:
    print("\t" + str(option) + ": " + str(corpus_statistics[option]) + " (" + str(round(corpus_statistics[option]*10000/len(corpus))/100) + "%)")

development_division_point = round(development_percentage * len(user_id_list))

for i in range(10):
    print("Alternative ", i+1)
    shuffle(user_id_list)

    development_user_id = user_id_list[:development_division_point]

    development = []
    test = []
    for message_dict in corpus:
        user_id = int(message_dict["user_id"])
        if user_id in development_user_id:
            development.append(message_dict)
        else:
            test.append(message_dict)

    development_statistics = get_statistics(development, task, task_options_list)
    test_statistics = get_statistics(test, task, task_options_list)
    for option in task_options_list:
        print("\t\tc: " + str(round(corpus_statistics[option]*10000/len(corpus))/100) + "%; d: " + str(round(development_statistics[option]*10000/len(development))/100) + "%; t: " + str(round(test_statistics[option] * 10000 / len(test)) / 100) + "%")

    print(development_user_id)


