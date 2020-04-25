import pickle

with open('data/dict/final_corpus_max_length_100.pickle', 'rb') as handle:
    corpus = pickle.load(handle)

ripartizione_geografica = []
for post in corpus:
    if post["country_part"] == [1, 0, 0, 0, 0]:
        post["country_part"] = "Nord-ovest"
    elif post["country_part"] == [0, 1, 0, 0, 0]:
        post["country_part"] = "Nord-est"
    elif post["country_part"] == [0, 0, 1, 0, 0]:
        post["country_part"] = "Centro"
    elif post["country_part"] == [0, 0, 0, 1, 0]:
        post["country_part"] = "Sud"
    else:
        post["country_part"] = "Isole"
    ripartizione_geografica.append(post["country_part"])
ripartizione_geografica_list = list(set(ripartizione_geografica))
dict = {}
for part in ripartizione_geografica_list:
    dict[part] = 0
for post in corpus:
    dict[post["country_part"]] += 1
for part in ripartizione_geografica_list:
    print(part)
    print(dict[part])