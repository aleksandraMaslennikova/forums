from random import shuffle

import readCorpusData


def create_list_of_possible_categories_gender(corpus):
    gender = []
    for post in corpus:
        if post["gender"] == 0:
            post["gender"] = "Male"
        else:
            post["gender"] = "Female"
        gender.append(post["gender"])
    return list(set(gender))


def create_list_of_possible_categories_age(corpus):
    age_list = []
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
        age_list.append(post["age"])
    return list(set(age_list))


def create_list_of_possible_categories_country_part(corpus):
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
    return list(set(ripartizione_geografica))


def create_list_of_possible_categories_region(corpus):
    region = []
    for post in corpus:
        if post["region"] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Abruzzo"
        elif post["region"] == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Marche"
        elif post["region"] == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Calabria"
        elif post["region"] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Toscana"
        elif post["region"] == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Valle d'Aosta/Vallée d'Aoste"
        elif post["region"] == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Sardegna"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Puglia"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Lazio"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Lombardia"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Trentino-Alto Adige/Südtirol"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Umbria"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Emilia-Romagna"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Sicilia"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]:
            post["region"] = "Piemonte"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]:
            post["region"] = "Campania"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]:
            post["region"] = "Veneto"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]:
            post["region"] = "Friuli-Venezia Giulia"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]:
            post["region"] = "Basilicata"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]:
            post["region"] = "Liguria"
        elif post["region"] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
            post["region"] = "Molise"
        region.append(post["region"])
    return list(set(region))


def create_list_of_possible_forums_thematics(corpus):
    thematics = []
    for post in corpus:
        thematics.append(post["forums_thematic"])
    return list(set(thematics))


def get_statistics_for_category(corpus, category_name, category_list, thematic):
    dict = {}
    for category in category_list:
        dict[category] = 0
    for post in corpus:
        if post["forums_thematic"] == thematic or thematic == "All":
            dict[post[category_name]] += 1
    return dict

def get_statistics(corpus,category,category_list,thematic):
    statistics = get_statistics_for_category(corpus, category, category_list, thematic)
    total = 0
    for option in category_list:
        total += statistics[option]
    line1 = category.title() + ";Number of posts;;"
    line2 = ";Percentage;;"
    line3 = ";;;;;;;;;;;;;;;;"
    for option in category_list:
        line1 += option + ";"
        line2 += ";"
        line1 += str(statistics[option]) + ";"
        line2 += str(round(int(statistics[option]) * 100.0 / int(total))) + "%;"
    line1 += ";Total;" + str(total) + ";"
    line2 += ";;;"
    return line1 + "\n" + line2 + "\n" + line3 + "\n"


def write_statistics_in_file(corpus, corpus_part, fout):
    fout.write(get_statistics(corpus, "gender", gender_list, "All"))
    fout.write(get_statistics(corpus_part, "gender", gender_list, "All"))
    fout.write(get_statistics(corpus, "age", age_list, "All"))
    fout.write(get_statistics(corpus_part, "age", age_list, "All"))
    fout.write(get_statistics(corpus, "country_part", country_part_list, "All"))
    fout.write(get_statistics(corpus_part, "country_part", country_part_list, "All"))
    fout.write(get_statistics(corpus, "region", region_list, "All"))
    fout.write(get_statistics(corpus_part, "region", region_list, "All"))


corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 100)

gender_list = create_list_of_possible_categories_gender(corpus)
age_list = create_list_of_possible_categories_age(corpus)
country_part_list = create_list_of_possible_categories_country_part(corpus)
region_list = create_list_of_possible_categories_region(corpus)
thematics_list = create_list_of_possible_forums_thematics(corpus)

thematics_list.append("All")

training_percentage = 0.5
validation_percentage = 0.25

user_id_list = []
for message_dict in corpus:
    user_id_list.append(int(message_dict["user_id"]))
user_id_list = list(set(user_id_list))

training_division_point = round(training_percentage * len(user_id_list))
validation_division_point = training_division_point + round(validation_percentage * len(user_id_list))

with open("data/statistics.csv", "w", encoding="utf-8") as f:
    for i in range(10):
        f.write(";;;;;;;;;;;;;;;;\n")
        f.write("Alternative " + str(i) + ";;;;;;;;;;;;;;;;\n")
        shuffle(user_id_list)

        # dividing users into users that go to training set, validation set and test set
        training_user_id = user_id_list[:training_division_point]
        validation_user_id = user_id_list[training_division_point:validation_division_point]

        training = []
        validation = []
        test = []
        for message_dict in corpus:
            user_id = int(message_dict["user_id"])
            if user_id in training_user_id:
                training.append(message_dict)
            elif user_id in validation_user_id:
                validation.append(message_dict)
            else:
                test.append(message_dict)

        f.write("Division proportion;Training;" + str(round((len(training)*100.0 / len(corpus)))) + "%;Validation;" + str(round((len(validation)*100.0 / len(corpus)))) + "%;Test;" + str(round((len(test)*100.0 / len(corpus)))) + "%;\n")

        f.write("Training set;;" + str(training_user_id) + ";;;;;;;;;;;;;;\n")
        write_statistics_in_file(corpus, training, f)
        f.write("Validation set;;" + str(validation_user_id) + ";;;;;;;;;;;;;;\n")
        write_statistics_in_file(corpus, validation, f)
        f.write("Test set;;;;;;;;;;;;;;;;\n")
        write_statistics_in_file(corpus, test, f)
        f.write(";;;;\n")
