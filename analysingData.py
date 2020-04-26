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


corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 1000)
gender_list = create_list_of_possible_categories_gender(corpus)
thematics_list = create_list_of_possible_forums_thematics(corpus)
thematics_list.append("All")
with open("data/gender_statistics.csv", "w", encoding="utf-8") as f:
    for thematic in thematics_list:
        statistics = get_statistics_for_category(corpus, "gender", gender_list, thematic)
        total = 0
        for gender in gender_list:
            total += statistics[gender]
        line1 = thematic + ";Number of posts;;"
        line2 = ";Percentage;;"
        line3 = ";;;;;;;;;;;;;;;;"
        for gender in gender_list:
            line1 += gender + ";"
            line2 += ";"
            line1 += str(statistics[gender]) + ";"
            line2 += str(round(int(statistics[gender])*100.0/int(total))) + "%;"
        line1 += ";Total;" + str(total) + ";"
        line2 += ";;;"
        f.write(line1 + "\n")
        f.write(line2 + "\n")
        f.write(line3 + "\n")
