import re
import sqlite3

def readCorpusFromFile(filePath, docMaxLength):
    corpus_arr = []
    with open(filePath, encoding="utf8") as f:
        line = f.readline()
        text_arr = []
        dict = {}
        num_words = 0
        while line:
            if line.startswith("<doc"):
                if len(corpus_arr) % 100 == 0:
                    print(str(len(corpus_arr)) + " messages analyzed.")
                id = re.findall('id=".+?"', line)[0].split('"')[1]
                forum = re.findall('forum=".+?"', line)[0].split('"')[1]
                dict["forum"] = forum
                user_id = re.findall('user_id=".+?"', line)[0].split('"')[1]
                dict["user_id"] = user_id
                gender = re.findall('task_gender=".+?"', line)[0].split('"')[1]
                if gender == "male":
                    dict["gender"] = 0
                else:
                    dict["gender"] = 1
                age = re.findall('task_age=".+?"', line)[0].split('"')[1]
                dict["age"] = age
                location = re.findall('task_location=".+?"', line)[0].split('"')[1]
                dict["location"] = location
                region = re.findall('task_regione=".+?"', line)[0].split('"')[1]
                if region == "Abruzzo":
                    dict["region"] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Marche":
                    dict["region"] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Calabria":
                    dict["region"] = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Toscana":
                    dict["region"] = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Valle d'Aosta/Vallée d'Aoste":
                    dict["region"] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Sardegna":
                    dict["region"] = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Puglia":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Lazio":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Lombardia":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Trentino-Alto Adige/Südtirol":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Umbria":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Emilia-Romagna":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Sicilia":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                elif region == "Piemonte":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                elif region == "Campania":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                elif region == "Veneto":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                elif region == "Friuli-Venezia Giulia":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                elif region == "Basilicata":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                elif region == "Liguria":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                elif region == "Molise":
                    dict["region"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                country_part = re.findall('task_ripartizione_geografica=".+?"', line)[0].split('"')[1]
                if country_part == "Nord-ovest":
                    dict["country_part"] = [1, 0, 0, 0, 0]
                elif country_part == "Nord-est":
                    dict["country_part"] = [0, 1, 0, 0, 0]
                elif country_part == "Centro":
                    dict["country_part"] = [0, 0, 1, 0, 0]
                elif country_part == "Sud":
                    dict["country_part"] = [0, 0, 0, 1, 0]
                else:
                    dict["country_part"] = [0, 0, 0, 0, 1]
                forums_thematic = re.findall('forums_thematic=".+?"', line)[0].split('"')[1]
                dict["forums_thematic"] = forums_thematic
            elif re.match("^[0-9]", line):
                line_arr = line.split("\t")
                # line_arr[2] - word after lemmatization
                text_arr.append(line_arr[1].lower() + "__" + line_arr[4])
                num_words += 1
            elif line == "\n" and num_words != 0:
                if num_words >= docMaxLength:
                    while not line.startswith("</doc"):
                        line = f.readline()
            if line.startswith("</doc"):
                dict["text"] = text_arr
                corpus_arr.append(dict)
                text_arr = []
                dict = {}
                num_words = 0
            line = f.readline()
    return corpus_arr

def getWordEmbeddingsItwac(word):
    conn = sqlite3.connect("C:\\Users\\user\\PycharmProjects\\forums_research\\data\\itwac_v2.db")
    cursor = conn.cursor()
    if '"' in word and "'" in word:
        return None
    if '"' not in word:
        cursor.execute('SELECT * FROM store WHERE key="' + word + '"')
    else:
        cursor.execute("SELECT * FROM store WHERE key='" + word + "'")
    results = cursor.fetchall()
    conn.close()
    try:
        return results[0][1:]
    except:
        pass
    return None

def transformTextToWordEmbeddings(corpus):
    for i in range(len(corpus)):
        if i % 100 == 0:
            print(str(i) + " messages transformed in Word Embeddings.")
        corpus[i]["word_embedding"] = []
        for j in range(len(corpus[i]["text"])):
            embedding = getWordEmbeddingsItwac(corpus[i]["text"][j])
            if embedding is not None:
                corpus[i]["word_embedding"].append(embedding)
    return corpus