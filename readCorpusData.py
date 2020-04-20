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
                dict["region"] = region
                country_part = re.findall('task_ripartizione_geografica=".+?"', line)[0].split('"')[1]
                dict["country_part"] = country_part
                forums_thematic = re.findall('forums_thematic=".+?"', line)[0].split('"')[1]
                dict["forums_thematic"] = forums_thematic
            elif line.startswith("</doc"):
                dict["text"] = text_arr
                corpus_arr.append(dict)
                if num_words >= docMaxLength:
                    break;
                text_arr = []
                dict = {}
            elif re.match("^[0-9]", line):
                line_arr = line.split("\t")
                # line_arr[2] - word after lemmatization
                text_arr.append(line_arr[1].lower() + "__" + line_arr[4])
                num_words += 1
            line = f.readline()
    return corpus_arr

def getWordEmbeddingsItwac(word):
    conn = sqlite3.connect("C:\\Users\\user\\PycharmProjects\\forums_research\\data\\itwac_v2.db")
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM store WHERE key="' + word + '"')
    results = cursor.fetchall()
    conn.close()
    try:
        return results[0][1:]
    except:
        print(word)
    return None

def transformTextToWordEmbeddings(corpus):
    for i in range(len(corpus)):
        corpus[i]["word_embedding"] = []
        for j in range(len(corpus[i]["text"])):
            embedding = getWordEmbeddingsItwac(corpus[i]["text"][j])
            if embedding is not None:
                corpus[i]["word_embedding"].append(embedding)
    return corpus

corpus = readCorpusFromFile("data/final_corpus.txt", 1000)
corpus = transformTextToWordEmbeddings(corpus)
print(len(corpus[0]["word_embedding"][0]))
print(corpus[0]["word_embedding"][1])
print(corpus[0]["word_embedding"][2])
