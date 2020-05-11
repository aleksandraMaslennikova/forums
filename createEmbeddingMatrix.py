import pickle

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import numpy
import readCorpusData

def create_embedding_matrix(word_embedding_dictionary, tokenizer):
    embedding_matrix = numpy.zeros((vocab_size, 129))
    num_words = 0
    for word, i in tokenizer.word_index.items():
        if num_words % 1000 == 0:
            print(str(num_words) + " words from " + str(len(t.word_index)) + " transformed in Word Embeddings.")
        word_arr = word.split("__")
        word_arr[-1] = word_arr[-1].upper()
        if word_embedding_dictionary == "itwac":
            embedding_vector = readCorpusData.getWordEmbeddingsItwac("__".join(word_arr))
        else:
            embedding_vector = readCorpusData.getWordEmbeddingsTwitter("__".join(word_arr))
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        num_words += 1
    return embedding_matrix

corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 1000)
docs = []
for post in corpus:
    docs.append(post["text"])

t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1

encoded_docs = t.texts_to_sequences(docs)
for i in range(len(corpus)):
    corpus[i]["text_sequence"] = encoded_docs[i]

with open('data/final_corpus_dictionary_max_length_1000.pickle', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 500)
docs = []
for post in corpus:
    docs.append(post["text"])
encoded_docs = t.texts_to_sequences(docs)
for i in range(len(corpus)):
    corpus[i]["text_sequence"] = encoded_docs[i]
with open('data/final_corpus_dictionary_max_length_500.pickle', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 200)
docs = []
for post in corpus:
    docs.append(post["text"])
encoded_docs = t.texts_to_sequences(docs)
for i in range(len(corpus)):
    corpus[i]["text_sequence"] = encoded_docs[i]
with open('data/final_corpus_dictionary_max_length_200.pickle', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 100)
docs = []
for post in corpus:
    docs.append(post["text"])
encoded_docs = t.texts_to_sequences(docs)
for i in range(len(corpus)):
    corpus[i]["text_sequence"] = encoded_docs[i]
with open('data/final_corpus_dictionary_max_length_100.pickle', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/itwac_word_embedding_matrix.pickle', 'wb') as handle:
    pickle.dump(create_embedding_matrix("itwac", t), handle, protocol=pickle.HIGHEST_PROTOCOL)
#with open('data/twitter_word_embedding_matrix.pickle', 'wb') as handle:
#    pickle.dump(create_embedding_matrix("twitter", t), handle, protocol=pickle.HIGHEST_PROTOCOL)
