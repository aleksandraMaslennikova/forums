import pickle
import readCorpusData

corpus = {}
corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 500)
corpus = readCorpusData.transformTextToWordEmbeddings(corpus)

with open('data/dict/final_corpus_max_length_500.pickle', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
corpus = {}
corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 1000)
corpus = readCorpusData.transformTextToWordEmbeddings(corpus)

with open('data/dict/final_corpus_max_length_1000.pickle', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
