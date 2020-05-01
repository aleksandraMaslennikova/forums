import pickle
import readCorpusData

corpus = {}
corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 500)
corpus = readCorpusData.transformTextToWordEmbeddings(corpus, "itwac")

with open('data/dict/final_corpus_itwac_max_length_500.pickle', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

try:
    del(corpus)
except Exception as e:
    print("Not deleted")
    pass
print()

corpus = {}
corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", 500)
corpus = readCorpusData.transformTextToWordEmbeddings(corpus, "twitter")

with open('data/dict/final_corpus_twitter_max_length_500.pickle', 'wb') as handle:
    pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

