import pickle
import readCorpusData

def createDictFile(max_length, word_embedding_dict):
    corpus = readCorpusData.readCorpusFromFile("../data/final_corpus.txt", max_length)
    corpus = readCorpusData.transformTextToWordEmbeddings(corpus, word_embedding_dict)

    with open('data/dict/' + str(word_embedding_dict) + '/' + str(max_length) + '/final_corpus_' + str(word_embedding_dict) + '_max_length_' + str(max_length) + '.pickle', 'wb') as handle:
        pickle.dump(corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        del(corpus)
    except Exception as e:
        print("Not deleted")
        pass


createDictFile(100, "itwac")
createDictFile(100, "twitter")
createDictFile(200, "itwac")
createDictFile(200, "twitter")
createDictFile(500, "itwac")
createDictFile(500, "twitter")
createDictFile(1000, "itwac")
createDictFile(1000, "twitter")

