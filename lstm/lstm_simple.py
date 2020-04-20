from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
import readCorpusData
from random import shuffle

def createXandY(corpus, task):
    X = []
    y = []
    for message in corpus:
        X.append(message["word_embedding"])
        y.append(message[task])
    return X, y


# constants
max_review_length = 100
task = "gender"    # 0 - male, 1 - female
training_set_percentage = 0.75

# load the dataset and create training set and test set
corpus = readCorpusData.readCorpusFromFile("data/final_corpus.txt", max_review_length)
corpus = readCorpusData.transformTextToWordEmbeddings(corpus)
division_point = training_set_percentage * len(corpus)
corpus = shuffle(corpus)
training = corpus[:division_point]
test = corpus[division_point:]
X_train, y_train = createXandY(training, task)
X_test, y_test = createXandY(test, task)
# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
model = Sequential()
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

