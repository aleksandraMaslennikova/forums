import pickle
from random import shuffle

max_review_length = 100
word_embedding_dict = "itwac"

with open('data/dict/final_corpus_max_length_' + str(max_review_length) + '.pickle', 'rb') as handle:
    corpus = pickle.load(handle)

training_percentage = 0.5
validation_percentage = 0.25

user_id_list = []
for message_dict in corpus:
    user_id_list.append(int(message_dict["user_id"]))
user_id_list = list(set(user_id_list))

training_division_point = round(training_percentage * len(user_id_list))
validation_division_point = training_division_point + round(validation_percentage * len(user_id_list))

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

with open('data/dict/final_corpus_itwac_max_length_500.pickle', 'wb') as handle:
    pickle.dump(training, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/dict/final_corpus_itwac_max_length_500.pickle', 'wb') as handle:
    pickle.dump(validation, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/dict/final_corpus_itwac_max_length_500.pickle', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

