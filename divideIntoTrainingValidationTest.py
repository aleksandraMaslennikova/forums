from random import shuffle

import readCorpusData





for i in range(10):
    shuffle(user_id_list)

    # dividing users into users that go to training set, validation set and test set
    training_user_id = user_id_list[:training_division_point]
    validation_user_id = user_id_list[training_division_point:validation_division_point]
    test_user_id = user_id_list[training_division_point+validation_division_point:]
