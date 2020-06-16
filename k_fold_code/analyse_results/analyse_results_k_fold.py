import re
import numpy as np

max_length = 1000
num_neurons = 50
task = "age"
word_embedding = "itwac"
fileName = "results_" + task + "_k-fold_" + word_embedding + "_max_length_" + str(max_length) + "_num_neurons_" + str(num_neurons) + ".txt"
path = "../../results/"

arr_results = []
n = -1
with open(path + fileName, 'r') as f:
    line = f.readline()
    while line:
        if re.match("^[A-Z]", line) and not re.match("Watches", line) and not re.match("Rest of the corpus", line):
            n += 1
            arr_results.append({})
            arr_results[n]["name"] = line[:-1]
            arr_results[n]["training_arr"] = []
            arr_results[n]["validation_arr"] = []
        elif re.match("^\t\tRESULTS", line):
            arr_results[n]["training_arr"].append(float(line.split("Train:")[1].split("%,")[0]))
            arr_results[n]["validation_arr"].append(float(line.split("Validation:")[1].split("%")[0]))
        line = f.readline()

for result in arr_results:
    print(result["name"])
    print("training")
    print(result["training_arr"])
    print(str(round(np.mean(result["training_arr"]), 2)) + "%")
    print("validation")
    print(result["validation_arr"])
    print(str(round(np.mean(result["validation_arr"]), 2)) + "%")
    print()
