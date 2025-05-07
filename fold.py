import pandas as pd
import random

def fold(training_filename: str, testing_filename: str, folds: int = 10) -> list[str]:
    """
    stratify the training into k folds and write it to a file
    """
    training_data = pd.read_csv(training_filename, header=None)
    training_list = training_data.values.tolist()
    # approximately equal yes and no in each fold
    training_yes = [record for record in training_list if record[-1] == "yes"]
    training_no = [record for record in training_list if record[-1] == "no"]
    
    # randomly shuffle the data
    random.shuffle(training_yes)
    random.shuffle(training_no)
    
    # split the data into k folds
    all_folds = []
    for i in range(folds):
        fold_yes = training_yes[i::folds]
        fold_no = training_no[i::folds]
        each_fold = fold_yes + fold_no
        if i == 0:
            if len(fold_yes) > len(fold_no):
                buffer = fold_yes.pop()
            else:
                buffer = fold_no.pop()
        elif i == folds - 1:
            if len(fold_yes) > len(fold_no):
                fold_yes.append(buffer)
            else:
                fold_no.append(buffer)
        # append each fold to a csv file
        with open(f"{testing_filename}", "a") as f:
            f.write(f"fold{i + 1}\n")
            for record in each_fold:
                f.write(",".join([str(x) for x in record]) + "\n")
            f.write("\n")
            print(f"fold {i + 1} has {len(fold_yes)} yes and {len(fold_no)} no")
            # accacy, precision, recall, f1 = find_accuracy_for_KNN(training_list, each_fold, 3) score  
            # dffeence eason: knn is not a good classifier for many features, but naive bayes is still a good classifier.
        
    print(len(all_folds))

    
if __name__ == "__main__":
    print("occupancy")
    training_file = "occupancy.csv"
    testing_file = "occupancy-folds.csv"
    folds = 10
    fold(training_file, testing_file, folds)
    print("pima")
    training_file = "pima.csv"
    testing_file = "pima-folds.csv"
    folds = 10
    fold(training_file, testing_file, folds)

