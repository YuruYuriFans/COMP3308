from naive import *
from nearest import *
from ensemble import *
from fold import *

import pandas as pd
import numpy as np
import math
import random

import csv


def read_folds(filename):
    folds = {}
    current_fold = None
    with open(filename, 'r', newline='') as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue  # skip empty lines
            # If the line indicates a new fold (e.g., "fold1")
            if stripped_line.lower().startswith("fold"):
                current_fold = stripped_line  # e.g., "fold1", "fold2", etc.
                folds[current_fold] = []        # prepare a list for this fold’s data
            else:
                # Use csv.reader to parse the comma-separated values.
                # We wrap the line in a list because csv.reader expects an iterable of lines.
                row = next(csv.reader([stripped_line]))
                folds[current_fold].append(row)
    return folds

# def find_accuracy()

def find_accuracy_for_KNN(train_data, test_data, kth_nearest_neighbour):
    training_points = [Point([float(x) for x in row[:-1]], row[-1]) for row in train_data]
    testing_points = [Point([float(x) for x in row[:-1]], row[-1]) for row in test_data]

    correct = 0
    for test_point in testing_points:
        predicted = test_against_training(training_points, Point(test_point.attributes, None), kth_nearest_neighbour)
        if predicted == test_point.label:
            correct += 1
    return correct, len(testing_points)


def classify_nb_from_lists(train_list: list[list[str]], test_list: list[list[str]]) -> list[str]:
    training_yes = [record for record in train_list if record[-1] == "yes"]
    training_no = [record for record in train_list if record[-1] == "no"]

    p_yes = len(training_yes) / len(train_list)
    p_no = len(training_no) / len(train_list)

    means_yes = [sum(float(record[i]) for record in training_yes) / len(training_yes) for i in range(len(training_yes[0]) - 1)]
    stds_yes = [math.sqrt(sum((float(record[i]) - means_yes[i]) ** 2 for record in training_yes) / len(training_yes)) for i in range(len(training_yes[0]) - 1)]
    means_no = [sum(float(record[i]) for record in training_no) / len(training_no) for i in range(len(training_no[0]) - 1)]
    stds_no = [math.sqrt(sum((float(record[i]) - means_no[i]) ** 2 for record in training_no) / len(training_no)) for i in range(len(training_no[0]) - 1)]

    result = []
    for each_test in test_list:
        p_yes_given_e = p_yes
        p_no_given_e = p_no
        for i in range(len(each_test)):
            p_ei_given_yes = normal_pdf(float(each_test[i]), means_yes[i], stds_yes[i])
            p_ei_given_no = normal_pdf(float(each_test[i]), means_no[i], stds_no[i])
            p_yes_given_e *= p_ei_given_yes
            p_no_given_e *= p_ei_given_no
        result.append("yes" if p_yes_given_e >= p_no_given_e else "no")
    return result


def calculate_accuracy_nb(train_data, test_data):
    true_labels = [row[-1] for row in test_data]
    test_data_no_labels = [row[:-1] for row in test_data]

    predicted_labels = classify_nb_from_lists(train_data, test_data_no_labels)

    correct_predictions = sum(1 for i in range(len(true_labels)) if true_labels[i] == predicted_labels[i])
    return correct_predictions, len(true_labels)


def cross_validation(filename: str, num_folds, algorithm_type, kth_nearest_neighbour):

    folds = read_folds(filename)

    sorted_folds = [folds[f"fold{i+1}"] for i in range(num_folds)] # f"fold{i+1}" is the same as "fold" + str(i+1)

    total_correct = 0
    total_samples = 0

    for fold_index in range(num_folds):
        test_fold = sorted_folds[fold_index]

        # Combine all other folds into the training set
        train_folds = []
        for other_index in range(num_folds):
            if other_index != fold_index:
                train_folds.extend(sorted_folds[other_index])

        if algorithm_type == 'KNN':
        # Evaluate accuracy using k-NN
            correct_predictions, num_samples = find_accuracy_for_KNN(train_folds, test_fold, kth_nearest_neighbour)

        elif algorithm_type == 'NB':
            correct_predictions, num_samples = calculate_accuracy_nb(train_folds, test_fold)

        elif algorithm_type == 'ENS':
            correct_predictions, num_samples = calculate_accuracy_ensemble(train_folds, test_fold, 1, 7)

        # Accumulate results over however many folds
        total_correct += correct_predictions
        total_samples += num_samples
        
        average_accuracy = total_correct / total_samples

    return average_accuracy


def classify_nn_from_lists(train_list: list[list[str]], test_list: list[list[str]], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be a positive integer")

    training_points = [Point([float(x) for x in row[:-1]], row[-1]) for row in train_list]
    testing_points = [Point([float(x) for x in row], None) for row in test_list]

    for test_point in testing_points:
        predicted_label = test_against_training(training_points, test_point, k)
        test_point.label = predicted_label

    return [point.label for point in testing_points]



def classify_ens_from_lists(train_data, test_data, k1, k2):
    test_data_no_labels = [row[:-1] for row in test_data]

    result1 = classify_nb_from_lists(train_data, test_data_no_labels)
    result2 = classify_nn_from_lists(train_data, test_data_no_labels, k1)
    result3 = classify_nn_from_lists(train_data, test_data_no_labels, k2)

    final_result = []
    for i in range(len(result1)):
        if result1[i] == result2[i] or result1[i] == result3[i]:
            final_result.append(result1[i])
        elif result2[i] == result3[i]:
            final_result.append(result2[i])
        else:
            final_result.append(result1[i])  # fallback (optional)
    return final_result



def calculate_accuracy_ensemble(train_data, test_data, k1, k2):
    true_labels = [row[-1] for row in test_data]
    predicted_labels = classify_ens_from_lists(train_data, test_data, k1, k2)

    correct_predictions = sum(1 for i in range(len(true_labels)) if true_labels[i] == predicted_labels[i])
    return correct_predictions, len(true_labels)




if __name__ == "__main__":

    filename = "occupancy-folds.csv"
    num_folds = 10

    average_accuracy = cross_validation(filename, num_folds, algorithm_type='ENS', kth_nearest_neighbour=1)

    print(f"Accuracy (from file-based folds): {average_accuracy:.4%}")

