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


def calculate_accuracy_nb(train_data, test_data):

    correct_predictions = 0

    true_labels = [row[-1] for row in test_data]
    total_predictions = len(true_labels)
    
    predicted_labels = classify_nb(train_data, test_data)
    
    for i in range(total_predictions):
        if true_labels[i] == predicted_labels[i]:
            correct_predictions += 1

    accuracy_value = correct_predictions / len(true_labels)

    return accuracy_value


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

        # Accumulate results over however many folds
        total_correct += correct_predictions
        total_samples += num_samples
        
        average_accuracy = total_correct / total_samples

    return average_accuracy

    

if __name__ == "__main__":

    filename = "pima-folds.csv"
    num_folds = 10

    average_accuracy = cross_validation(filename, num_folds, algorithm_type='KNN', kth_nearest_neighbour=7)

    print(f"Accuracy (from file-based folds): {average_accuracy:.4%}")

