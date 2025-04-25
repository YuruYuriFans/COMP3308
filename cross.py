from naive import *
from nearest import *
from ensemble import *

import pandas as pd
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

# Example usage:
folds = read_folds("pima-folds.csv")
# Print out the number of data rows in each fold:
for fold_name, rows in folds.items():
    print(f"{fold_name} has {len(rows)} rows.")
    # For demonstration, print the first row of each fold
    if rows:
        print(" First row:", rows[0])


def cross_validation(filename: str):
    return
    folds.data = pd.read_csv(filename, header=None)
    