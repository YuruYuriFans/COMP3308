import pandas as pd
import matplotlib.pyplot as plt

def classify_nn(training_filename, testing_filename, k):
    training_data = pd.read_csv(training_filename)
    testing_data = pd.read_csv(testing_filename)
    
    result : list[str] = list()
    return result