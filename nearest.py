import pandas as pd
import matplotlib.pyplot as plt

class Point:
    def __init__(self, attributes: list[float], label: str | None):
        self.attributes = attributes
        # if training, label = None
        self.label = label
        # Point and the distance to that point, moved to test_against_training.

    def __repr__(self):
        return f"Point(attributes={self.attributes}, label={self.label})"

def euclidean_distance(point1: Point, point2: Point) -> float:
    # ignore error handling 
    if len(point1.attributes) != len(point2.attributes):
        print("Error: Points have different dimensions")
    return sum((a - b) ** 2 for a, b in zip(point1.attributes, point2.attributes)) ** 0.5

def test_against_training(training_points: list[Point], test_point: Point, k: int) -> str:
    # Idea: calculate the distance between the test point and all training points
    # and select the k smallest distances
    distances = [(point, euclidean_distance(point, test_point)) for point in training_points]
    sorted_points = sorted(distances, key=lambda item: item[1])

    # Get the labels of the k nearest neighbors
    k_nearest_neighbors = sorted_points[:k]
    labels = [point.label for point, _ in k_nearest_neighbors]
    yes_count = labels.count("yes")
    no_count = labels.count("no")
    if yes_count >= no_count:
        return "yes"
    else:
        return "no"

def classify_nn(training_filename: str, testing_filename: str, k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be a positive integer")

    result: list[str] = list()

    training_data = pd.read_csv(training_filename, header=None)
    testing_data = pd.read_csv(testing_filename, header=None)
    # print(testing_data)
    training_list = training_data.values.tolist()
    testing_list = testing_data.values.tolist()

    training_points = [Point([float(x) for x in row[:-1]], row[-1]) for row in training_list]
    testing_points = [Point([float(x) for x in row], None) for row in testing_list]
    for each_testing_point in testing_points:
        predicted_label = test_against_training(training_points, each_testing_point, k)
        each_testing_point.label = predicted_label

    result = [each_testing_point.label for each_testing_point in testing_points]

    return result


def evaluate_accuracy(training_file: str, testing_file: str, k: int) -> float:
    testing_data = pd.read_csv(testing_file, header=None).values.tolist()
    true_labels = [row[-1] for row in testing_data]
    
    predicted_labels = classify_nn(training_file, testing_file, k)
    
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)

    accuracy_value = correct / len(true_labels)

    return accuracy_value


if __name__ == "__main__":
    training_file = "training.csv"
    testing_file = "testing.csv"
    k = 1

    predictions = classify_nn(training_file, testing_file, k)
    print(predictions)

    accuracy = (evaluate_accuracy(training_file, testing_file, k))
    print(accuracy)
