import pandas as pd
import math

def normal_pdf(x, mu=0, sigma=1):
    # If the standard deviation is 0, then all values are equal to the mean. Thus, P(X = mean) = 1 and all other probabilities are 0.
    if sigma == 0:
        return 1.0 if x == mu else 0.0
    coefficient = 1 / (sigma * math.sqrt(2 * math.pi))
    exponent = math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coefficient * exponent

def classify_nb(training_filename: str, testing_filename: str) -> list[str]:
    training_data = pd.read_csv(training_filename, header=None)
    testing_data = pd.read_csv(testing_filename, header=None)
    # print(testing_data)
    training_list = training_data.values.tolist()
    testing_list = testing_data.values.tolist()
    #print(training_list)
    # P(yes | E) = P(E | yes) * P(yes) / P(E)
    # P(E | yes) = P(E1 | yes) * P(E2 | yes) * ... * P(En | yes)
    # each Ei is calculated using normal distribution pdf
    
    training_yes = [record for record in training_list if record[-1] == "yes"]
    training_no = [record for record in training_list if record[-1] == "no"]
    
    p_yes = len(training_yes) / len(training_list)
    p_no = len(training_no) / len(training_list)
    
    #calculate mean and std for each attribute
    means_yes = [sum(float(record[i]) for record in training_yes) / len(training_yes) for i in range(len(training_yes[0]) - 1)]
    stds_yes = [math.sqrt(sum((float(record[i]) - means_yes[i]) ** 2 for record in training_yes) / len(training_yes)) for i in range(len(training_yes[0]) - 1)]
    means_no = [sum(float(record[i]) for record in training_no) / len(training_no) for i in range(len(training_no[0]) - 1)]
    stds_no = [math.sqrt(sum((float(record[i]) - means_no[i]) ** 2 for record in training_no) / len(training_no)) for i in range(len(training_no[0]) - 1)]
    
    result = []
    for each_test in testing_list:
        p_yes_given_e = p_yes
        p_no_given_e = p_no
        for i in range(len(each_test)):
            # calculate P(Ei | yes) and P(Ei | no)
            p_ei_given_yes = normal_pdf(float(each_test[i]), means_yes[i], stds_yes[i])
            p_ei_given_no = normal_pdf(float(each_test[i]), means_no[i], stds_no[i])
            # multiply P(Ei | yes) and P(Ei | no)
            p_yes_given_e *= p_ei_given_yes
            p_no_given_e *= p_ei_given_no
        if p_yes_given_e >= p_no_given_e:
            result.append("yes")
        else:
            result.append("no")
    return result

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

def classify_ens(training_filename, testing_filename, k1, k2):
    result1 = classify_nb(training_filename, testing_filename)
    result2 = classify_nn(training_filename, testing_filename, k1)
    result3 = classify_nn(training_filename, testing_filename, k2)
    
    final_result = []
    for i in range(len(result1)):
        if result1[i] == result2[i] or result1[i] == result3[i]:
            final_result.append(result1[i])
        elif result2[i] == result3[i]:
            final_result.append(result2[i])
    return final_result