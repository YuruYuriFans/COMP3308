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

if __name__ == "__main__":
    training_file = "training.csv"
    testing_file = "testing.csv"
    result = classify_nb(training_file, testing_file)
    print(result)
