import matplotlib.pyplot as plt
import numpy as np
import statistics
import pandas as pd
import math
import random

""" 
    Helper functions
    (You can define your helper functions here.)
"""

def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    df = pd.read_csv(filename, sep=',', header = 0)
    return df


### HELPERS END ###


''' Functions to implement '''

# TODO: Implement this function!
def get_histogram(dataset, chosen_anime_id="199"):
    npa = np.array(dataset[chosen_anime_id])
    
    ls = list()
    for i in range(-1, 11):
        count = np.sum(npa == i)
        ls.append(count)
        
    plt.bar(list(range(-1, 11)), ls,
        width = 1)

    #plt.show()
    return ls


# TODO: Implement this function!
def get_dp_histogram(counts, epsilon: float):
    new_counts = []
    sens = 2
    
    for idx, i in enumerate(counts):
        c = i + np.random.laplace(0, sens / epsilon)
        new_counts.append(c)
        
    plt.bar(list(range(-1, 11)), new_counts,
        width = 1)
    plt.title(f"Plot for epsilon = {str(epsilon)}")

    #plt.show()
        
    return new_counts
        

# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    sumit = 0
    for idx, i in enumerate(actual_hist):
        sumit += abs(noisy_hist[idx] - i)
    return sumit / len(actual_hist)


# TODO: Implement this function!
def calculate_mean_squared_error(actual_hist, noisy_hist):
    sumit = 0
    for idx, i in enumerate(actual_hist):
        sumit += ((noisy_hist[idx] - i) ** 2)
    return sumit / len(actual_hist)

"""**** AVERAGE ERROR ****
eps =  0.0001  error =  19268.357591499855
eps =  0.001  error =  2155.3406170449643
eps =  0.005  error =  417.03793770586634
eps =  0.01  error =  204.250444683325
eps =  0.05  error =  41.81246589463069
eps =  0.1  error =  20.933133109411113
eps =  1.0  error =  2.120396410678489
**** MEAN SQUARED ERROR ****
eps =  0.0001  error =  700299504.7501767
eps =  0.001  error =  9219271.965513766
eps =  0.005  error =  335584.66276952496
eps =  0.01  error =  88457.61628669249
eps =  0.05  error =  3519.265155341103
eps =  0.1  error =  822.5478113626071
eps =  1.0  error =  8.725651756991018
**** EXPONENTIAL EXPERIMENT RESULTS ****

**** EXPONENTIAL EXPERIMENT RESULTS ****
1535
eps =  0.001  accuracy =  0.101
eps =  0.005  accuracy =  0.186
eps =  0.01  accuracy =  0.358
eps =  0.03  accuracy =  0.864
eps =  0.05  accuracy =  0.968
eps =  0.1  accuracy =  0.999"""

# TODO: Implement this function!
def epsilon_experiment(counts, eps_values: list):
    final_avg_ls = []
    final_sq_ls = []
    
    for eps in eps_values:
        avg_ls = []
        sq_ls = []
        for i in range(40):
            changed_hist = get_dp_histogram(counts, eps)
            
            avge = calculate_average_error(counts, changed_hist)
            sqe = calculate_mean_squared_error(counts, changed_hist)
            
            avg_ls.append(avge)
            sq_ls.append(sqe)
            
        final_avg_ls.append(np.mean(avg_ls))
        final_sq_ls.append(np.mean(sq_ls))
        
    return final_avg_ls, final_sq_ls
        


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def most_10rated_exponential(dataset, epsilon):
    counts = dataset.isin([10,]).sum(axis=0)[1:]
    sens = 2
    exp_vals = np.exp((epsilon * counts) / (2*sens))
    probs = exp_vals / np.sum(exp_vals)
    return random.choices(list(probs.index), weights=probs, k=1)[0]


# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    final_avg_ls = []
    counts = dataset.isin([10,]).sum(axis=0)[1:]
    true_val = counts.index[np.argmax(counts)]
    
    for eps in eps_values:
        true_ct = 0
        for i in range(1000):
            perturb_val = most_10rated_exponential(dataset, eps)
            if true_val == perturb_val:
                true_ct += 1
                
        final_avg_ls.append(true_ct / 1000)
        
    return final_avg_ls


# FUNCTIONS TO IMPLEMENT END #

def main():
    filename = "anime-dp.csv"
    dataset = read_dataset(filename)

    counts = get_histogram(dataset)

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg, error_mse = epsilon_experiment(counts, eps_values)
    print("**** AVERAGE ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])
    print("**** MEAN SQUARED ERROR ****")
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_mse[i])


    print ("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    exponential_experiment_result = exponential_experiment(dataset, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])


if __name__ == "__main__":
    main()

