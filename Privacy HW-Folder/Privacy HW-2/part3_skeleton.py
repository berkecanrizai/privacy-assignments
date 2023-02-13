import math, random
import matplotlib.pyplot as plt

""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

""" Helpers """

import numpy as np
from copy import deepcopy


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):
    d = len(DOMAIN)
    p = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    
    if random.random() < p:
        return val
    else:
        new_l = [i for i in DOMAIN if i != val]
        return random.choice(new_l)


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):
    hist_ls = []

    d = len(DOMAIN)
    p = np.exp(epsilon) / (np.exp(epsilon) + d + 1)
    q = (1 - p) / (len(DOMAIN) - 1)


    perturbed_counts = [0] * len(DOMAIN)

    for val in perturbed_values:
        perturbed_counts[val - 1] += 1

    for idx, nv in enumerate(perturbed_counts):
        i_v = nv * p + ((len(perturbed_values) - nv) * q)

        c = (i_v - len(perturbed_values) * q) / (p - q)
        hist_ls.append(c)
        
    return hist_ls


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    
    """perturbed_data = [perturb_grr(val, epsilon) for val in dataset]
    
    est_ls = estimate_grr(perturbed_data, epsilon)
    
    sumit = 0
    for idx, i in enumerate(est_ls):
        sumit += abs(dataset[idx] - i)"""
        
        
    
    real_counts = [0] * len(DOMAIN)

    for val in dataset:
        real_counts[val - 1] += 1

    perturbed_data = [perturb_grr(val, epsilon) for val in dataset]

    est_ls = estimate_grr(perturbed_data, epsilon)

    sumit = 0
    for idx, i in enumerate(est_ls):
        sumit += abs(real_counts[idx] - i)
    return sumit / len(DOMAIN)


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):
    vec = [0] * len(DOMAIN)
    vec[val - 1] = 1
    return vec


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    e2 = np.exp(epsilon/2)
    p = e2 / (e2 + 1)
    #q = 1 / (e2 + 1)
    
    encode_ls = []
    for idx, i in enumerate(encoded_val):
        if random.random() < p:
            encode_ls.append(i)
        else:
            encode_ls.append(abs(i - 1))
    return encode_ls


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    hist_ls = []
    e2 = np.exp(epsilon/2)
    p = e2 / (e2 + 1)
    q = 1 / (e2 + 1)
    
    nv_list = np.sum(perturbed_values, axis=0)
    for idx, nv in enumerate(nv_list):
        i_v = nv * p + ((len(perturbed_values) - nv) * q)
            
        c = (i_v - len(perturbed_values) * q) / (p - q)
        hist_ls.append(c)
    return hist_ls


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    vec_ls = [encode_rappor(val) for val in dataset]

    copy_vec_ls = deepcopy(vec_ls)
    perturbed_data = [perturb_rappor(vec, epsilon) for vec in copy_vec_ls]

    est_ls = estimate_rappor(perturbed_data, epsilon)
    error = np.sum(np.abs(est_ls - np.sum(vec_ls, axis=0)))

    return error / len(DOMAIN)


# OUE

# TODO: Implement this function!
def encode_oue(val):
    vec = [0] * len(DOMAIN)
    vec[val - 1] = 1
    return vec


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):
    e2 = np.exp(epsilon)
    p1 = 0.5
    p0 = 1 / (e2 + 1)
    
    encode_ls = []
    for idx, i in enumerate(encoded_val):
        if i == 1:
            if random.random() < p1:
                encode_ls.append(i)
            else:
                encode_ls.append(abs(i - 1))
        else:
            if random.random() < p0:
                encode_ls.append(1)
            else:
                encode_ls.append(0)
        
    return encode_ls


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):
    ee = np.exp(epsilon)
    
    num_users = len(perturbed_values)
    estimate_ls = []
    
    for actuali in np.sum(perturbed_values, axis=0):
        ci = 2 * ((ee + 1) * actuali - num_users) / (ee - 1)
        estimate_ls.append(ci)
        
    return estimate_ls


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):
    vec_ls = [encode_oue(val) for val in dataset]

    copy_vec_ls = deepcopy(vec_ls)
    perturbed_data = [perturb_oue(vec, epsilon) for vec in copy_vec_ls]

    est_ls = estimate_oue(perturbed_data, epsilon)
    error = np.sum(np.abs(est_ls - np.sum(vec_ls, axis=0)))

    return error / len(DOMAIN)


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")

    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))


if __name__ == "__main__":
    main()

