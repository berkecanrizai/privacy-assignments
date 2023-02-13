import sys
import random

import numpy as np
import pandas as pd
import copy

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def model_selector(model_type):
    if model_type == 'DT':
        model = DecisionTreeClassifier(max_depth=5, random_state=0)
    elif model_type == 'LR':
        model = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    elif model_type == 'SVC':
        model = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
    return model
    
    

###############################################################################
############################### Label Flipping ################################
###############################################################################

import warnings
warnings.filterwarnings("ignore")

def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n):
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    model = model_selector(model_type)
    accuracies = []
    select_count = int(n * len(X_train))
    
    for i in range(100):
        original_labels = copy.deepcopy(y_train)
        selected = np.random.choice(len(original_labels), select_count, replace=False)
        original_labels[selected] -= 1
        changed_labels = np.abs(original_labels)
        model.fit(X_train, changed_labels)
        
        pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, pred))
        
    return np.mean(accuracies)

###############################################################################
############################## Inference ########################################
###############################################################################

def inference_attack(trained_model, samples, t):
    # TODO: You need to implement this function!
    tp = 0
    fn = 0
    
    for sample in samples:
        probs = trained_model.predict_proba(sample.reshape(1, -1))
        
        if np.max(probs) >= t:
            tp += 1
        else:
            fn += 1
        
    return tp/(tp + fn)    

###############################################################################
################################## Backdoor ###################################
###############################################################################

def backdoor_attack(X_train, y_train, model_type, num_samples):    
    # TODO: You need to implement this function!
    # You may want to use copy.deepcopy() if you will modify data
    model = model_selector(model_type)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X_train, y_train)
    model.fit(Xtrain, ytrain)
    pred = model.predict(Xtest)
    base_acc = accuracy_score(ytest, pred)
    
    if model_type == 'SVC':
        train_add = model.support_vectors_[np.random.choice(len(model.support_vectors_), num_samples, replace=False)]
        
        if num_samples != 0:
            
            if num_samples != 1:
                labels_add = model.predict(train_add)
                labels_add -= 1
                labels_add = np.abs(labels_add)
                model.fit(np.vstack([Xtrain, train_add]), np.hstack([ytrain, labels_add]))
            else:
                labels_add = model.predict(train_add.reshape(1, -1))
                labels_add -= 1
                labels_add = np.abs(labels_add)
                model.fit(np.vstack([Xtrain, train_add]), np.hstack([ytrain, labels_add]))
        
        
    elif model_type == 'DT':
        if num_samples != 0:
            train_add = np.random.rand(num_samples, Xtrain.shape[1]) * (Xtrain.max(axis=0) - Xtrain.min(axis=0)) + Xtrain.min(axis=0)
            # create vectors for each variable between min and max ie first feature between 20-40 etc.
            pred = model.predict(train_add)
            #labels_add = np.repeat(0, len(train_add))
            labels_add = np.abs(pred - 1)
            model.fit(np.vstack([Xtrain, train_add]), np.hstack([ytrain, labels_add]))
        
    elif model_type == 'LR':
        if num_samples > 0:
            train_add = np.random.rand(num_samples, Xtrain.shape[1]) * (Xtrain.max(axis=0) - Xtrain.min(axis=0)) + Xtrain.min(axis=0)
            ch_idx = np.argmax(model.coef_)
            train_1 = np.mean(Xtrain[ytrain == 1][:, ch_idx])
            #train_0 = np.mean(Xtrain[ytrain == 0][:, ch_idx])

            n1 = np.random.normal(train_1, (np.var(Xtrain[ytrain == 1][:, ch_idx]))**1/2, size=num_samples)
            train_add[:, ch_idx] = n1
            
            pred = model.predict(train_add)
            labels_add = np.abs(pred - 1) #reverse labels
            model.fit(np.vstack([Xtrain, train_add]), np.hstack([ytrain, labels_add]))
        
    prednew = model.predict(Xtest)
    new_acc = accuracy_score(ytest, prednew)
    
    
    #abs(base_acc - new_acc) / 0.5
    return np.mean(prednew != pred)



###############################################################################
############################## Evasion ########################################
###############################################################################
def perturb(arr, index, val):
    ls = copy.deepcopy(arr)
    ls[index] *= val
    return ls

def evade_model(trained_model, actual_example):
    def choose_side(modified_example, idx): # will use in two places
            orig = copy.deepcopy(modified_example)
            modified_example[idx] = modified_example[idx] * 0.9
            prob0 = np.max(trained_model.predict_proba(modified_example.reshape(1, -1)))
            modified_example[idx] = modified_example[idx] * 1.2
            prob1 = np.max(trained_model.predict_proba(modified_example.reshape(1, -1)))
            
            multi = 1.1
            if prob0 < prob1:
                multi = 0.9
            modified_example = orig
            return multi
        
    real_example = copy.deepcopy(actual_example)
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    #modified_example2 = copy.deepcopy(actual_example)
    
    if str(type(trained_model)) == "<class 'sklearn.tree._classes.DecisionTreeClassifier'>":
        orig = copy.deepcopy(modified_example)
        """modified_example[trained_model.tree_.feature[0]] = (trained_model.tree_.threshold - 0.01)[trained_model.tree_.feature[0]]
        if trained_model.predict(modified_example.reshape(1, -1)) == actual_class:
            modified_example[trained_model.tree_.feature[0]] = (trained_model.tree_.threshold + 0.01)[trained_model.tree_.feature[0]]
            
        if trained_model.predict(modified_example.reshape(1, -1)) == actual_class:
            return modified_example"""
        
        
        clf = trained_model

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        feature = clf.tree_.feature
        threshold = clf.tree_.threshold

        node_indicator = clf.decision_path(modified_example.reshape(1, -1).astype('float32'))
        leaf_id = clf.apply(modified_example.reshape(1, -1).astype('float32'))

        sample_id = 0
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]

        for node_id in node_index:
            if leaf_id[sample_id] == node_id:
                continue
            
            if modified_example[feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
                modified_example[feature[node_id]] = threshold[node_id] + 0.000000001
            else:
                threshold_sign = ">"
                modified_example[feature[node_id]] = threshold[node_id] - 0.000000001
        
        if trained_model.predict(modified_example.reshape(1, -1)) != actual_class:
            #modified_example[feature[node_id]] = threshold[node_id] - 1
            return modified_example
        
        ##### Decision treede burada bitmesi lazım, doğrudan tüm yolda gezip kararları değiştirdim
        #Buradan sonraki kod eski kalma perturbation ama bıraktım yinede, diğer class modeller aşağıda
        # CHANGED ALL THE VALUES ON THE DECISIONS
        
        
        important_features = []
        #important_features = np.argwhere(trained_model.feature_importances_ >= 0.04).flatten()
        
        for i in np.argwhere(trained_model.feature_importances_ >= 0.04).flatten():
            if i in np.argsort(trained_model.feature_importances_)[::-1]:
                important_features.append(i)
        
        if trained_model.predict(modified_example.reshape(1, -1)) == actual_class:
            idx = important_features[0]
            modified_example[idx] = modified_example[idx] * 0.9
            prob0 = np.max(trained_model.predict_proba(modified_example.reshape(1, -1)))
            modified_example[idx] = modified_example[idx] * 1.2
            prob1 = np.max(trained_model.predict_proba(modified_example.reshape(1, -1)))
            
            multi = 1.1
            if prob0 < prob1:
                multi = 0.9
            
            k = 0
            
            while trained_model.predict(modified_example.reshape(1, -1)) == actual_class:
                if np.max(real_example) * 4 <= np.max(modified_example) and np.argmax(modified_example) in important_features:
                    important_features.remove(np.argmax(modified_example))
                    k = k%len(important_features)
                idx = important_features[k]
                multi = choose_side(modified_example, idx) # up or down for scale
                modified_example[idx] = modified_example[idx] * multi
                k += 1
                k = k%len(important_features)
                
                
        return modified_example
    
    if str(type(trained_model)) == "<class 'sklearn.linear_model._logistic.LogisticRegression'>":
        
        def perturb_idx(idx, num):
            for i in range(10):
                modified_example[idx] *= num
                
        
                
        important_features = np.argwhere(np.abs(trained_model.coef_) >= 0.1)[:, 1] # first col for indices
        idx = np.argmax(np.abs(trained_model.coef_))
        if trained_model.predict(modified_example.reshape(1, -1)) == actual_class:
            #idx = important_features[0]
            #print(idx)
            modified_example[idx] = modified_example[idx] * 0.9
            prob0 = np.max(trained_model.predict_proba(modified_example.reshape(1, -1)))
            modified_example[idx] = modified_example[idx] * 1.2
            prob1 = np.max(trained_model.predict_proba(modified_example.reshape(1, -1)))
            
            multi = 1.1
            if prob0 < prob1:
                multi = 0.9
            
            k = 0
            while trained_model.predict(modified_example.reshape(1, -1)) == actual_class:
                idx = important_features[k]
                multi = choose_side(modified_example, idx) # up or down for scale
                
                modified_example[idx] = modified_example[idx] * multi
                k += 1
                k = k % len(important_features)
        return modified_example
    
    if str(type(trained_model)) == "<class 'sklearn.svm._classes.SVC'>":
        label = abs(actual_class - 1)
        
        supports = trained_model.support_vectors_
        options = supports[np.argwhere(trained_model.predict(supports) == label).flatten()]
        closest = options[np.argmin(np.sum(np.abs(options - modified_example), axis=1))]
        return closest
    
    
    ### CODE IS NOT SUPPOSED TO REACH HERE, RANDOM PERTURB EKLEDIM
    actual_example = actual_example.reshape(1, -1)
    pred_class = trained_model.predict(actual_example)[0]
    original_prob = np.max(trained_model.predict_proba(actual_example))
    
    effect_ls = []
    
    for i in range(len(actual_example[0])):
        perturbed = perturb(modified_example, i, 0.9).reshape(1, -1)
        predict = trained_model.predict(perturbed)
        eff = (abs(pred_class - predict), np.max(trained_model.predict_proba(perturbed)), perturbed, i, 0.9)
        effect_ls.append(eff)
        
    for i in range(len(actual_example[0])):
        perturbed = perturb(modified_example, i, 1.1).reshape(1, -1)
        predict = trained_model.predict(perturbed)
        eff = (abs(pred_class - predict), np.max(trained_model.predict_proba(perturbed)), perturbed, i, 1.1)
        effect_ls.append(eff)
        
    max_prob = 0
    effective = 0
    multip = 0
    idx_ls = []
    for c, p, changed, idx, multi in effect_ls:
        if c == 1:
            return changed.flatten()
        if p > max_prob:
            max_prob = p
            effective = idx
            idx_ls.append(idx)
            multip = multi
            
            
    for idx in range(len(actual_example[0])):
        #print(idx)
        if idx not in idx_ls:
            idx_ls.append(idx)
            
    

    idx_ls = np.array(idx_ls)
    #idx_ls = np.sort(idx_ls)[::-1]
    actual_class = trained_model.predict(actual_example)[0]
    pred_class = trained_model.predict(actual_example)[0]
    ct = 0
    
    circular = False
    while pred_class == actual_class:
        idx = idx_ls[ct % len(idx_ls)]
        if idx < len(modified_example) and not circular:
            for i in range(10):
                modified_example[idx] = modified_example[idx] * multip
                modified_example = modified_example.reshape(1, -1)
                modified_example = np.nan_to_num(modified_example, posinf=1000, neginf=-1000)
                pred_class = trained_model.predict(modified_example)[0]

                if pred_class != actual_class:
                    break
        ct += 1
        
        if ct >= 3 * len(modified_example):
            circular = True
            
        if circular:
            modified_example = np.random.normal(actual_example[0].mean(axis=0), actual_example[0].std(axis=0), size=actual_example[0].shape)
            pred_class = trained_model.predict(modified_example.reshape(1, -1))[0]
            #modified_example = np.random.rand(1, Xtrain.shape[1]) * (Xtrain.max(axis=0) - Xtrain.min(axis=0)) + Xtrain.min(axis=0)
    #print(modified_example.reshape(actual_example.shape)[0])
    #print(actual_example.shape)
    
    return modified_example.flatten()
    #print(actual_example)
    #return modified_example.reshape(actual_example.shape).flatten()

def calc_perturbation(actual_example, adversarial_example):
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i]-adversarial_example[i])
        return tot/len(actual_example)

###############################################################################
############################## Transferability ################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    # TODO: You need to implement this function!
    print("Here, you need to conduct some experiments related to transferability and print their results...")

    dt_perturbeds = []
    lr_perturbeds = []
    svc_perturbeds = []
    
    for example in actual_examples:
        dt_perturbeds.append(evade_model(DTmodel, example))
        
    for example in actual_examples:
        lr_perturbeds.append(evade_model(LRmodel, example))
        
    for example in actual_examples:
        svc_perturbeds.append(evade_model(SVCmodel, example))
        
    dt_perturbeds = np.array(dt_perturbeds)
    lr_perturbeds = np.array(lr_perturbeds)
    svc_perturbeds = np.array(svc_perturbeds)
        
    dt_lr = np.mean(DTmodel.predict(actual_examples) == DTmodel.predict(lr_perturbeds))
    dt_svc = np.mean(DTmodel.predict(actual_examples) == DTmodel.predict(svc_perturbeds))
    
    lr_dt = np.mean(LRmodel.predict(actual_examples) == LRmodel.predict(dt_perturbeds))
    lr_svc= np.mean(LRmodel.predict(actual_examples) == LRmodel.predict(svc_perturbeds))
    
    svc_dt = np.mean(SVCmodel.predict(actual_examples) == SVCmodel.predict(dt_perturbeds))
    svc_lr = np.mean(SVCmodel.predict(actual_examples) == SVCmodel.predict(lr_perturbeds))
    
    print('Transferabilities')
    
    print('DT - LR', 1-dt_lr)
    print('DT - SVC', 1-dt_svc)
    
    print('LR - DT', 1-lr_dt)
    print('LR - SVC', 1-lr_svc)
    
    print('SVC - DT', 1-svc_dt)
    print('SVC - LR', 1-svc_lr)
    
    dt_lr = np.sum(DTmodel.predict(actual_examples) == DTmodel.predict(lr_perturbeds))
    dt_svc = np.sum(DTmodel.predict(actual_examples) == DTmodel.predict(svc_perturbeds))
    
    lr_dt = np.sum(LRmodel.predict(actual_examples) == LRmodel.predict(dt_perturbeds))
    lr_svc= np.sum(LRmodel.predict(actual_examples) == LRmodel.predict(svc_perturbeds))
    
    svc_dt = np.sum(SVCmodel.predict(actual_examples) == SVCmodel.predict(dt_perturbeds))
    svc_lr = np.sum(SVCmodel.predict(actual_examples) == SVCmodel.predict(lr_perturbeds))
    
    print('DT - LR', dt_lr)
    print('DT - SVC', dt_svc)
    
    print('LR - DT', lr_dt)
    print('LR - SVC', lr_svc)
    
    print('SVC - DT', svc_dt)
    print('SVC - LR', svc_lr)
    
###############################################################################
########################## Model Stealing #####################################
###############################################################################

def steal_model(remote_model, model_type, examples):
    # TODO: You need to implement this function!
    # This function should return the STOLEN model, but currently it returns the remote model
    # You should change the return value once you have implemented your model stealing attack
    
    stolen_model = model_selector(model_type)
    
    X_train = examples
    y_train = []
    
    y_train = remote_model.predict(examples)
        
    stolen_model.fit(X_train, y_train)
    return stolen_model
    

###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ## 
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##

def main():
    data_filename = "forest_fires.csv"
    features = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI"]
    
    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    df["DC"] = df["DC"].astype('float64')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)    
    X = df[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 
    

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))
    
    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=1000)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))
    
    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0,probability=True)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    

    # Label flipping attack executions:
    model_types = ["DT", "LR", "SVC"]
    n_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for n in n_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, n)
            print("Accuracy of poisoned", model_type, str(n), ":", acc)
    
    # Inference attacks:
    samples = X_train[0:100]
    t_values = [0.99,0.98,0.96,0.8,0.7,0.5]
    for t in t_values:
        print("Recall of inference attack", str(t), ":", inference_attack(mySVC,samples,t))
    
    # Backdoor attack executions:
    counts = [0, 1, 3, 5, 10]
    for model_type in model_types:
        for num_samples in counts:
            success_rate = backdoor_attack(X_train, y_train, model_type, num_samples)
            print("Success rate of backdoor:", success_rate, "model_type:", model_type, "num_samples:", num_samples)
    
    #Evasion attack executions:
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"] 
    num_examples = 40
    for a,trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a] , ":" , total_perturb/num_examples)

    
    # Transferability of evasion attacks:
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])
    
    # Model stealing:
    budgets = [8, 12, 16, 20, 24]
    for n in budgets:
        print("******************************")
        print("Number of queries used in model stealing attack:", n)
        stolen_DT = steal_model(myDEC, "DT", X_test[0:n])
        stolen_predict = stolen_DT.predict(X_test)
        print('Accuracy of stolen DT: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_LR = steal_model(myLR, "LR", X_test[0:n])
        stolen_predict = stolen_LR.predict(X_test)
        print('Accuracy of stolen LR: ' + str(accuracy_score(y_test, stolen_predict)))
        stolen_SVC = steal_model(mySVC, "SVC", X_test[0:n])
        stolen_predict = stolen_SVC.predict(X_test)
        print('Accuracy of stolen SVC: ' + str(accuracy_score(y_test, stolen_predict)))
    

if __name__ == "__main__":
    main()
