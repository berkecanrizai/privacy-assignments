##############################################################################
# This skeleton was created by Efehan Guner  (efehanguner21@ku.edu.tr)       #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
from enum import unique
import glob
import os
import sys
from copy import deepcopy
import numpy as np
import datetime

import pandas as pd

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)

##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file."""
    
    with open(DGH_file, newline = '') as games:                                                                                          
        game_reader = csv.reader(games, delimiter='\t')
        ls = [i for i in game_reader]
    parents = dict()
    for lis in ls:
        for j in lis:
            if j != '':
                parents[j] = getParent(ls, j)
    return parents
    


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)
    
    diff = 0
    for idx, original in enumerate(raw_dataset):
        modified = anonymized_dataset[idx]

        for k in original:
            
            #print(modified[k], original[k])
            diff += difference(DGHs, k, modified[k], original[k])
    return diff
    #TODO: complete this function.



def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
    DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    diff = 0
    for idx, original in enumerate(raw_dataset):
        modified = anonymized_dataset[idx]

        for k in original:
            if original[k] != modified[k]:
                score = (countChildren(DGHs, DGHs[k], modified[k]) - 1) / (countLeaves(DGHs, k) - 1)
                diff += score / (len(original) - 1)
    return diff
            

def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)    

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)
    
    #TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...
    
    df = pd.DataFrame(list(raw_dataset))
    
    df = df.sample(frac=1)
    
    levels = read_Levels(DGH_folder)
    
    for i in range(0, len(df), k):
        curdf = (df.iloc[i: i+k])
        randCol = list(df.sample(axis='columns').columns)[0]
        
        if 1 != len(curdf.drop_duplicates(subset=curdf.columns[:-2])):
            anonimized_df = anonimizeDf(curdf, randCol, DGHs, levels)

        anonimized_df.reset_index(inplace=True)
        del anonimized_df['level_0']
        ls = []
        for i in range(len(anonimized_df)):
            ls.append(anonimized_df.iloc[i].to_dict())
        clusters.append(ls)
        

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']
    #print("len anon", len(anonymized_dataset))
    write_dataset(anonymized_dataset, output_file)



def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    #np.random.shuffle(raw_dataset)
    df = pd.DataFrame(list(raw_dataset))
    
    sdf = df
    dists = len(sdf) * [np.inf]
    clusters = []
    marks = len(sdf) * [False]
    idx = selectRecord(marks)
    total_added = 0
    cur_cluster = []
    DGHs = read_DGHs(DGH_folder)

    while total_added < len(sdf) + 2*k - 1:

        added_count = 1
        dists = len(sdf) * [np.inf] #will update this for each
        idx = selectRecord(marks)
        
        if idx is None: # no other left, we iterated all
            break
            
        marks[idx] = True
        cur_cluster = []
        cur_cluster.append(idx)
        
        if idx is None:
            break
        total_added += 1
        for i in range(len(sdf)):
            if not marks[i]: # if not already chosen
                if idx != i: # dont add the same thing to list (itself)
                    dists[i] = single_LM(sdf.iloc[idx], sdf.iloc[i], DGHs)

        sort_dists = np.array(dists).argsort() # sort for closest ones

        added_count = 0
        #add_ls = []
        ct = 0
        while added_count < k and ct < len(sort_dists): # only add as much as needed and stop after list ends
            val = sort_dists[ct]
            if not marks[val]:
                added_count += 1
                marks[val] = True
                cur_cluster.append(val)
                total_added += 1
            ct += 1
            #print(next_added_idx)
            """marks[idx] = True
            marks[next_added_idx] = True
            idx = next_added_idx
            cur_cluster.append(idx)
            added_count += 1
            total_added += 1
            dists = len(sdf) * [np.inf]"""
        clusters.append(cur_cluster)
        
    last_cluster = []
    
    for i in range(len(marks)):
        if not marks[i]:
            last_cluster.append(i)
            
    if len(last_cluster) > 0:
        clusters.append(last_cluster)
    
    levels = read_Levels(DGH_folder)
    results = []
    c = 0
    for cluster in clusters:
        curdf = anonimizeDf(sdf.iloc[cluster], "col", DGHs, levels)
        curdf.reset_index(inplace=True)
        
        if 'level_0' in curdf.columns:
            del curdf['level_0']

        ls = []
        for i in range(len(curdf)):
            ls.append(curdf.iloc[i].to_dict())
        results.append(ls)
    anonymized_dataset = [None] * total_added
    #print('len anon', len(anonymized_dataset))

    for cluster in results:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']
    
    # Finally, write dataset to a file
    write_dataset(anonymized_dataset, output_file)



def bottomup_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
    output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    
    all_level_dict = read_Levels(DGH_folder)
    
    sdf = pd.DataFrame(list(raw_dataset))
    ret_solutions = getPossibleBottomUpSolutions(sdf, all_level_dict, k, DGHs)
    
    lm_costs = []
    for sol in ret_solutions:
        lm_costs.append(genCost(sdf, generalizeColumn(sdf, sol, all_level_dict, DGHs), DGHs))

    final_version = generalizeColumn(sdf, ret_solutions[np.argmin(lm_costs)], all_level_dict, DGHs)
    
    clusters = []
    
    final_version.reset_index(inplace=True)
    #del final_version['level_0']
    ls = []
    for i in range(len(final_version)):
        ls.append(final_version.iloc[i].to_dict())
    clusters.append(ls)
    
        
    anonymized_dataset = [None] * len(sdf)

    for cluster in clusters:        #restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    # Finally, write dataset to a file
    write_dataset(anonymized_dataset, output_file)
    
def anonimizeDf(df, col, DGHs, levels):
    ls = []

    import warnings
    warnings.filterwarnings("ignore")
    for col in df.columns[:-2]:
        if not len(df[col].unique()) == 1: ## they are not same
            level_dc = levels[col]
            while len(df[col].unique()) != 1:
                level_df = df[col].map(levels[col])
                ch_idx = level_df.argmax()

                df[col].iloc[ch_idx] = DGHs[col][df[col].iloc[ch_idx]]
                #print((df[col].iloc[ch_idx]))
                
    return df
            
            
            
            
    for i, d in df.groupby(list(df.columns[:-1])):
        #d[col] = DGHs[col][d[col].iloc[0]]
        d[col] = DGHs[col][d[col].iloc[0]]
        ls.append(d)
    return pd.concat(ls)

def single_LM(original, modified, DGHs) -> float:
    qi_ls = list(original.index)[:-1]
    
    original = original.T.to_dict()
    modified = modified.T.to_dict()
    diff = 0

    for k in qi_ls:
        if original[k] != modified[k]:
            score = (countChildren(DGHs, DGHs[k], modified[k]) - 1) / (countLeaves(DGHs, k) - 1)
            diff += score / (len(original) - 1)
            
    return diff

def selectRecord(marks):
    for idx, m in enumerate(marks):
        if not m:
            return idx
    
    return None


def difference(DGHs, k, mdf, org):
    cost = 0
    ne = org
    while ne != mdf:
        ne = DGHs[k][ne]
        cost += 1
    return cost

def getParent0(dc, name):
    return dc[name]
    
        
def getParent(ls, name):
    placeidx = 0
    y = 0
    for i in range(len(ls)):
        littlelist = ls[i]
        if name in littlelist:
            placeidx = i
            y = littlelist.index(name)
    
    for i in range(placeidx, -1, -1):
        littlelist = ls[i]
        if littlelist[y - 1] != '':
            return littlelist[y - 1]
        
def read_DGH(DGH_file):
    with open(DGH_file, newline = '') as games:                                                                                          
        game_reader = csv.reader(games, delimiter='\t')
        ls = [i for i in game_reader]
    parents = dict()
    for lis in ls:
        for j in lis:
            if j != '':
                parents[j] = getParent(ls, j)
    return parents

def countLeaves(DGHs, k):
    counter = {}
    dc = DGHs[k]
    for k in dc:
        val = dc[k]
        counter[val] = counter.get(val, 0) + 1
    
    return len(set(dc) - set(counter))

def isLeaf(nd, dc):
    return not (nd in [v for k, v in dc.items()])

def countChildren(DGHs, dc, name):
    if isLeaf(name, dc):
        return 1
    res = 0
    chls = []
    for k in dc:
        val = dc[k]
        if val == name and name != k:
            chls.append(k)
            
    tt = 0
    for i in chls:
        tt += countChildren(DGHs, dc, i)

    return res + tt

def getK(df):
    mini = np.inf
    for i, d in df.groupby(list(df.columns[:-1])):
        #print(len(d))
        if mini > len(d):
            mini = len(d)
    #print("kanon", mini)
    return mini

def getLevelDict(DGH_file):
    with open(DGH_file, newline = '') as games:                                                                                          
        game_reader = csv.reader(games, delimiter='\t')
        ls = [i for i in game_reader]
    level_dict = {}
    for l in ls:
        count = 0
        for c in l:
            if c == '':
                count += 1
            else:
                level_dict[c] = count

    return level_dict

def read_Levels(DGH_folder: str) -> dict:
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = getLevelDict(DGH_file)
        
    return DGHs




def genCost(sdf, changed, DGHs):
    s = 0
    for i in range(len(sdf)):
        a = changed.iloc[i]
        b = sdf.iloc[i]
        s += single_LM(b, a, DGHs)
        
    return s

def levelsEnough(current, target):
    for idx, c in enumerate(current):
        if c > target[idx]:
            return False
    return True


def getCurrentLevelsDF(df, levels):
    res = []
    for ix, col in enumerate(levels):
        mapped_levels_df = df[col].map(levels[col])
        res.append(mapped_levels_df.max())
    return res

def generalizeColumn(sdf, target_arr, levels, DGHs):
    xdf = deepcopy(sdf)
    while not levelsEnough(getCurrentLevelsDF(xdf, levels), list(target_arr)):
        for ix, col in enumerate(levels):
            mapped_levels_df = xdf[col].map(levels[col])
            change_indexes = mapped_levels_df.index[mapped_levels_df > target_arr[ix]].tolist()
            xdf[col].loc[change_indexes] = xdf.loc[change_indexes][col].map(DGHs[col])
            
    return xdf


class Tree:
    def __init__(self, val):
        self.val = val
        self.childrens = []

levels_set = []
def populateNode(node):
    for b in branchFromSingle(node.val):
        b = list(b)
        if b not in levels_set:
            levels_set.append(b)
            node.childrens.append(Tree(b))


def addLevel(root):
    if root.childrens:
        for ch in root.childrens:
            levels_set.append(ch)
            addLevel(ch)
    else:
        populateNode(root)

        
        
levels_set = []
def populateNode(node):
    for b in branchFromSingle(node.val):
        b = list(b)
        if b not in levels_set:
            levels_set.append(b)
            node.childrens.append(Tree(b))
            
def addLevel(root):
    if root.childrens:
        for ch in root.childrens:
            levels_set.append(ch)
            addLevel(ch)
    else:
        populateNode(root)
        
def populateNodeWithLS(node, ls):
    for x in ls:
        for b in branchFromSingle(x):
            b = list(b)
            if b not in levels_set:
                levels_set.append(b)
                node.childrens.append(Tree(b))

def getBottoms(root):
    if root.childrens:
        for ch in root.childrens:
            getBottoms(ch)
    else:
        l = list(root.val)
        #print('in getbottoms', l)
        #print(bots)
        if not l in bots:
            #print('in getbottoms appending', l)
            bots.append(l)


def getNexts(bots):
    res = []
    for i in bots:
        for x in branchFromSingle(i):
            y = x - 1
            y[y < 0] = 0
            x = list(x)
            y = list(y)
            if x not in res and x not in all_tried:
               	res.append(x)
                all_tried.append(x)
                
            if y not in res and y not in all_tried:
               	res.append(y)
                all_tried.append(y)
        
    return res

def getValidCombinations(ls, df, all_level_dict, K, DGHs):
    res = []
    for lv in ls:
        currGeneralization = generalizeColumn(df, lv, all_level_dict, DGHs)
        curK = getK(currGeneralization)
        #print(lv)
        #print(curK)
        if curK >= K:
            res.append(lv)
            
    return res


def branchFromSingle(ls):
    levels = []
    ls = np.array(ls)
    for idx, num in enumerate(ls):
        addNum = ls[idx] - 1
        addNum = max(addNum, 0)
        levels.append(np.concatenate([ls[:idx], [addNum], ls[idx + 1:]]))
        
    return levels


all_tried = []
global bots
bots = []
levels_set = []

def getPossibleBottomUpSolutions(sdf, all_level_dict, K, DGHs):
    max_levels = getCurrentLevelsDF(sdf, all_level_dict)
    #print(max_levels)
    llength = len(max_levels)
    global bots
    root = Tree(max_levels)
    populateNode(root)
    #bots = []
    for i in range(4):
        addLevel(root)
        bots = []
        #print(bots)
        getBottoms(root)
        ret_ls = getValidCombinations(bots, sdf, all_level_dict, K, DGHs)

        if ret_ls:
            return ret_ls
    
    #print('For ended ', len(bots))
    #bots = []
    getBottoms(root)

    any_comb = []
    for i in range(llength - 1):
        any_comb.append(0)

    while True:
        ret_ls = getValidCombinations(bots, sdf, all_level_dict, K, DGHs)

        if ret_ls:
            return ret_ls

        bots = getNexts(bots)
        #print(len(bots))

        if any_comb in bots:
            return any_comb


# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now() ##
print(start_time) ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now() ##
print(end_time) ##
print(end_time - start_time)  ##

# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300 5