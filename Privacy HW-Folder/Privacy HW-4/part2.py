with open('digitalcorp.txt') as f:
    lines = f.readlines()
    
import pandas as pd
import numpy as np
import hashlib

passwords = list(pd.read_table('rockyou.txt', header=None)[0])

sdf = pd.read_table('salty-digitalcorp.txt', sep=',')

dcs = {}
for i in range(len(sdf)):
    salt = (sdf.iloc[i].salt)
    #pw = sdf.iloc[i].hash_outcome
    
    for st in passwords:
        inp = bytes(st + salt, 'utf-8')
        inp2 = bytes(salt + st, 'utf-8')
        dcs[(hashlib.sha512(inp).hexdigest())] = st
        dcs[(hashlib.sha512(inp2).hexdigest())] = st
        
        
cracked_salted = []
for hashed_pw in sdf.hash_outcome:
    if hashed_pw in dcs:
        cracked_salted.append(dcs[hashed_pw])
        

sdf['cracked'] = cracked_salted

sdf.to_csv('part2.csv')