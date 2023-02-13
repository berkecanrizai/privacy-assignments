with open('digitalcorp.txt') as f:
    lines = f.readlines()
    
import pandas as pd
import numpy as np
import hashlib

df = pd.read_table('digitalcorp.txt', sep=',')

passwords = list(pd.read_table('rockyou.txt', header=None)[0])

dc = {}
for st in passwords:
    inp = bytes(st, 'utf-8')
    dc[(hashlib.sha512(inp).hexdigest())] = st
    
cracked = []
for hashed_pw in df.hash_of_password:
    if hashed_pw in dc:
        cracked.append(dc[hashed_pw])
        
df['cracked'] = cracked

df.to_csv('part1_passwords.csv')

pd.DataFrame([dc.keys(), dc.values()]).T.to_csv('part1_dict.csv')