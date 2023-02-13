with open('digitalcorp.txt') as f:
    lines = f.readlines()
    
import pandas as pd
import numpy as np
import hashlib

df = pd.read_table('digitalcorp.txt', sep=',')

passwords = list(pd.read_table('rockyou.txt', header=None)[0])

kdf = pd.read_table('keystreching-digitalcorp.txt', sep=',')

def crack_with_salt(salt):
    xi = ''
    exp_ls = []
    dict_2000 = {}

    for pw in passwords:
        xi = ''
        xi1 = ''
        xi2 = ''
        xi3 = ''

        exp_ls = []
        for i in range(2000): #TODO: change
            xi = bytes(xi + salt + pw, 'utf-8') #xi + pw + salt #pw + salt + xi #xi + salt + pw #xi + salt + pw
            #xi + pw + salt, xi + salt + pw, pw + salt + xi, 
            xi = hashlib.sha512(xi).hexdigest()
            exp_ls.append(xi)

            xi1 = bytes(pw + xi1 + salt, 'utf-8')
            xi1 = hashlib.sha512(xi1).hexdigest()
            exp_ls.append(xi1)

            #xi2 = bytes(xi2 + salt + pw, 'utf-8')
            #xi2 = hashlib.sha512(xi2).hexdigest()
            #exp_ls.append(xi2)

            xi3 = bytes(salt + xi3 + pw, 'utf-8')
            xi3 = hashlib.sha512(xi3).hexdigest()
            exp_ls.append(xi3)


        dict_2000[pw] = exp_ls
    
    crack_list_2000 = []
    key = ''

    for hashed in kdf.hash_outcome:
        for key in dict_2000:
            cur_ls = dict_2000[key]

            if hashed in cur_ls:
                crack_list_2000.append(key)
                return key
    return key

cracked_list_p3 = []

for salt in kdf.salt.iloc[:]:
    cr = crack_with_salt(salt)
    cracked_list_p3.append(cr)
    
kdf['cracked'] = cracked_list_p3

kdf.to_csv('part3.csv')