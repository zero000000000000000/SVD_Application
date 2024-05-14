import pandas as pd
import numpy as np
import os

path = './Data/ch/Reduced'
dic = {'C000007':'Automobile','C000008':'Finance','C000010':'Internet Technology',
       'C000013':'Health','C000014':'Sports','C000016':'Travel','C000020':'Education',
       'C000022':'Recruitment','C000023':'Culture','C000024':'Military'}
type_list = os.listdir(path)

choose_list = np.random.choice(type_list, size = 6, replace = False)
lis = []
f_list = []
for j in choose_list:
    file_list =  os.listdir(path+'/'+j)
    choose_file_list = np.random.choice(file_list, size = 50, replace = False)
    lis.extend([j]*50)
    for k in choose_file_list:
        with open(path+'/'+j+'/'+k,'r',encoding='gb18030') as f:
            s = f.read()
            f_list.append(s)

df = pd.DataFrame({'class':lis,'text':f_list})
df.to_csv('./Data/ch_for_cluster.csv',index=False)