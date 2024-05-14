import pandas as pd
import re
import numpy as np

def contains_chinese(s):
    return bool(re.search(r'[\u4e00-\u9fff]', s))


search_words = ['O1','P1','R3','F7','G8','C8']

# classes = []
# docs = []

# for w in search_words:
#     df = pd.read_excel('./Data/ch_for_cluster.xlsx',sheet_name=w)
    
#     for i in range(len(df)):
#         df.loc[i,'abstract'] = str(df.loc[i,'abstract'])[5:]
#         if contains_chinese(df.loc[i,'abstract']):
#             docs.append(' '.join([str(df.loc[i,'title']),str(df.loc[i,'keywords']),str(df.loc[i,'abstract'])]))
#             classes.append(w)

# new_df = pd.DataFrame({'class':classes,'text':docs})

# new_df.to_csv('./Data/docs_cluster_process.csv',index=False)

df = pd.read_csv('./Data/docs_cluster_process.csv')
for key,df1 in df.groupby('class'):
    print(key,len(df1))

del_list = []
for i in range(len(df)):
    if '下载App查看全文' in df.loc[i,'text']:
        del_list.append(i)


df.drop(del_list,axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)

doc_lis = []
for j in search_words:
    if j != 'P1':
        doc_lis.extend(np.random.choice(df[df['class']==j].index.tolist(),50,replace=False))
    else:
        doc_lis.extend(df[df['class']==j].index.tolist())

df1 = df.iloc[doc_lis,:].reset_index(drop=True)



df_p1 = pd.read_excel('./Data/docs_for_cluster_P1_add.xlsx',sheet_name='P1')
p1_docs = []
p1_classes = []
for i in range(len(df_p1)):
    df_p1.loc[i,'abstract'] = str(df_p1.loc[i,'abstract'])[5:]
    if contains_chinese(df_p1.loc[i,'abstract']):
        p1_docs.append(' '.join([str(df_p1.loc[i,'title']),str(df_p1.loc[i,'keywords']),str(df_p1.loc[i,'abstract'])]))
        p1_classes.append('P1')

p1_add = np.random.choice(p1_docs,size = 6, replace=False)

p1_df = pd.DataFrame({'class':['P1']*6,'text':p1_add})

df_res = pd.concat([df1,p1_df],axis=0,ignore_index=True)

for i,df_new1 in df_res.groupby('class'):
    print(i,len(df_new1))

df_res.to_csv('./Data/docs_for_cluster_new.csv',index=False)