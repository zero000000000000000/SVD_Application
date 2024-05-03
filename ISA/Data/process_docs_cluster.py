import pandas as pd

search_words = ['O1','O4','P1','R3','F7','G8']

classes = []
docs = []

for i in search_words:
    df = pd.read_excel('./Data/docs_cluster.xlsx',sheet_name=i)
    classes.extend([i]*len(df))
    for i in range(len(df)):
        df.loc[i,'abstract'] = str(df.loc[i,'abstract'])[5:]
        docs.append(' '.join([df.loc[i,'title'],df.loc[i,'abstract']]))

new_df = pd.DataFrame({'classes':classes,'docs':docs})

new_df.to_csv('./Data/docs_cluster_process.csv',index=False)