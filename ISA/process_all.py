import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.manifold import TSNE
from Bio.Cluster import kcluster
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
import pandas as pd
from scipy.optimize import linear_sum_assignment

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18

def plot_change(s1,inp):
    plt.figure(figsize=(12,9))
    plt.plot(range(len(s1)),s1,'b')
    plt.xlabel('奇异值数量')
    plt.ylabel('奇异值大小')
    plt.title('奇异值变化图')
    # plt.xticks(np.arange(0,301,30))
    if inp == 'ch':
        plt.savefig('./Pic/ch_wc/Svd_plot.png')
    else:
        plt.savefig('./Pic/docs_wc/Svd_plot.png')

def plot_total_change(s1,inp):
    ss = np.cumsum(s1/np.sum(s1))
    plt.figure(figsize=(12,9))
    plt.plot(range(len(ss)),ss,'b')
    plt.xlabel('奇异值数量')
    plt.ylabel('部分奇异值和解释率')
    plt.title('部分奇异值和解释率变化图')
    # plt.xticks(np.arange(0,301,30))
    if inp == 'ch':
        plt.savefig('./Pic/ch_wc/Svd_total_plot.png')
    else:
        plt.savefig('./Pic/docs_wc/Svd_total_plot.png')

def latent_space(u,s,vh,k):
    # un = u[:,:k]
    s = s[:k]
    sn = np.diag(s)
    vhn = vh[:k,:]
    return np.dot(sn,vhn)

def transform_x(X):
    column_norms = np.linalg.norm(X, axis=0)
    column_norms[column_norms == 0] = 1
    Xn = X / column_norms
    Xn = Xn.transpose()
    return Xn

def clustering_kmeans(X):
    st = time.time()
    # kmeans = KMeans(n_clusters=6,init='k-means++',n_init=1000)  # 设置聚类中心的数量为6
    # kmeans.fit(Xn)
    # labels = kmeans.labels_
    clusterid, error, nfound = kcluster(X, 6, dist='u', npass=1000)
    et = time.time()
    t = et-st

    return clusterid,t

def clustering_gmm(X):
    st = time.time()
    gmm = GaussianMixture(n_components=6, random_state=0,n_init=100,max_iter=1000)
    gmm.fit(X)
    labels = gmm.predict(X)
    et = time.time()
    t = et-st

    return labels,t

def plot_cluster(X,labels,inp):
    num = X.shape[1]
    tsne = TSNE(n_components=2)
    decomposition_data = tsne.fit_transform(X)
    
    x = []
    y = []
    
    for i in decomposition_data:
        x.append(i[0])
        y.append(i[1])
    
    plt.figure(figsize=(12, 9))
    plt.axes()
    plt.scatter(x, y, c=labels, marker="o")
    plt.xticks(())
    plt.yticks(())
    if inp == 'ch':
        plt.savefig(f'./Pic/ch_wc/cluster_plot_{num}.png')
    else:
        plt.savefig(f'./Pic/docs_wc/cluster_plot_{num}.png')
    # plt.show()

def compute_purity(y_true,label):
    num = len(y_true)
    clusters = np.unique(label)
    cc = []
    for c in clusters:
        idx = np.where(label == c)[0]
        true_value = [y_true[i] for i in idx]
        # print(true_value)
        # print(np.bincount(true_value))
        cc.append(np.bincount(true_value).max())
    # print(cc)
    return np.sum(cc)/num

def compute_f_measure_new(y_true,label):
    clusters = np.unique(label)
    c_len = len(clusters)
    cross_mat = np.zeros((c_len,c_len))

    num = len(label)
    label_num = np.bincount(label)
    true_num = np.bincount(y_true)

    i = 0
    for c in clusters:
        idx = np.where(label == c)[0]
        # print(idx)
        true_value = [y_true[i] for i in idx]
        every_num = np.bincount(true_value,minlength=c_len)
        # print(every_num)
        cross_mat[i,:] = every_num
        i+=1
        
    tp = 0
    for i in range(c_len):
        for j in range(c_len):
            if cross_mat[i,j] >= 2:
                tp += cross_mat[i,j]*(cross_mat[i,j]-1)/2

    tp_fp =  np.sum([i*(i-1)/2 for i in label_num])
    fp = tp_fp -tp

    tp_fn = np.sum([i*(i-1)/2 for i in true_num])
    fn = tp_fn - tp

    all_4 = num*(num-1)/2

    tn = all_4 - tp - fp - fn
    
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f = 2*precision*recall/(precision+recall)
    ari = 2*(tp*tn-fn*fp)/((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn))
    return f,ari

 
def max_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64) 
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
 
    # print(w.max() - w)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

if __name__ == '__main__':

    inp = 'ch'
    path = './Mat/{}'.format(inp)+'_tfidf.npy'
    df = pd.read_csv(f'./Data/{inp}_prepared.csv')
    dic = {'C000010':0,
        'C000013':1,
        'C000014':2,
        'C000020':3,
        'C000022':4,
        'C000023':5}
    # dic = {'O1':0,'C8':1,'P1':2,'R3':3,'F7':4,'G8':5}
    tfidf_vec = np.load(path)
    # st1 = time.time()
    u1, s1, vh1 = np.linalg.svd(tfidf_vec, full_matrices=True)
    # et1 = time.time()
    # print(et1-st1)

    # plot
    # plot_change(s1,inp)
    # plot_total_change(s1,inp)
    k_list = np.arange(10,310,20)
    # k_list
    ls_list2 = []
    for k in k_list:
        ls_list2.append(latent_space(u1,s1,vh1,k))

    kmeans_t = []
    gmm_t = []

    purity, fl, aril, acc = [],[],[],[]
    s_list = np.cumsum(s1/np.sum(s1))

    info_list = []

    df['type'] = [dic[i] for i in df['class']]
    y_true = df['type'].tolist()
    y_true=np.array(y_true)

    for i in tqdm(ls_list2):
        i = transform_x(i)
        info_list.append(s_list[i.shape[1]-1])
        label,t = clustering_kmeans(i)
        kmeans_t.append(t)
        purity.append(compute_purity(y_true,label))
        f,ari = compute_f_measure_new(y_true,label)
        fl.append(f)
        aril.append(ari)
        # acc.append(max_acc(y_true,label))
    
    new_df = pd.DataFrame({'k':k_list,
                           'info_list':info_list,
                           't':kmeans_t,
                           'purity':purity,
                           'f_list':fl,
                           'ari_list':aril})

    new_df.to_csv(f'./Metrics/{inp}.csv',index=False)
    
    max_ind = [1,5]
    #max_ind = [4,11]
    for i in max_ind:
        X = transform_x(ls_list2[i])
        label,t = clustering_kmeans(X)
        plot_cluster(X,label,inp)
