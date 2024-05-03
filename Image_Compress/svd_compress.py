from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import pandas as pd
import argparse

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18

def check_img_type(image_array):
    '''
    Check whether the img is grey or color
    '''
    if len(image_array.shape) == 2:
        return 'Grey'
    else:
        return 'Color'

def svd_full(image_array, img_type):
    '''
    Conduct the full svd and extract the S matrix
    '''
    if img_type == 'Grey':
        U, S, V = np.linalg.svd(image_array, full_matrices=False)
    else:
        U1, S1, V1 = np.linalg.svd(image_array[:,:,0], full_matrices=False)
        U2, S2, V2 = np.linalg.svd(image_array[:,:,1], full_matrices=False)
        U3, S3, V3 = np.linalg.svd(image_array[:,:,2], full_matrices=False)
        S = [S1,S2,S3]

    return S

def plot_sv_curve(S,img_type):
    '''
    plot the curve of singular value
    '''
    if img_type == 'Grey':
        sv_list = S.tolist()
        plt.figure(figsize=(12,9))
        n = len(sv_list)

        x = [i for i in range(1,n+1)]
        plt.plot(x, sv_list, 'r-')

        xx = np.arange(20,max(x),20).tolist()
        plt.xticks(xx)
        plt.xlabel('奇异值数量')
        plt.ylabel('奇异值大小')
        plt.title('奇异值变化图')

        plt.savefig('./Pic/{}/sv_curve.jpg'.format(img_type))
    else:
        num = 0
        for i in range(len(S)):
            num += 1
            sv_list = S[i].tolist()
            plt.figure(figsize=(12,9))
            n = len(sv_list)

            x = [i for i in range(1,n+1)]
            plt.plot(x, sv_list, 'r-')

            xx = np.arange(20,max(x),20).tolist()
            plt.xticks(xx)
            plt.xlabel('奇异值数量')
            plt.ylabel('奇异值大小')
            plt.title('奇异值变化图')
            plt.savefig('./Pic/{0}/sv_curve_{1}.jpg'.format(img_type,num))

def plot_info_curve(S,img_type):
    '''
    Plot infomation curve
    '''
    if img_type == 'Grey':
        S_rate = np.cumsum(S/np.sum(S))
        sv_list = S_rate.tolist()
        plt.figure(figsize=(12,9))
        n = len(sv_list)

        x = [i for i in range(1,n+1)]
        plt.plot(x, sv_list, 'r-')

        xx = np.arange(20,max(x),20).tolist()
        plt.xticks(xx)
        plt.xlabel('奇异值数量')
        plt.ylabel('部分奇异值和解释率')
        plt.title('部分奇异值和解释率变化图')

        plt.savefig('./Pic/{}/info_curve.jpg'.format(img_type))
    else:
        for i in range(len(S)):
            S_rate = np.cumsum(S[i]/np.sum(S[i]))
            sv_list = S_rate.tolist()
            plt.figure(figsize=(12,9))
            n = len(sv_list)

            x = [i for i in range(1,n+1)]
            plt.plot(x, sv_list, 'r-')

            xx = np.arange(20,max(x),20).tolist()
            plt.xticks(xx)
            plt.xlabel('奇异值数量')
            plt.ylabel('部分奇异值和解释率')
            plt.title('部分奇异值和解释率变化图')
            plt.savefig('./Pic/{0}/info_curve_{1}.jpg'.format(img_type,i+1))
    
def get_k_list(S):
    '''
    Get sv number k list
    '''
    if type(S) == list:
        Sp = S[0]
    else:
        Sp = S
    
    S_rate = np.cumsum(Sp/np.sum(Sp))
    k_max = S_rate[S_rate<0.95].size
    print(k_max)
    knum = k_max//10

    k_list = [1]
    k_list.extend(np.arange(knum,k_max,knum).tolist())
    return k_list

def normalize_image(image_array):
    '''
    Put the value between 0 and 255
    '''
    max_pixel = np.max(image_array)
    min_pixel = np.min(image_array)

    normalized_image = 255 * (image_array - min_pixel) / (max_pixel - min_pixel)
    normalized_image = normalized_image.astype(np.uint8)
    return normalized_image

def compute_cr(image_array,k):
    '''
    Compute compression ratio of k
    '''
    m = image_array.shape[0]
    n = image_array.shape[1]
    return m*n/((m+n+1)*k)



def compute_ncc(imageA, imageB):
    """
    Compute NCC
    """
    meanA = np.mean(imageA)
    meanB = np.mean(imageB)
    ncc_val = np.sum((imageA - meanA) * (imageB - meanB)) / (
                np.sqrt(np.sum((imageA - meanA) ** 2)) * np.sqrt(np.sum((imageB - meanB) ** 2)))
    return ncc_val

def svd_compress_grey(image_array,k_list):
    '''
    Conduct svd on grey image by compressing k_list number of k
    '''
    U, S, V = np.linalg.svd(image_array, full_matrices=False)

    n_image = []
    cr_list, ssim_list= [], []
    ncc_list = []
    for k in tqdm(k_list):
        Up = U[:,:k]
        Sp = np.diag(S[:k])
        Vp = V[:k,:]
        compressed_image = np.dot(Up, np.dot(Sp, Vp))
        normalized_image = normalize_image(compressed_image)

        # Compute cr
        cr = round(compute_cr(image_array,k),4)

        # Compute ssim
        ssim_value = round(ssim(image_array, normalized_image, multichannel=False),4)

        # Compute ncc
        ncc = round(compute_ncc(image_array, normalized_image),4)


        n_image.append(normalized_image)
        cr_list.append(cr)
        ssim_list.append(ssim_value)
        ncc_list.append(ncc)
    
    return n_image, cr_list, ssim_list, ncc_list

def svd_compress_color(image_array,k_list):
    '''
    Conduct svd on color image by compressing k_list number of k
    ''' 
    n_image = []
    cr_list, ssim_list, ncc_list = [], [], []
    nl1,crl1,ssiml1,nccl1 = svd_compress_grey(image_array[:,:,0],k_list)
    nl2,crl2,ssiml2,nccl2 = svd_compress_grey(image_array[:,:,1],k_list)
    nl3,crl3,ssiml3,nccl3 = svd_compress_grey(image_array[:,:,2],k_list)

    for i in range(len(k_list)):
        normalized_image = np.zeros_like(image_array, dtype=np.uint8)
        normalized_image[:,:,0] = nl1[i]
        normalized_image[:,:,1] = nl2[i]
        normalized_image[:,:,2] = nl3[i]

        n_image.append(normalized_image)

        cr = round(np.mean(np.array([crl1[i],crl2[i],crl3[i]])),4)
        cr_list.append(cr)

        ssim_value = round(np.mean(np.array([ssiml1[i],ssiml2[i],ssiml3[i]])),4)
        ssim_list.append(ssim_value)

        ncc = round(np.mean(np.array([nccl1[i],nccl2[i],nccl3[i]])),4)
        ncc_list.append(ncc)
    return n_image, cr_list, ssim_list, ncc_list
                       
def plot_pics_grey(image_array,n_image,k_list,img_type):
    '''
    Plot the raw image and compressed image
    '''
    plt.figure(figsize=(12, 9))
    
    plt.subplot(3, 4, 1)
    plt.title("原始图像")
    plt.imshow(image_array,cmap='gray')
    plt.axis("off")
    
    for i, (k, compressed_image) in enumerate(zip(k_list, n_image)):
        plt.subplot(3, 4, i + 2)
        plt.title(f"压缩图像 (k={k})")
        plt.imshow(compressed_image,cmap='gray')
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig('./Pic/{}/image_all.jpg'.format(img_type))
    plt.show()

def plot_pics_color(image_array,n_image,k_list,img_type):
    '''
    Plot the raw image and compressed image
    '''
    plt.figure(figsize=(12, 9))
    
    plt.subplot(3, 4, 1)
    plt.title("原始图像")
    plt.imshow(image_array)
    plt.axis("off")
    
    for i, (k, compressed_image) in enumerate(zip(k_list, n_image)):
        plt.subplot(3, 4, i + 2)
        plt.title(f"压缩图像 (k={k})")
        plt.imshow(compressed_image)
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig('./Pic/{}/image_all.jpg'.format(img_type))
    plt.show()

def plot_metrics(df,img_type):
    '''
    Plot the similarity metrics
    '''
    plt.figure(figsize=(12,9))
    x = df.loc[:,'奇异值数量'].tolist()
    plt.plot(x,df.loc[:,'结构相似度'],'ro-',label='SSIM')
    plt.plot(x,df.loc[:,'归一化互相关'],'bs--',label='NCC')
    plt.legend()
    plt.xticks(x)
    plt.xlabel('奇异值数量')
    plt.ylabel('指标')
    plt.title('不同奇异值数量下的相似度指标')
    plt.savefig('./Pic/{}/metrics.jpg'.format(img_type))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Image compression with SVD')
    parser.add_argument('--image', type=str, default='grey_process', help='name of the image')
    args = parser.parse_args()
    path = args.image

    img = Image.open(f'./Data/{path}.jpg','r')
    # print(img)

    # 图像类型
    image_array = np.array(img)
    img_type = check_img_type(image_array)

    # 绘制奇异值和解释率变动曲线
    S = svd_full(image_array,img_type)
    # plot_sv_curve(S,img_type)
    # plot_info_curve(S,img_type)

    # 以50为间隔，设置K值
    k_list = get_k_list(S)

    if img_type == 'Grey':
        n_image, cr_list, ssim_list, ncc_list = svd_compress_grey(image_array,k_list)
        plot_pics_grey(image_array,n_image,k_list,img_type)
    else:
        n_image, cr_list, ssim_list, ncc_list = svd_compress_color(image_array,k_list)
        plot_pics_color(image_array,n_image,k_list,img_type)

    # Save metrics
    dic = {'奇异值数量':k_list,
        '压缩比':cr_list,
        '结构相似度':ssim_list,
        '归一化互相关':ncc_list}
    df = pd.DataFrame(dic)
    df.to_csv('./Metrics/{}.csv'.format(img_type),index=False,encoding='utf-8-sig')

    # Save metrics plot
    plot_metrics(df,img_type)

    # Save different images
    for k, compressed_image in zip(k_list, n_image):
        compressed_image = Image.fromarray(compressed_image)
        compressed_image.save(f"./Pic/{img_type}/compressed_images/compressed_image_{k}.jpg")