from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import pandas as pd
from math import log

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18

# color_image_path = './Noise_Process/noise.jpg'
# color_image = Image.open(color_image_path)

# # 将彩色图像转换为灰度图像
# gray_image = color_image.convert('L')

# # 保存灰度图像到文件
# gray_image_path = './Noise_Process/noise_process.jpg'
# gray_image.save(gray_image_path)

# color_image_path = './Noise_Process/noise_process.jpg'
# image = Image.open(color_image_path)
# image_array = np.array(image)

# 1. 均匀噪声
# min_val = -20
# max_val = 20

# uniform_noise = np.random.uniform(min_val, max_val, image_array.shape)

# noisy_image_array = image_array + uniform_noise

# noisy_image_array = np.clip(noisy_image_array, 0, 255).astype('uint8')

# noisy_image = Image.fromarray(noisy_image_array)

# noisy_image.save('./Noise_Process/noise_uniform.jpg')
#noisy_image.show()

# 2. 高斯噪声
# sigma = 20

# gaussian_noise = np.random.normal(0, sigma, image_array.shape)

# noisy_image_array = image_array + gaussian_noise

# noisy_image_array = np.clip(noisy_image_array, 0, 255).astype('uint8')

# noisy_image = Image.fromarray(noisy_image_array)

# noisy_image.save('./Noise_Process/noise_gaussian.jpg')
# noisy_image.show()

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

def plot_sv_curve(S,path):
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

        plt.savefig('./Noise_Process/{}_sv_curve.jpg'.format(path))


def plot_info_curve(S,path):
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

        plt.savefig('./Noise_Process/{}_info_curve.jpg'.format(path))

    
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

# def normalize_image(image_array):
#     '''
#     Put the value between 0 and 255
#     '''
#     max_pixel = np.max(image_array)
#     min_pixel = np.min(image_array)

#     normalized_image = 255 * (image_array - min_pixel) / (max_pixel - min_pixel)
#     normalized_image = normalized_image.astype(np.uint8)
#     return normalized_image

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

def svd_compress_grey(image_array,k_list,raw_image_array):
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
        normalized_image = np.clip(compressed_image,0, 255).astype('uint8')

        # Compute cr
        cr = round(compute_cr(raw_image_array,k),4)

        # Compute ssim
        # ssim_value = round(ssim(raw_image_array, normalized_image, multichannel=False),4)
        mse = np.mean((raw_image_array - normalized_image) ** 2)
        
        # 避免除以0的情况
        if mse == 0:
            return float('inf')
        
        # 计算PSNR
        ssim_value = 20 * log(255 / np.sqrt(mse))
        # Compute ncc
        ncc = round(compute_ncc(raw_image_array, normalized_image),4)


        n_image.append(normalized_image)
        cr_list.append(cr)
        ssim_list.append(ssim_value)
        ncc_list.append(ncc)
    
    return n_image, cr_list, ssim_list, ncc_list

def compute_psnr(raw_image_array,normalized_image):
    mse = np.mean((raw_image_array - normalized_image) ** 2)
    
    # 避免除以0的情况
    if mse == 0:
        return float('inf')
    
    # 计算PSNR
    psnr_value = 20 * log(255 / np.sqrt(mse))

    return psnr_value
                       
def plot_pics_grey(image_array,n_image,k_list,path):
    '''
    Plot the raw image and compressed image
    '''
    plt.figure(figsize=(12, 9))
    
    plt.subplot(3, 4, 1)
    plt.title("加噪图像")
    plt.imshow(image_array,cmap='gray')
    plt.axis("off")
    
    for i, (k, compressed_image) in enumerate(zip(k_list, n_image)):
        plt.subplot(3, 4, i + 2)
        plt.title(f"降噪图像 (k={k})")
        plt.imshow(compressed_image,cmap='gray')
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig('./Noise_Process/{}_image_change.jpg'.format(path))
    # plt.show()


def plot_metrics(df,path):
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
    plt.savefig('./Noise_Process/{}.jpg'.format(path))

if __name__ == '__main__':

    noise_type = 'uniform'

    img = Image.open(f'./Noise_Process/noise_{noise_type}.png','r')
    image_array = np.array(img)
    img_type = check_img_type(image_array)

    raw_img = Image.open(f'./Noise_Process/noise_process.jpg','r')
    raw_image_array = np.array(raw_img)

    # 绘制奇异值和解释率变动曲线
    S = svd_full(image_array,img_type)
    plot_sv_curve(S,noise_type)
    plot_info_curve(S,noise_type)

    k_list = get_k_list(S)

    if img_type == 'Grey':
        n_image, cr_list, ssim_list, ncc_list = svd_compress_grey(image_array,k_list,raw_image_array)
        plot_pics_grey(image_array,n_image,k_list,noise_type)

    # Save metrics
    dic = {'奇异值数量':k_list,
        '压缩比':cr_list,
        '结构相似度':ssim_list,
        '归一化互相关':ncc_list}
    df = pd.DataFrame(dic)

    df.to_csv('./Noise_Process/{}_metrics.csv'.format(noise_type),index=False,encoding='utf-8-sig')

    # plot_metrics(df,noise_type)

    # for k, compressed_image in zip(k_list, n_image):
    #     compressed_image = Image.fromarray(compressed_image)
    #     compressed_image.save(f"./Noise_Process/{noise_type}_image_{k}.jpg")