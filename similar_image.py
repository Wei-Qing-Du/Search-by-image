import cv2
import os
import numpy as np
import math
from PIL import Image
from numpy import average, dot, linalg 

# phash
def phash(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    #Create two dimension
    h ,w = img.shape[ :2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = img
    

    #DCT
    img_dct = cv2.dct(cv2.dct(vis0))
    img_dct = np.resize(img_dct, (8,8))
    imgTolist = img_dct.tolist()
    latten = lambda l: [item for sublist in l for item in sublist]
    img_list = latten(imgTolist)

    #Caculate the mean
    img_mean = cv2.mean(np.array(img_list))
    avg_list = ['0' if i < img_mean[0] else '1' for i in img_list]
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,64,4)])

def hammingDistance(x, y):
    hamming_distance = 0
    s = str(bin(x^y))
    for i in range(2,len(s)):
        if int(s[i]) is 1:
            hamming_distance += 1
    return hamming_distance 

 # ��Ϥ��i��Τ@�ƳB�z
def get_thum(image, size=(64,64), greyscale=False):
    # �Q��image��v�H�j�p���s�]�w, Image.ANTIALIAS������q��
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # �N�Ϥ��ഫ��L�Ҧ��A�䬰�ǫ׹ϡA��C�ӵe����8��bit���
        image = image.convert('L')
    return image

# ?��?�����E���Z��
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear�]?�ʡ^+algebra�]�N?�^�Anorm?��ܭS?
        # �D?�����S?�H�H
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot��^���O��?�A?�G???�]�x?�^?��?��
    res = dot(a / a_norm, b / b_norm)
    return res

def psnr(img1, img2):
    mse = np.mean( (np.array(img1) - np.array(img2)) ** 2 )
    if mse == 0:
        return 100

    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


#art_models = ['models_anime_style_art', 'models_anime_style_art_rgb', 'models_cunet_art', 'models_upconv_7_art', 
               # 'models_vgg_7_art', 'models_vgg_7_art_y']
art_models = ['models_anime_style_art']
photo_models = ['models_photo', 'models_upconv_7_photo', 'models_vgg_7_photo']
dis =[]
dis_image = dict()
base_path = 'C:\\Users\\Willy\\Desktop\\'
target_image = 'C:\\Users\\Willy\\Desktop\\output_cpu.png'#phash('C:\\Users\\Willy\\Desktop\\Illustration, upsample x2, denoise=100.png')

metrics = input("Please input your metric:")

if(metrics == "distance"):
    img = Image.open(target_image)
elif(metrics == "psnr"):
    img = cv2.imread(target_image)
    img = cv2.resize(img, (800, 558))


for models_path in art_models:
    i = 0
    base_path = base_path + models_path

    if os.path.exists(base_path):#Check wether this file is exist or not
        files = os.listdir(base_path) #Find all files in this file
        for image_art in files:
            image_art = base_path + "\\" + image_art
            if(metrics == "distance"):
                img1 = Image.open(image_art)
                dis.append(image_similarity_vectors_via_numpy(img, img1)) #hammingDistance(int(target_image, 16), int(phash(image_art), 16)"")
            elif(metrics == "psnr"):
                img1 = cv2.imread(image_art)
                #img1 = cv2.resize(img1, (800, 558)) #For difference size
                dis.append(psnr(img, img1))

            dis_image[files[i]] = dis[i]
            i = i + 1
    else:
        print("Not exist")
    print(dis_image)
    print("====================================\n")
    dis_image.clear()
    base_path = 'C:\\Users\\Willy\\Desktop\\'


