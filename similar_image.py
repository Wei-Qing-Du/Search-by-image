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

 # ¹ï¹Ï¤ù¶i¦æ²Î¤@¤Æ³B²z
def get_thum(image, size=(64,64), greyscale=False):
    # §Q¥Îimage¹ï¼v¶H¤j¤p­«·s³]©w, Image.ANTIALIAS¬°°ª½è¶qªº
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # ±N¹Ï¤ùÂà´«¬°L¼Ò¦¡¡A¨ä¬°¦Ç«×¹Ï¡A¨ä¨C­Óµe¯À¥Î8­Óbitªí¥Ü
        image = image.convert('L')
    return image

# ?ºâ?¤ùªº§E©¶¶ZÖÃ
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
        # linalg=linear¡]?©Ê¡^+algebra¡]¥N?¡^¡Anorm?ªí¥Ü­S?
        # ¨D?¤ùªº­S?¡H¡H
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dotªð¦^ªº¬O˜ò?¡A?¤G???¡]¯x?¡^?¦æ?ºâ
    res = dot(a / a_norm, b / b_norm)
    return res

art_models = ['models_anime_style_art', 'models_anime_style_art_rgb', 'models_cunet_art', 'models_upconv_7_art', 
                'models_vgg_7_art', 'models_vgg_7_art_y']
#art_models = ['models_anime_style_art', 'models_cunet_art']
photo_models = ['models_photo', 'models_upconv_7_photo', 'models_vgg_7_photo']
dis =[]
dis_image = dict()
base_path = 'C:\\Users\\Willy\\Desktop\\'
target_image = 'C:\\Users\\Willy\\Desktop\\Illustration, upsample x2, denoise=100.png'#phash('C:\\Users\\Willy\\Desktop\\Illustration, upsample x2, denoise=100.png')
img = Image.open(target_image)

for models_path in art_models:
    i = 0
    base_path = base_path + models_path

    if os.path.exists(base_path):#Check wether this file is exist or not
        files = os.listdir(base_path) #Find all files in this file
        for image_art in files:
            image_art = base_path + "\\" + image_art
            img1 = Image.open(image_art)
            dis.append(image_similarity_vectors_via_numpy(img, img1)) #hammingDistance(int(target_image, 16), int(phash(image_art), 16)"")
            dis_image[files[i]] = dis[i]
            i = i + 1
    else:
        print("Not exist")
    print(dis_image)
    print("====================================\n")
    dis_image.clear()
    base_path = 'C:\\Users\\Willy\\Desktop\\'


