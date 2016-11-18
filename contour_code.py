import cv2, sys, bisect
import numpy as np
import nibabel as nib
import skimage
import matplotlib.pyplot as plt

# Get local adjacency
#---------------------
def get_neighborhood(neighborhood):
    if neighborhood == 4:
        return [(-1,0), (0,1), (1,0), (0,-1)]
    if neighborhood == 6:
        return [(-1,0,0),(0,1,0),(1,0,0),(0,-1,0),(0,0,-1),(0,0,1)]
    if neighborhood == 8:
        return [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    if neighborhood == 24:
        return [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1),
                (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,2),
                (2,2), (2,1), (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2)]


#Initialize all pixels with a specific cost rule
#------------------------------------------------
def initialize_costs(img, neighborhood):
    cost = {}
    img_copy = img.copy()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            img_copy[(y,x)] = 255
            cost[(y,x)] = calc_cost(img, (y,x), neighborhood)
            cv2.imshow("teste", img_copy)
            cv2.waitKey(10)
    return cost

#Check whether the element is inside the image/volume
#----------------------------------------------------
def is_valid_pixel(img, pixel):
    y = pixel[0]
    x = pixel[1]
    if len(img.shape) == 3:
        z = pixel[2]
    if y >= 0 and y < img.shape[0]:
        if x >= 0 and x < img.shape[1]:
            if len(img.shape) == 2:
                return True
            else:
                if z >= 0 and z < img.shape[2]:
                    return True
    return False

#Calculates the cost at a specific pixel/voxel
#---------------------------------------------
def calc_cost(img, point, neighborhood):
    points = [(point[0] + i[0], point[1] + i[1]) for i in neighborhood]
    I = []
    for p in points:
        if is_valid_pixel(img, p):
            I.append(img[p])
    mean = np.mean(I)
    stdd = np.std(I)
    print(mean, stdd)
    #return mean + 1.0/stdd


def main():
    img = cv2.imread(sys.argv[1], 0)
    ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("teste", thresh)
    cv2.waitKey(0)

    # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    # tam = len(contours[0])
    # for i in range(tam):
    #     xi = contours[0][i][0][0]
    #     yi = contours[0][i][0][1]
    #     pixel_sum = 0
    #     for j in range(10):
    #         xjA = contours[0][(i+j)%tam][0][0]
    #         yjA = contours[0][(i+j)%tam][0][1]
    #         xjB = contours[0][i-j][0][0]
    #         yjB = contours[0][i-j][0][1]
    #         pixel_sum += img[yjA, xjA] + img[yjB, xjB]
    #     print(pixel_sum)

    neighborhood = get_neighborhood(8)
    initialize_costs(img, neighborhood)




    # cv2.imshow("teste", thresh)
    # cv2.waitKey(0)





if __name__=="__main__":
    main()
