import cv2, sys, bisect
import numpy as np
import nibabel as nib
import skimage
import matplotlib.pyplot as plt
import graph



# Get local adjacency
#---------------------
def get_neighborhood(neighborhood):
    if neighborhood == 4:
        return [(-1,0), (0,1), (1,0), (0,-1)]
    if neighborhood == 6:
        return [(-1,0,0),(0,1,0),(1,0,0),(0,-1,0),(0,0,-1),(0,0,1)]
    if neighborhood == 8:
        return [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]


#-------------------------
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

#Image-Forest Transform
#---------------------------------
def ift(img, current, neighborhood, conquest, Q, cost_dic, predecessor):
    for i in neighborhood:
        pixel = (current[0] + i[0], current[1] + i[1])
        if not is_valid_pixel(img, pixel):
            continue
        if conquest[pixel] != 255:
            cost = cost_dic[current] + ((int(img[current]) + int(img[pixel]))/2)
            if cost < cost_dic[pixel]:
                cost_dic[pixel] = cost
                predecessor[pixel] = current
                Q.put(pixel, cost)

#Initialize all pixels with cost infinity
#----------------------------------------
def initialize_costs(img):
    cost = {}
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            #COLOCAR 3D aqui
            if img[(y,x)] < 30:
                cost[(y,x)] = 0
            else:
                cost[(y,x)] = sys.maxsize
    return cost


def main():
    #load data
    img3D    = nib.load("../1.nii.gz")
    img_data = img3D.get_data()
    max_val  = img_data[:,:,:].max()
    norm_img = np.uint8(img_data[:,:,:]*255.0/max_val)
    conquest = np.zeros(norm_img.shape, dtype="uint8")

    #extract slice from an axis
    #for i in range(20, 220):
    i=121
    slc = norm_img[:, i, :]
    slcc = conquest[:, i, :]
    cv2.imshow("teste", slc)
    cv2.imwrite("slice.png", slc)
    cv2.waitKey(0)

    Q = graph.PriorityQueue()
    n = (144,56)
    Q.put(n, 0)
    neighborhood = get_neighborhood(4)
    predecessor = {}
    cost = initialize_costs(slc)
    cost[n] = 0

    while not Q.is_empty():
        lowest = Q.pop()
        slcc[lowest] = 255
        ift(slc, lowest, neighborhood, slcc, Q, cost, predecessor)
        cv2.imshow("teste2", slcc)
        cv2.waitKey(1)


    print("finished")






if __name__=="__main__":
    main()
