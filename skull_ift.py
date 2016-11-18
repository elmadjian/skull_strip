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
    if neighborhood == 24:
        return [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1),
                (-2,-2), (-2,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,2),
                (2,2), (2,1), (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2)]

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

#Image-Foresting Transform
#--------------------------
def ift(img, current, neighborhood, conquest, Q, cost_dic, predecessor):
    for i in neighborhood:
        pixel = (current[0] + i[0], current[1] + i[1])
        if not is_valid_pixel(img, pixel):
            continue
        if conquest[pixel] != 255:
            #cost = cost_dic[current] + ((int(img[current]) + int(img[pixel]))/2)
            #if cost < cost_dic[pixel]:
            if cost_dic[pixel] >= 0 and img[pixel] > 30:
                #cost_dic[pixel] = cost
                predecessor[pixel] = current
                Q.put(pixel, cost_dic[pixel])

#Initialize all pixels with a specific cost rule
#------------------------------------------------
def initialize_costs(img, neighborhood):
    cost = {}
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            cost[(y,x)] = calc_cost(img, (y,x), neighborhood)
    return cost

#Calculates the cost at a specific pixel/voxel
#---------------------------------------------
def calc_cost(img, point, neighborhood):
    if img[point] == 0:
        return 0
    points = [(point[0] + i[0], point[1] + i[1]) for i in neighborhood]
    I = []
    for p in points:
        if is_valid_pixel(img, p):
            I.append(img[p])
    mean = np.mean(I)
    stdd = np.std(I)
    return 1/(3 * stdd + 1) - 1/(mean + 1)



def main():
    #load data
    img3D    = nib.load("../1.nii.gz")
    img_data = img3D.get_data()
    max_val  = img_data[:,:,:].max()
    norm_img = np.uint8(img_data[:,:,:]*255.0/max_val)
    conquest = np.zeros(norm_img.shape, dtype="uint8")

    #extract slice from an axis
    for i in range(80, 220):
        #i=181
        slc = norm_img[:, i, :]
        slcc = conquest[:, i, :]
        cv2.imshow("teste", slc)
        cv2.imwrite("slice.png", slc)
        cv2.waitKey(0)

        Q = graph.PriorityQueue()
        n = (144,56)
        Q.put(n, 0)
        neighborhood = get_neighborhood(4)
        window = get_neighborhood(24)
        predecessor = {}
        cost = initialize_costs(slc, window)
        cost[n] = 1

        while not Q.is_empty():
            lowest = Q.pop()
            slcc[lowest] = 255
            ift(slc, lowest, neighborhood, slcc, Q, cost, predecessor)
            # cv2.imshow("teste2", slcc)
            # cv2.waitKey(1)

        cv2.imshow("teste2", slcc)
        cv2.waitKey(0)


    print("finished")






if __name__=="__main__":
    main()
