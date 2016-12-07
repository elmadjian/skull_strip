import cv2, sys, bisect
import numpy as np
import nibabel as nib
import skimage
import matplotlib.pyplot as plt
import graph
from skimage import morphology



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
    z = pixel[2]
    if y >= 0 and y < img.shape[0]:
        if x >= 0 and x < img.shape[1]:
            if z >= 0 and z < img.shape[2]:
                return True
    return False

#Image-Foresting Transform - Background
#--------------------------------------
def ift_bg(img, current, neighborhood, conquest, Q, cost_dic):#, predecessor):
    for i in neighborhood:
        pixel = (current[0] + i[0], current[1] + i[1], current[2] + i[2])
        if not is_valid_pixel(img, pixel):
            continue
        if conquest[pixel] != 255:
            cost = cost_dic[current]
            if img[pixel] < img[current]:
                cost = np.log(int(img[pixel])/20.0 + 1) * 32
            if cost < cost_dic[pixel]:
                cost_dic[pixel] = cost
                #predecessor[pixel] = current
                Q.put(pixel, cost)

#Image-Foresting Transform - Foreground
#--------------------------------------
def ift_fg(img, current, neighborhood, conquest, Q, cost_dic):#, predecessor):
    for i in neighborhood:
        pixel = (current[0] + i[0], current[1] + i[1], current[2] + i[2])
        if not is_valid_pixel(img, pixel):
            continue
        if conquest[pixel] != 255:
            cost = 255 * np.exp(-int(img[pixel])/20.0)
            if cost < cost_dic[pixel]:
                cost_dic[pixel] = cost
                Q.put(pixel, cost)

#Initialize all pixels with a specific cost rule
#------------------------------------------------
def initialize_costs(img):#, neighborhood):
    cost = {}
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for z in range(img.shape[2]):
                cost[(y,x,z)] = np.log(int(img[(y,x,z)])/20.0 + 1) * 25#calc_cost(img, (y,x,z), neighborhood)
    return cost

#Calculates the cost at a specific pixel/voxel
#---------------------------------------------
def calc_cost(img, point, neighborhood):
    if img[point] == 0:
        return 0
    points = [(point[0] + i[0], point[1] + i[1], point[2] + i[2]) for i in neighborhood]
    I = []
    for p in points:
        if is_valid_pixel(img, p):
            I.append(img[p])
    mean = np.mean(I)
    stdd = np.std(I)
    return 1/(3 * stdd + 1) - 1/(mean + 1)


# Post-process to extract brain unrelated parts
#----------------------------------------------
def post_process(img, thresh):
    maximum = -1
    for i in range(img.shape[1]-1, -1, -2):
        slc = img[:, i, :]
        mean = np.mean(slc)
        if mean >= maximum:
            maximum = mean
        else:
            if mean < thresh:
                img[:, 0:i, :] = 0
                break

# Find starting point
#--------------------
def get_init_point(img):
    y = img.shape[0]
    x = img.shape[1]
    z = img.shape[2]
    yd = [[0, y//3], [y//3, (y//3)*2], [(y//3)*2, y]]
    xd = [[0, x//3], [x//3, (x//3)*2], [(x//3)*2, x]]
    zd = [[0, z//3], [z//3, (z//3)*2], [(z//3)*2, z]]
    cubes, coord = [], []
    for iy in yd:
        for ix in xd:
            for iz in zd:
                coords = [iy, ix, iz]
                cubes.append(img[iy[0]:iy[1], ix[0]:ix[1], iz[0]:iz[1]])
                coord.append(coords)
    argmax, highest_mean = -1, -1
    for i in range(len(cubes)):
        mean = np.mean(cubes[i])
        if mean > highest_mean:
            highest_mean = mean
            argmax = (y,x,z)
        y = int(coord[i][0][0] + (coord[i][0][1]-coord[i][0][0])/1.6)
        x = int(coord[i][1][0] + (coord[i][1][1]-coord[i][1][0])/1.6)
        z = int(coord[i][2][0] + (coord[i][2][1]-coord[i][2][0])/1.6)
    return argmax


#skull strip program
#-------------------
def main():
    #load data
    img3D    = nib.load("../6.nii.gz")
    img_data = img3D.get_data()
    max_val  = img_data[:,:,:].max()
    norm_img = np.uint8(img_data[:,:,:]*255.0/max_val)
    conquest = np.zeros(norm_img.shape, dtype="uint8")

    #grow ift 3D
    Q_fg = graph.PriorityQueue()
    n_fg = get_init_point(norm_img)
    Q_fg.put(n_fg, 0)
    neighborhood = get_neighborhood(6)

    print("applying 3D erosion...")
    norm_img = morphology.erosion(norm_img, morphology.ball(2))

    print("initializing costs...")
    cost = norm_img.astype("float32")
    cost = np.log(cost/20.0 + 1) * 30
    cost[n_fg] = 1

    print("processing foreground...")
    while not Q_fg.is_empty():
        lowest = Q_fg.pop()
        conquest[lowest] = 255
        ift_fg(norm_img, lowest, neighborhood, conquest, Q_fg, cost)

    print("applying post-processing...")
    rebuilt = morphology.dilation(conquest, morphology.ball(4))
    post_process(rebuilt, 6)

    print("saving label...")
    new_img = nib.Nifti1Image(rebuilt, np.eye(4))
    processed = nib.Nifti1Image(img_data, np.eye(4))
    nib.save(processed, "test_6.nii.gz")
    nib.save(new_img, "test_label_6.nii.gz")



if __name__=="__main__":
    main()
