import cv2, sys, pq
import numpy as np
import nibabel as nib
from skimage import morphology

# Global parameters
#~~~~~~~~~~~~~~~~~~
alpha   = 31    #controls the boundary between background and foreground
erosion = 2     #erosion parameter to disconnect the brain
dilat   = 5     #dilation parameter to rebuild the segmented brain
closing = 4     #closing parameter of brain holes
cut     = 6     #cut parameter for brainstem


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


#Image-Foresting Transform - Foreground
#--------------------------------------
def ift_fg(img, current, neighborhood, conquest, Q, cost_dic):
    for i in neighborhood:
        pixel = (current[0] + i[0], current[1] + i[1], current[2] + i[2])
        if not is_valid_pixel(img, pixel):
            continue
        if conquest[pixel] != 255:
            cost = 255 * np.exp(-int(img[pixel])/20.0)
            if cost < cost_dic[pixel]:
                cost_dic[pixel] = cost
                Q.put(pixel, cost)


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
    y, x, z = img.shape[0], img.shape[1], img.shape[2]
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
        y = int(coord[i][0][0] + (coord[i][0][1]-coord[i][0][0])/1.2)
        x = int(coord[i][1][0] + (coord[i][1][1]-coord[i][1][0])/1.2)
        z = int(coord[i][2][0] + (coord[i][2][1]-coord[i][2][0])/1.2)
    return argmax


#opens a 3D image
#----------------
def open_file(sys_input):
    if len(sys_input) != 3:
        print("usage: <this_program> <3D_image> <3D_labelname>")
        sys.exit()
    return nib.load(sys_input[1])



#skull strip program
#-------------------
def main():
    #load data
    img3D    = open_file(sys.argv)
    img_data = img3D.get_data()
    max_val  = img_data[:,:,:].max()
    norm_img = np.uint8(img_data[:,:,:]*255.0/max_val)
    conquest = np.zeros(norm_img.shape, dtype="uint8")

    Q_fg = pq.PriorityQueue()
    n_fg = get_init_point(norm_img)
    Q_fg.put(n_fg, 0)
    neighborhood = get_neighborhood(6)

    print("applying 3D erosion...")
    norm_img = morphology.erosion(norm_img, morphology.ball(erosion))

    print("initializing costs...")
    cost = norm_img.astype("float32")
    cost = np.log(cost/20.0 + 1) * alpha
    cost[n_fg] = 1

    print("processing voxel expansion...")
    while not Q_fg.is_empty():
        lowest = Q_fg.pop()
        conquest[lowest] = 255
        ift_fg(norm_img, lowest, neighborhood, conquest, Q_fg, cost)

    print("applying post-processing...")
    rebuilt = morphology.dilation(conquest, morphology.ball(dilat))
    rebuilt = morphology.closing(rebuilt, morphology.ball(closing))
    post_process(rebuilt, cut)

    print("saving label...")
    new_img = nib.Nifti1Image(rebuilt, np.eye(4))
    nib.save(new_img, sys.argv[2])


if __name__=="__main__":
    main()
