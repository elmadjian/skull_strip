import cv2
import numpy as np
import nibabel as nib
import skimage
import maxflow
from skimage import morphology

#-------------------------------------
def is_valid_pixel(img, pixel):
    y = pixel[0]
    x = pixel[1]
    if y >= 0 and y < img.shape[0]:
        if x >= 0 and x < img.shape[1]:
            return True
    return False


#-------------------------------------
def get_neighbors(level):
    lvl_0 = [(0,-1), (-1,0), (0,1), (1,0)]
    lvl_1 = [(-1,-1),(-1,1), (1,1), (1,-1)]
    lvl_2 = [(-2,-2),(-2,-1),(-2,0),(-2,1),
             (-2,2), (-1,2), (0,2), (1,2),
             (2,2),  (2,1),  (2,0), (2,-1),
             (2,-2), (1,-2), (0,-2),(-1,-2)]
    lvl_3 = [(-3,-3),(-3,-2),(-3,-1),(-3,0),(-3,1), (-3,2),
             (-3,3), (-2,3), (-1,3), (0,3), (1,3),  (2,3),
             (3,3),  (3,2),  (3,1),  (3,0), (3,-1), (3,-2),
             (3,-3), (2,-3), (1,-3), (0,-3),(-1,-3),(-2,-3)]
    if level == 0:
        return lvl_0
    elif level == 1:
        return lvl_1 + lvl_0
    elif level == 2:
        return lvl_2 + lvl_1 + lvl_0
    elif level == 3:
        return lvl_3 + lvl_2 + lvl_1 + lvl_0
    else:
        print("There's no such level yet...")


#-------------------------------------
def grow_bg(image, conquest, current, neighborhood, stack):
    for i in neighborhood:
        n_pixel = (current[0] + i[0], current[1] + i[1])
        if not is_valid_pixel(image, n_pixel):
            continue
        if conquest[n_pixel] == 0:
            cost = image[n_pixel]
            if cost < 30:
                stack.append(n_pixel)
                conquest[n_pixel] = 100
            else:
                conquest[n_pixel] = 150

# #------------------------------------
def grow_fg(threshold, conquest, current, neighborhood, stack):
    pixels = [(current[0] + i[0], current[1] + i[1]) for i in neighborhood]
    borders = [conquest[i] for i in pixels]
    if borders.count(100) < threshold:
        for p in pixels:
            if conquest[p] != 100 and conquest[p] != 255:
                stack.append(p)
                conquest[p] = 255


#-------------------------------------
def main():
    #load data
    img3D    = nib.load("../1.nii.gz")
    img_data = img3D.get_data()
    max_val  = img_data[:,:,:].max()
    norm_img = np.uint8(img_data[:,:,:]*255.0/max_val)

    #extract slice from an axis
    for i in range(115, 220):
        slc = norm_img[:, i, :]

        cv2.imshow("teste", slc)
        cv2.waitKey(0)

        seed = (197,46)
        stack = [seed]
        conquest = np.zeros(slc.shape, dtype=np.uint8)
        conquest[seed] = 100
        conquest[slc==0] = 100
        neighborhood = get_neighbors(2)

        while stack:
            current = stack.pop()
            grow_bg(slc, conquest, current, neighborhood, stack)
            #print(stack)
            # cv2.imshow("teste2", conquest)
            # cv2.waitKey(1)

        # closing = morphology.closing(conquest, morphology.disk(1))
        # closing = skimage.img_as_ubyte(closing)
        cv2.imshow("teste2", conquest)
        cv2.waitKey(0)

        seed = (slc.shape[0]//2, slc.shape[1]//2)
        stack = [seed]
        neighborhood = get_neighbors(2)
        threshold = len(neighborhood)//5
        print("threshold:", threshold)
        while stack:
            current = stack.pop()
            grow_fg(threshold, conquest, current, neighborhood, stack)
            # cv2.imshow("teste2", conquest)
            # cv2.waitKey(1)

        cv2.imshow("teste2", conquest)
        cv2.waitKey(0)
        print("saindo")

        #cv2.imwrite("141.jpg", slc)


if __name__=="__main__":
    main()
