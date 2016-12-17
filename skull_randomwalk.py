import cv2, sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import skimage
from skimage.segmentation import random_walker

def main():
    #load data
    img3D    = nib.load("../1.nii.gz")
    img_data = img3D.get_data()
    max_val  = img_data[:,:,:].max()
    norm_img = np.uint8(img_data[:,:,:]*255.0/max_val)

    #extract slice from an axis
    #for i in range(10, 220):
    i = 140
    slc = norm_img[:, i, :]
    label = np.zeros(slc.shape, np.int)

    cv2.imshow("teste", slc)
    cv2.imwrite("random1.png", slc)
    cv2.waitKey(0)

    label[slc==0] = -1
    label[slc<10] = 60
    label[slc>90] = 200

    cv2.imshow("teste3", np.uint8(label))
    cv2.imwrite("random2.png", np.uint8(label))
    cv2.waitKey(0)

    result = random_walker(slc, label, beta=1000)
    result = skimage.img_as_ubyte(result)
    result = np.uint8(result*255.0/np.max(result))
    cv2.imshow("teste2", result)
    cv2.imwrite("random3.png", result)
    cv2.waitKey(0)



if __name__=="__main__":
    main()
