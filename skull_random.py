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
    label[slc<10] = 100
    label[slc==100] = 200
    # label[126,154] = 1
    # label[125,100] = 2
    # label[208,121] = 1
    # label[185,111] = 2

    cv2.imshow("teste3", np.uint8(label))
    cv2.imwrite("random2.png", np.uint8(label))
    cv2.waitKey(0)

    result = random_walker(slc, label, beta=1000)
    result = skimage.img_as_ubyte(result)
    result = np.uint8(result*255.0/np.max(result))
    cv2.imshow("teste2", result)
    cv2.imwrite("random3.png", result)
    cv2.waitKey(0)

    # half = slc.shape[0]//1.5
    # x_list = []
    # y_list = []
    # previous = 0
    # for i in range(slc.shape[1]):
    #     x_list.append(i)
    #     y_list.append(slc[half,i])
    #     previous = int(slc[half,i])
    #     slc[half,i] = 255
    #
    # cv2.imshow("teste", slc)
    # cv2.waitKey(0)
    # plt.plot(x_list, y_list)
    # plt.show()


    # ret, thresh = cv2.threshold(slc, 0, 255, cv2.THRESH_OTSU)
    # ret, label, stats, cent = cv2.connectedComponentsWithStats(thresh)
    # area = [stats[i][-1] for i in range(1, len(stats))]
    # maxv = np.argmax(area)+1
    # thresh[label!=maxv] = 0
    # cv2.imshow("teste", thresh)
    # cv2.waitKey(0)

    # slice1 = np.uint16(slice1*65535.0/np.max(slice1))
    # slice2 = np.uint16(slice2*65535.0/np.max(slice2))
    # slice3 = np.uint16(slice3*65535.0/np.max(slice3))
    #
    # cv2.imshow("teste", slice1)
    # cv2.waitKey(0)
    #
    # cv2.imshow("teste", slice2)
    # cv2.waitKey(0)
    #
    # cv2.imshow("teste", slice3)
    # cv2.waitKey(0)


    #watch the movie:)
    # for i in range(img_data.shape[1]):
    #     slc = img_data[:, i, :]
    #     slc = np.uint8(slc*255.0/slc.max())
    #     cv2.imshow("teste", slc)
    #     cv2.waitKey(50)


if __name__=="__main__":
    main()
