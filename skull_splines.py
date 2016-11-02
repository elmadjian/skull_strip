import cv2, sys
import numpy as np
import nibabel as nib
import skimage

def main():
    #load data
    img3D    = nib.load("../1.nii.gz")
    img_data = img3D.get_data()

    #extract slice from an axis
    for i in range(90, 180):
        slc = img_data[:, i, :]
        slc = np.uint8(slc*255.0/slc.max())

        cv2.imshow("teste", slc)
        cv2.waitKey(0)

        ret, thresh = cv2.threshold(slc, 0, 255, cv2.THRESH_OTSU)
        ret, label, stats, cent = cv2.connectedComponentsWithStats(thresh)
        area = [stats[i][-1] for i in range(1, len(stats))]
        maxv = np.argmax(area)+1
        thresh[label!=maxv] = 0
        cv2.imshow("teste", thresh)
        cv2.waitKey(0)

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
