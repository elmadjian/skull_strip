import cv2, sys, bisect
import numpy as np
import nibabel as nib
import skimage
import matplotlib.pyplot as plt



def main():
    img = cv2.imread(sys.argv[1], 0)
    ret, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("teste", thresh)
    cv2.waitKey(0)

    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    tam = len(contours[0])
    for i in range(tam):
        xi = contours[0][i][0][0]
        yi = contours[0][i][0][1]
        pixel_sum = 0
        for j in range(10):
            xjA = contours[0][(i+j)%tam][0][0]
            yjA = contours[0][(i+j)%tam][0][1]
            xjB = contours[0][i-j][0][0]
            yjB = contours[0][i-j][0][1]
            pixel_sum += img[yjA, xjA] + img[yjB, xjB]
        print(pixel_sum)



    cv2.imshow("teste", thresh)
    cv2.waitKey(0)





if __name__=="__main__":
    main()
