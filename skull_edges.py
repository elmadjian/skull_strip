import cv2, sys
import numpy as np
import nibabel as nib
import skimage
import matplotlib.pyplot as plt

def main():
    #load data
    img3D    = nib.load("../6.nii.gz")
    img_data = img3D.get_data()
    max_val  = img_data[:,:,:].max()
    norm_img = np.uint8(img_data[:,:,:]*255.0/max_val)

    print(norm_img.shape)

    y = norm_img.shape[0]
    x = norm_img.shape[1]
    z = norm_img.shape[2]
    yd = [[0, y//3], [y//3, (y//3)*2], [(y//3)*2, y]]
    xd = [[0, x//3], [x//3, (x//3)*2], [(x//3)*2, x]]
    zd = [[0, z//3], [z//3, (z//3)*2], [(z//3)*2, z]]

    cubes = []
    coord = []
    for iy in yd:
        for ix in xd:
            for iz in zd:
                coords = [iy, ix, iz]
                cubes.append(norm_img[iy[0]:iy[1], ix[0]:ix[1], iz[0]:iz[1]])
                coord.append(coords)

    argmax = -1
    highest_mean = -1
    for i in range(len(cubes)):
        mean = np.mean(cubes[i]))
        if mean > highest_mean:
            highest_mean = mean
            argmax = (y,x,z)
        y = coord[i][0][0] + (coord[i][0][1]-coord[i][0][0])//1.5
        x = coord[i][1][0] + (coord[i][1][1]-coord[i][1][0])//1.5
        z = coord[i][2][0] + (coord[i][2][1]-coord[i][2][0])//1.5


        # for i in range(cube.shape[1]):
        #
        #     slc = cube[:,i,:]
        #     cv2.imshow("teste", slc)
        #     cv2.waitKey(0)

    # maxpeak = -1
    # minpeak = sys.maxsize
    # interval = [0 for i in range(10)]
    # prev = 0
    # count = 0
    #
    # maxpoint = {}

    # #extract slice from an axis
    # for i in range(0, 240):
    #     #i = 140
    #     slc = norm_img[:, i, :]
    #
    #     cv2.imshow("teste", slc)
    #     cv2.waitKey(0)
    #
    #     # diff = abs(np.mean(slc) - np.std(slc))
    #     # interval.pop(0)
    #     # interval.append(diff)
    #     # top = np.max(interval)
    #     # if top > maxpeak:
    #     #     maxpeak = top
    #     # elif top < minpeak:
    #     #     minpeak = top
    #     # if top > prev:
    #     #     count += 1
    #     #     print("crescendo---->", count)
    #     # if top < prev:
    #     #     count -= 1
    #     #     print("descendo<----", count)
    #     # prev = top
    #     #
    #     #
    #     # print("max:", maxpeak, "min:", minpeak, "diff:", maxpeak-minpeak)
    #     #
    #     # print("calc:", top)
    #     point = np.unravel_index(slc.argmax(), slc.shape)
    #     if point not in maxpoint.keys():
    #         maxpoint[point] = 0
    #     maxpoint[point] += 1
    #
    #     print("argmax:", np.unravel_index(slc.argmax(), slc.shape))
    #     print("---------------------")
    #
    #     half = slc.shape[0]//2
    #     x_list = []
    #     y_list = []
    #     for i in range(slc.shape[1]):
    #         x_list.append(i)
    #         y_list.append(slc[half][i])
    #         slc[half][i] = 255
    #
    #     # cv2.imshow("teste", slc)
    #     # cv2.waitKey(0)
    #     # plt.plot(x_list, y_list)
    #     # plt.show()
    # for k in maxpoint.keys():
    #     print(k, maxpoint[k])


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
