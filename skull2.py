import cv2, sys
import numpy as np
import nibabel as nib
import skimage
import maxflow

def main():
    #load data
    img3D    = nib.load("../1.nii.gz")
    img_data = img3D.get_data()

    #extract slice from an axis
    slc = img_data[:, 141, :]
    slc = np.uint8(slc*255.0/slc.max())

    cv2.imshow("teste", slc)
    cv2.waitKey(0)
    cv2.imwrite("slice.png", slc)

    g = maxflow.Graph[int]()
    node_ids = g.add_grid_nodes(slc.shape)
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [1, 1, 1]])
    weights = slc.copy()
    g.add_grid_edges(node_ids, 1, structure=structure, symmetric=True)
    y, x = slc.shape[0], slc.shape[1]
    g.add_tedge(node_ids[129,100], sys.maxsize, 0)
    g.add_tedge(node_ids[0,0], 0, sys.maxsize)
    g.add_tedge(node_ids[53,113], 0, sys.maxsize)
    #print(node_ids[slc.shape[0]//2,slc.shape[1]//2])

    g.maxflow()
    sgm = g.get_grid_segments(node_ids)
    #bw  = np.logical_not(sgm)
    img = np.uint8(sgm*255.0/sgm.max())

    cv2.imshow("teste", img)
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
