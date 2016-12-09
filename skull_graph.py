import cv2
import numpy as np
import nibabel as nib
import skimage
import maxflow

def main():
    #load data
    img3D    = nib.load("../1.nii.gz")
    img_data = img3D.get_data()
    max_val  = img_data[:,:,:].max()
    norm_img = np.uint8(img_data[:,:,:]*255.0/max_val)

    #extract slice from an axis
    slc = img_data[:, 141, :]

    cv2.imshow("teste", slc)
    cv2.waitKey(0)

    g = maxflow.Graph[int]()
    node_ids = g.add_grid_nodes(slc.shape)
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [1, 1, 1]])
    weights = slc.copy()
    g.add_grid_edges(node_ids, 1, structure=structure, symmetric=True)
    g.add_grid_tedges(node_ids, 200-slc, slc)
    g.maxflow()

    sgm = g.get_grid_segments(node_ids)
    bw  = np.logical_not(sgm)
    img = np.uint8(bw*255.0/bw.max())

    cv2.imshow("teste2", img)
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
