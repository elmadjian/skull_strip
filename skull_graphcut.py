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
    slc = norm_img[:, 141, :]

    cv2.imshow("teste", slc)
    cv2.waitKey(0)

    g = maxflow.Graph[int]()
    node_ids = g.add_grid_nodes(slc.shape)
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [1, 1, 1]])
    weights = slc.copy()
    g.add_grid_edges(node_ids, 15, structure=structure, symmetric=True)
    g.add_grid_tedges(node_ids, 100-slc, slc)
    g.maxflow()

    sgm = g.get_grid_segments(node_ids)
    bw  = np.logical_not(sgm)
    img = np.uint8(bw*255.0/bw.max())
    img = 255 - img

    cv2.imshow("teste2", img)
    cv2.imwrite("141_high.jpg", img)
    cv2.waitKey(0)


if __name__=="__main__":
    main()
