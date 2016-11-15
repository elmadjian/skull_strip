import cv2, sys, bisect
import numpy as np
import nibabel as nib
import skimage
import matplotlib.pyplot as plt
from skimage import morphology

#===============================================================================
class Graph():
    def __init__(self, img=None):
        self.seed = None
        self.node = {}
        self.img = img
        self.visited = set()

    def set_img(self, img):
        self.img = img

    def set_arch(self, node_A, node_B, color):
        if node_A.has_out_arch():
            node = node_A.get_out_arch()
            node.remove_in_arch(node_A)
        node_A.set_out_arch(node_B)
        node_B.set_in_arch(node_A)
        self.node[node_A.pixel] = node_A
        self.node[node_B.pixel] = node_B
        self._mark_node(node_B.pixel, color)

    def has_arch(self, pixel_A, pixel_B):
        node_B = self.node[pixel_B]
        if self.node[pixel_A].has_out_arch(node_B):
            return True
        return False

    def add_node(self, node):
        self.node[node.pixel] = node

    def get_node(self, pixel):
        return self.node[pixel]

    def has_node(self, pixel):
        if pixel in self.node:
            return True
        return False

    def add_visited(self, pixel):
        self.visited.add(pixel)

    def is_visited(self, pixel):
        if pixel in self.visited:
            return True
        return False

    def has_8_neighbor(self, pixel):
        adj_list = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
        for i in adj_list:
            p = (pixel[0] + i[0], pixel[1] + i[1])
            if p in self.node:
                return self.node[p]
        return False

    def has_4_neighbor(self, pixel):
        adj_list = [(-1,0), (0,1), (1,0), (0,-1)]
        for i in adj_list:
            p = (pixel[0] + i[0], pixel[1] + i[1])
            if p in self.node:
                return self.node[p]
        return False

    def _mark_node(self, pixel, color):
        self.img[pixel] = color

#===============================================================================
class Node():
    def __init__(self, pixel, cost=sys.maxsize, distance=0):
        self.in_arch = set()
        self.out_arch = None
        self.pixel = pixel
        self.cost = cost
        self.distance = distance

    def has_in_arch(self):
        if self.in_arch:
            return True
        return False

    def has_out_arch(self):
        if self.out_arch:
            return True
        return False

    def get_in_arch(self):
        if len(self.in_arch) > 0:
            return list(self.in_arch)[0]

    def get_out_arch(self):
        return self.out_arch

    def get_in_degree(self):
        return len(self.in_arch)

    def set_in_arch(self, node):
        self.in_arch.add(node)

    def set_out_arch(self, node):
        self.out_arch = node

    def remove_in_arch(self, node):
        self.in_arch.discard(node)

    def y(self):
        return int(self.pixel[0])

    def x(self):
        return int(self.pixel[1])

    def __lt__(self, other):
        self.cost < other.cost

#===============================================================================

class PriorityQueue():
    def __init__(self):
        self.queue = []
        self.nodes = {}

    def put(self, item, priority):
        if item in self.nodes:
            pos = bisect.bisect_right(self.queue, [self.nodes[item], item])
            del self.queue[pos-1]
        bisect.insort_right(self.queue, [priority, item])
        self.nodes[item] = priority

    def pop(self):
        if not self.empty():
            item = self.queue.pop(0)[1]
            if item in self.nodes:
                del self.nodes[item]
            return item
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        if self.queue:
            return False
        return True

#===============================================================================


def get_lines360(img):
    lx = img.shape[1]-1
    ly = img.shape[0]-1
    x0, y0 = 0, 0
    x1, y1 = int(lx*0.25), int(ly*0.25)
    x2, y2 = int(lx*0.5),  int(ly*0.5)
    x3, y3 = int(lx*0.75), int(ly*0.75)
    x_list = [(x0,lx),(x1,x3),(x2,x2),(x3,x1),(lx,x0),(lx,x0),(lx,x0),(lx,x0)]
    y_list = [(y0,ly),(y0,ly),(y0,ly),(y0,ly),(y0,ly),(y1,y3),(y2,y2),(y3,y1)]
    return x_list, y_list

def get_lines_horiz(img):
    lx = img.shape[1]-1
    ly = img.shape[0]-1
    x0 = 0
    y2 = int(ly*0.2)
    y3 = int(ly*0.3)
    y4 = int(ly*0.4)
    y5 = int(ly*0.5)
    y6 = int(ly*0.6)
    y7 = int(ly*0.7)
    y8 = int(ly*0.8)
    x_list = [(x0,lx),(x0,lx),(x0,lx),(x0,lx),(x0,lx),(x0,lx),(x0,lx)]
    y_list = [(y2,y2),(y3,y3),(y4,y4),(y5,y5),(y6,y6),(y7,y7),(y8,y8)]
    return x_list, y_list

# Get local adjacency
#---------------------
def get_neighborhood(neighborhood):
    if neighborhood == 4:
        return [(-1,0), (0,1), (1,0), (0,-1)]
    if neighborhood == 8:
        return [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]


#---------------------------------
def ift_bg(img, current, neighborhood, G, Q):
    for i in neighborhood:
        n_pixel = (current.y() + i[0], current.x() + i[1])
        if not is_valid_pixel(img, n_pixel):
            continue
        # if G.is_visited(n_pixel):
        #     print("c:", current.pixel, "n:", n_pixel, "distance_c:", current.distance, "distance_n:", G.node[n_pixel].distance)
        if not G.is_visited(n_pixel):
            n = Node(n_pixel) if not G.has_node(n_pixel) else G.get_node(n_pixel)
            cost = np.log(int(img[n_pixel])/20.0) * 180
            if cost < n.cost:
                n.cost = cost
                n.distance = current.distance + 1
                G.set_arch(n, current, (0,0,255))
                Q.put(n, cost)

#---------------------------------
def ift_fg(img, current, neighborhood, G, Q):
    for i in neighborhood:
        n_pixel = (current.y() + i[0], current.x() + i[1])
        if not is_valid_pixel(img, n_pixel):
            continue
        if not G.is_visited(n_pixel):
            n = Node(n_pixel) if not G.has_node(n_pixel) else G.get_node(n_pixel)
            cost = 255 * np.exp(-int(img[n_pixel])/20.0)
            if cost < n.cost:
                n.cost = cost
                G.set_arch(n, current, (0,255,0))
                Q.put(n, cost)

#-------------------------
def is_valid_pixel(img, pixel):
    y = pixel[0]
    x = pixel[1]
    if y >= 0 and y < img.shape[0]:
        if x >= 0 and x < img.shape[1]:
            return True
    return False

#--------------------------
def find_csf(line_pts):
    accum  = 0
    window = [0 for i in range(5)]
    maxval = 0
    minval = sys.maxsize
    csf_left, csf_right = None, None
    for i in range(len(line_pts)//2):
        v3 = (3, line_pts[i+3] - line_pts[i])
        v2 = (2, line_pts[i+2] - line_pts[i])
        v1 = (1, line_pts[i+1] - line_pts[i])
        vr = (v3[0]+v2[0]+v1[0], v3[1]+v2[1]+v1[1])
        r  = np.arctan(vr[1]/vr[0])
        accum += r
        if maxval > 1.0 and maxval not in window:
            if accum < minval:
                minval = accum
        if accum > maxval:
            if minval < maxval:
                csf_left = i-1
                break
            maxval = accum
        window.pop(0)
        window.append(accum)
        #print("x:", i, "r:", r, "s:", accum, "max:", maxval, "min:", minval)
    accum  = 0
    window = [0 for i in range(5)]
    maxval = 0
    minval = sys.maxsize
    for i in range(len(line_pts)-1, len(line_pts)//2, -1):
        v3 = (3, line_pts[i-3] - line_pts[i])
        v2 = (2, line_pts[i-2] - line_pts[i])
        v1 = (1, line_pts[i-1] - line_pts[i])
        vr = (v3[0]+v2[0]+v1[0], v3[1]+v2[1]+v1[1])
        r  = np.arctan(vr[1]/vr[0])
        accum += r
        if maxval > 1.0 and maxval not in window:
            if accum < minval:
                minval = accum
        if accum > maxval:
            if minval < maxval:
                csf_right = i+1
                break
            maxval = accum
        window.pop(0)
        window.append(accum)
        #print("x:", i, "r:", r, "s:", accum, "max:", maxval, "min:", minval)
    return csf_left, csf_right



def main():
    #load data
    img3D    = nib.load("../3.nii.gz")
    img_data = img3D.get_data()
    max_val  = img_data[:,:,:].max()
    norm_img = np.uint8(img_data[:,:,:]*255.0/max_val)

    #extract slice from an axis
    for i in range(20, 220):
        #i=120
        #slc = norm_img[:, i, :]
        slc = norm_img[:, :, i]
        slc[slc==0] = 255
        cv2.imshow("teste", slc)
        cv2.imwrite("slice.png", slc)
        cv2.waitKey(0)
        teste = np.zeros(slc.shape, dtype="uint8")
        teste = cv2.cvtColor(teste, cv2.COLOR_GRAY2BGR)

        ret, thresh = cv2.threshold(slc, 0, 255, cv2.THRESH_OTSU)
        # cv2.imshow("teste", thresh)
        # cv2.waitKey(0)

        eroded = morphology.opening(thresh, morphology.diamond(1))
        eroded = skimage.img_as_ubyte(eroded)
        # cv2.imshow("teste", eroded)
        # cv2.waitKey(0)

        ret, label, stats, cent = cv2.connectedComponentsWithStats(eroded)
        areas = stats[1:,cv2.CC_STAT_AREA]
        arg = np.argmax(areas) + 1
        label[label != arg] = 0
        label[label == arg] = 1
        label = np.uint8(label)
        slc = cv2.multiply(slc, label)
        # cv2.imshow("teste", slc)
        # cv2.waitKey(0)
    # for i in range(1, len(stats)):
    #     width  = stats[i][cv2.CC_STAT_WIDTH]
    #     height = stats[i][cv2.CC_STAT_HEIGHT]
    #     area   = stats[i][cv2.CC_STAT_AREA]
    #
    #     print("i:", i, "area:", area)
    #     temp = np.zeros(label.shape)
    #     temp[label==i] = 255
    #     temp[label!=i] = 0
    #     cv2.imshow("teste", temp)
    #     cv2.waitKey(0)

    # xl, yl = get_lines_horiz(teste)
    # ellipse_pts = []
    # seeds = []
    # for i in range(len(xl)):
    #     #i=1
    #     temp  = np.zeros(slc.shape, dtype="uint8")
    #     cv2.line(temp, (xl[i][0],yl[i][0]), (xl[i][1],yl[i][1]), 255)
    #     # cv2.line(teste, (xl[i][0],yl[i][0]), (xl[i][1],yl[i][1]), 255)
    #     line = np.where(temp == 255)
    #     pts  = list(zip(line[0], line[1]))
    #     xlist= [i for i in range(len(pts))]
    #     I    = [int(slc[p]) for p in pts]
    #     left, right = find_csf(I)
    #     print(left, right)
    #     if left is not None:
    #         slc[yl[i][0],left] = 255
    #         ellipse_pts.append((left,yl[i][0]))
    #         seeds.append((yl[i][0],left))
    #     if right is not None:
    #         slc[yl[i][0],right] = 255
    #         ellipse_pts.append((right,yl[i][0]))
    #         seeds.append((yl[i][0],right))
    #
    # ellipse_pts = np.array(ellipse_pts)
    #
    # # ellipse = cv2.fitEllipse(ellipse_pts)
    # # cv2.ellipse(teste, ellipse, 255, 1)
    # # cv2.imshow("teste", teste)
    # # cv2.waitKey(0)
    # # plt.plot(xlist, I)
    # # plt.show()
    # # sys.exit()
    #
    #
    # G = Graph(teste)
    # Q_fg = PriorityQueue()
    # Q_bg = PriorityQueue()
    # n_fg = Node((144,56), 0)
    # #n_bg = Node((144,19), 0)
    # for p in seeds:
    #     n_bg = Node(p, 0)
    #     G.add_node(n_bg)
    #     Q_bg.put(n_bg, 0)
    # G.add_node(n_fg)
    # #G.add_node(n_bg)
    # Q_fg.put(n_fg, 0)
    # #Q_bg.put(n_bg, 0)
    # maxval = np.max(slc)
    # neighborhood = get_neighborhood(4)
    # temp_visited = set()
    # previous_area = 0
    #
    # while not Q_bg.empty():
    #     bg = Q_bg.pop()
    #     G.add_visited(bg.pixel)
    #     ift_bg(slc, bg, neighborhood, G, Q_bg)
    #     # cont = cv2.cvtColor(teste, cv2.COLOR_BGR2GRAY)
    #     #ret, cont = cv2.threshold(cont, 20, 255, cv2.THRESH_BINARY)
    #     #contours = cv2.findContours(cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    #     # epsilon = 0.01*cv2.arcLength(contours[0],True)
    #     # approx = cv2.approxPolyDP(contours[0],epsilon,True)
    #     cont = np.zeros(teste.shape, dtype="uint8")
    #     #cont = cv2.drawContours(cont, approx, -1, (0,255,0), 1)
    #     # if len(contours[0]) > 5:
    #     #     ellipse = cv2.fitEllipse(contours[0])
    #     #     cv2.ellipse(cont, ellipse, (0,255,0), 1)
    #     cv2.imshow("growing", teste)
    #     #cv2.imshow("contours", cont)
    #     cv2.waitKey(1)
    #
    # G.visited = set()
    # while not Q_fg.empty():
    #     fg = Q_fg.pop()
    #     G.add_visited(fg.pixel)
    #     ift_fg(slc, fg, neighborhood, G, Q_fg)
    #     cv2.imshow("growing", teste)
    #     cv2.waitKey(1)
    # cv2.imshow("growing", teste)
    # cv2.waitKey(0)
    #
    #
    #
    #
    # print("finished")






    #lx, ly = get_lines(slc)
    # for i in range(len(lx)):
    #     #i = 0
    #     img = np.zeros(slc.shape, dtype="uint8")
    #     tst = slc.copy()
    #     cv2.line(img, (lx[i][0], ly[i][0]), (lx[i][1], ly[i][1]), 255)
    #     cv2.line(tst, (lx[i][0], ly[i][0]), (lx[i][1], ly[i][1]), 255)
    #     line = np.where(img == 255)
    #     pts  = list(zip(line[0], line[1]))
    #
    #     x_list = [i for i in range(len(pts))]
    #     y_list = [slc[p] for p in pts]
    #
    #     cv2.imshow("teste", tst )
    #     cv2.waitKey(0)
    #     plt.plot(x_list, y_list)
    #     plt.show()






if __name__=="__main__":
    main()
