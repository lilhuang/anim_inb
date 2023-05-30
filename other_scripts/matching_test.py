import cv2
import os
import numpy as np
import math
from scipy.spatial.distance import cdist, cosine
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

import pdb


class ShapeContext(object):

    def __init__(self, nbins_r=5, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
        # number of radius zones
        self.nbins_r = nbins_r
        # number of angles zones
        self.nbins_theta = nbins_theta
        # maximum and minimum radius
        self.r_inner = r_inner
        self.r_outer = r_outer

    def _hungarian(self, cost_matrix):
        """
            Here we are solving task of getting similar points from two paths
            based on their cost matrixes. 
            This algorithm has dificulty O(n^3)
            return total modification cost, indexes of matched points
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total = cost_matrix[row_ind, col_ind].sum()
        indexes = zip(row_ind.tolist(), col_ind.tolist())
        return total, indexes
    
    def get_points_from_img(self, image, simpleto=100):
        """
            This is much faster version of getting shape points algo.
            It's based on cv2.findContours algorithm, which is basically return shape points
            ordered by curve direction. So it's gives better and faster result
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cnts
        points = []
        for contour in contours:
            points.extend(contour)
        points = np.squeeze(np.array(points), axis=1)
        # if len(cnts[1]) > 1:
        #     points = np.concatenate([points, np.array(cnts[1][1]).reshape((-1, 2))], axis=0)
        points = points.tolist()
        step = len(points) // simpleto
        points = [points[i] for i in range(0, len(points), step)][:simpleto]
        if len(points) < simpleto:
            points = points + [[0, 0]] * (simpleto - len(points))
        
        return points

    '''def get_points_from_img(self, image, threshold=50, simpleto=100, radius=2):
        """
            That is not very good algorithm of choosing path points, but it will work for our case.
            Idea of it is just to create grid and choose points that on this grid.
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = cv2.Canny(image, threshold, threshold * 3, 3)
        py, px = np.gradient(image)
        # px, py gradients maps shape can be smaller then input image shape
        points = [index for index, val in np.ndenumerate(dst)
                  if val == 255 and index[0] < py.shape[0] and index[1] < py.shape[1]]
        h, w = image.shape
        _radius = radius
        while len(points) > simpleto:
            newpoints = points
            xr = range(0, w, _radius)
            yr = range(0, h, _radius)
            for p in points:
                if p[0] not in yr and p[1] not in xr:
                    newpoints.remove(p)
                    if len(points) <= simpleto:
                        T = np.zeros((simpleto, 1))
                        for i, (y, x) in enumerate(points):
                            radians = math.atan2(py[y, x], px[y, x])
                            T[i] = radians + 2 * math.pi * (radians < 0)
                        return points, np.asmatrix(T)
            _radius += 1
        T = np.zeros((simpleto, 1))
        for i, (y, x) in enumerate(points):
            radians = math.atan2(py[y, x], px[y, x])
            T[i] = radians + 2 * math.pi * (radians < 0)
        return points, np.asmatrix(T)'''

    def _cost(self, hi, hj):
        cost = 0
        for k in range(self.nbins_theta * self.nbins_r):
            if (hi[k] + hj[k]):
                cost += ((hi[k] - hj[k])**2) / (hi[k] + hj[k])

        return cost * 0.5

    def cost_by_paper(self, P, Q, qlength=None):
        p, _ = P.shape
        p2, _ = Q.shape
        d = p2
        if qlength:
            d = qlength
        C = np.zeros((p, p2))
        for i in range(p):
            for j in range(p2):
                C[i, j] = self._cost(Q[j] / d, P[i] / p)

        return C

    def compute(self, points):
        """
          Here we are computing shape context descriptor
        """
        t_points = len(points)
        # getting euclidian distance
        r_array = cdist(points, points)
        # getting two points with maximum distance to norm angle by them
        # this is needed for rotation invariant feature
        am = r_array.argmax()
        max_points = [am // t_points, am % t_points]
        # normalizing
        r_array_n = r_array / r_array.mean()
        # create log space
        r_bin_edges = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.nbins_r)
        r_array_q = np.zeros((t_points, t_points), dtype=int)
        # summing occurences in different log space intervals
        # logspace = [0.1250, 0.2500, 0.5000, 1.0000, 2.0000]
        # 0    1.3 -> 1 0 -> 2 0 -> 3 0 -> 4 0 -> 5 1
        # 0.43  0     0 1    0 2    1 3    2 4    3 5
        for m in range(self.nbins_r):
            r_array_q += (r_array_n < r_bin_edges[m])

        fz = r_array_q > 0

        # getting angles in radians
        theta_array = cdist(points, points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
        norm_angle = theta_array[max_points[0], max_points[1]]
        # making angles matrix rotation invariant
        theta_array = (theta_array - norm_angle * (np.ones((t_points, t_points)) - np.identity(t_points)))
        # removing all very small values because of float operation
        theta_array[np.abs(theta_array) < 1e-7] = 0

        # 2Pi shifted because we need angels in [0,2Pi]
        theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0)
        # Simple Quantization
        theta_array_q = (1 + np.floor(theta_array_2 / (2 * math.pi / self.nbins_theta))).astype(int)

        # building point descriptor based on angle and distance
        nbins = self.nbins_theta * self.nbins_r
        descriptor = np.zeros((t_points, nbins))
        for i in range(t_points):
            sn = np.zeros((self.nbins_r, self.nbins_theta))
            for j in range(t_points):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            descriptor[i] = sn.reshape(nbins)

        return descriptor

    def cosine_diff(self, P, Q):
        """
            Fast cosine diff.
        """
        P = P.flatten()
        Q = Q.flatten()
        assert len(P) == len(Q), 'number of descriptors should be the same'
        return cosine(P, Q)

    def diff(self, P, Q, qlength=None):
        """
            More precise but not very speed efficient diff.
            if Q is generalized shape context then it compute shape match.
            if Q is r point representative shape contexts and qlength set to 
            the number of points in Q then it compute fast shape match.
        """
        result = None
        C = self.cost_by_paper(P, Q, qlength)

        result = self._hungarian(C)

        return result

    @classmethod
    def tests(cls):
        # basics tests to see that all algorithm invariants options are working fine
        self = cls()

        def test_move():
            p1 = np.array([
                [0, 100],
                [200, 60],
                [350, 220],
                [370, 100],
                [70, 300],
            ])
            # +30 by x
            p2 = np.array([
                [0, 130],
                [200, 90],
                [350, 250],
                [370, 130],
                [70, 330]
            ])
            c1 = self.compute(p1)
            c2 = self.compute(p2)
            assert (np.abs(c1.flatten() - c2.flatten())
                    ).sum() == 0, "Moving points in 2d space should give same shape context vector"

        def test_scale():
            p1 = np.array([
                [0, 100],
                [200, 60],
                [350, 220],
                [370, 100],
                [70, 300],
            ])
            # 2x scaling
            p2 = np.array([
                [0, 200],
                [400, 120],
                [700, 440],
                [740, 200],
                [149, 600]
            ])
            c1 = self.compute(p1)
            c2 = self.compute(p2)
            assert (np.abs(c1.flatten() - c2.flatten())
                    ).sum() == 0, "Scaling points in 2d space should give same shape context vector"

        def test_rotation():
            p1 = np.array(
                [(144, 196), (220, 216), (330, 208)]
            )
            # 90 degree rotation
            theta = np.radians(90)
            c, s = np.cos(theta), np.sin(theta)
            R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
            p2 = np.dot(p1, R).tolist()

            c1 = self.compute(p1)
            c2 = self.compute(p2)
            assert (np.abs(c1.flatten() - c2.flatten())
                    ).sum() == 0, "Rotating points in 2d space should give same shape context vector"

        test_move()
        test_scale()
        test_rotation()
        print('Tests PASSED')


def save_shape_context_matches_img(img1, img2, points1, points2, indices):
    fig, axs = plt.subplots(1, 3, figsize=(40,20))
    axs[0].imshow(img1, cmap="gray")
    for point in points1:
        axs[0].plot(point[0], point[1], marker="o")
    axs[1].imshow(img2, cmap="gray")
    for point in points2:
        axs[1].plot(point[0], point[1], marker="o")
    axs[2].imshow(img1, cmap="gray", alpha=0.5)
    axs[2].imshow(img2, cmap="gray", alpha=0.5)
    color_arr = np.array([[1, 0, 0], [0, 1, 0]])
    for i in range(len(indices)):
        index1 = indices[i][0]
        index2 = indices[i][1]
        x_arr = np.array([points1[index1][0],points2[index2][0]])
        y_arr = np.array([points1[index1][1],points2[index2][1]])
        axs[2].scatter(x_arr,y_arr,color=color_arr,marker=".")
        axs[2].plot(x_arr,y_arr,color="black",marker=None,linewidth=0.5)
    # asp = np.diff(axs[2].get_xlim())[0] / np.diff(axs[2].get_ylim())[0]
    # asp /= np.abs(np.diff(axs[0].get_xlim())[0] / np.diff(axs[0].get_ylim())[0])
    # axs[2].set_aspect(asp)
    plt.savefig("yoongiToT_coarse.jpg")
    plt.clf()


def contour_test():
    image1_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_2048x1024_t_3_k_3/frame1.jpg"
    image3_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_2048x1024_t_3_k_3/frame3.jpg"

    image1small_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_256x128_t_3_k_3/frame1.jpg"

    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(image3_path, cv2.IMREAD_GRAYSCALE)

    cnts1, hierarchy1 = cv2.findContours(img1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts3, hierarchy3 = cv2.findContours(img3, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    plt.clf()
    blur = cv2.GaussianBlur(img1, (15,15),0)
    cv2.imwrite("yoongi_blur.jpg",blur)
    ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh, "gray", vmin=0, vmax=255)
    plt.savefig("yoongi_thresh.jpg")

    plt.clf()
    blur3 = cv2.GaussianBlur(img3, (15,15),0)
    cv2.imwrite("yoongi_blur3.jpg",blur3)
    ret3, thresh3 = cv2.threshold(blur3, 150, 255, cv2.THRESH_BINARY)
    plt.imshow(thresh3, "gray", vmin=0, vmax=255)
    plt.savefig("yoongi_thresh3.jpg")

    contours1, h1 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours3, h3 = cv2.findContours(thresh3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    count3 = 0

    for i in range(len(contours1)):
        count += len(contours1[i])
    for i in range(len(contours3)):
        count3 += len(contours3[i])

    pdb.set_trace()

    # longest = 0
    # index = 0
    # for i in range(len(contours1)):
    #     if len(contours1[i]) > longest:
    #         longest = len(contours1[i])
    #         index = i
    # pdb.set_trace()

    plt.clf()
    cur_cont = np.transpose(np.squeeze(contours1[40], axis=1))
    colors = np.arange(len(contours1[40]))
    # plt.imshow(img1, cmap="gray",alpha=0.5)
    plt.scatter(cur_cont[0], cur_cont[1],c=colors,s=1,cmap="gray")
    plt.savefig("whydouonlyloveyoongi.jpg")
    
    
    empty = np.ones(img1.shape)*255
    # taetae = cv2.drawContours(empty, contours1, 40, (0,255,0),1)
    taetae = cv2.drawContours(empty, contours1, -1, (0,255,0),1)
    cv2.imwrite("yoontae_4.jpg", taetae)

    



def shape_context():
    sc = ShapeContext(nbins_r=10, nbins_theta=24, r_inner=0.25, r_outer=4.0)

    image1_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_2048x1024_t_3_k_3/frame1.jpg"
    image3_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_2048x1024_t_3_k_3/frame3.jpg"
    # image1_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_256x128_t_3_k_3/frame1.jpg"
    # image3_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_256x128_t_3_k_3/frame3.jpg"

    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(image3_path, cv2.IMREAD_GRAYSCALE)

    points1 = sc.get_points_from_img(img1)
    points3 = sc.get_points_from_img(img3)

    desc1 = sc.compute(points1)
    desc3 = sc.compute(points3)

    match, indices = sc.diff(desc1, desc3, qlength=len(desc3))
    indices = list(indices)

    save_shape_context_matches_img(img1, img3, points1, points3, indices)



def sift():
    image1_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_2048x1024_t_3_k_3/frame1.jpg"
    image3_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_2048x1024_t_3_k_3/frame3.jpg"
    # image1_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_256x128_t_3_k_3/frame1.jpg"
    # image3_path = "/fs/vulcan-projects/anim_inb_lilhuang/datasets/test_2k_original_preprocess_dog/Disney_v4_12_001856_s2_256x128_t_3_k_3/frame3.jpg"

    img1 = cv2.imread(image1_path)
    img3 = cv2.imread(image3_path)

    #sift
    sift = cv2.SIFT_create(nOctaveLayers=1)

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_3, descriptors_3 = sift.detectAndCompute(img3,None)

    #feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_3)
    matches = sorted(matches, key = lambda x:x.distance)

    print("number of matches:", len(matches))

    img_match = cv2.drawMatches(img1, keypoints_1, img3, keypoints_3, matches[:50], img3, flags=2)
    cv2.imwrite("yoongi_is_29_1.jpg", img_match)
    # plt.imshow(img_match),plt.show()





if __name__ == "__main__":
    # sift()
    # ShapeContext.tests()
    # shape_context()
    contour_test()

