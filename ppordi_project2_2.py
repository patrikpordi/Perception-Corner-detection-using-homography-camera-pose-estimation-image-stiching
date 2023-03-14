import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import Image
# Functions

# Filtering the matches based on the distance error
def filter(matches):
    good = []
    for m in matches:
        if (m[0].distance < 0.3*m[1].distance):
            good.append(m)
    matches = np.asarray(good)
    return matches

# Compute homography between adjacent images if enough good matches are found
def homography(matches,kp1,kp2):
    if (len(matches[:,0]) >= 4):
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
        A=[]
        for i in range(len(src)):
            x,y=src[i][0][0], src[i][0][1]
            x_d,y_d=dst[i][0][0], dst[i][0][1]
            A.append([x,y,1,0,0,0,-x_d*x,-x_d*y,-x_d])
            A.append([0,0,0,x,y,1,-y_d*x,-y_d*y,-y_d])
        A=np.array(A)
        # Singular Value Decomposition (SVD)
        U, S, V = np.linalg.svd(A)
        # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
        # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
        H = np.reshape(V[-1], (3, 3))
        # Normalization
        H = (1 / H.item(8)) * H
        return H

# Attaching the pictures from right to left, so the inverse H is required
def attach(img_left,img_right,H_left_right):
    width=img_right.shape[1]+img_left.shape[1]
    height=min(img_left.shape[0],img_right.shape[0])
    result=cv.warpPerspective(img_right,np.linalg.inv(H_left_right),(width,height))
    result[0:img_left.shape[0],0:img_left.shape[1]]=img_left
    return result

# Read input images
img_1 = cv.imread('image_1.jpg')
img_2 = cv.imread('image_2.jpg')
img_3 = cv.imread('image_3.jpg')
img_4 = cv.imread('image_4.jpg')

# Initialize SIFT object
sift = cv.xfeatures2d.SIFT_create()

# Compute keypoints and descriptors for each input image
kp1, des1 = sift.detectAndCompute(img_1,None)
kp2, des2 = sift.detectAndCompute(img_2,None)
kp3, des3 = sift.detectAndCompute(img_3,None)
kp4, des4 = sift.detectAndCompute(img_4,None)

# Initialize FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Perform matching between adjacent images
matches_12= flann.knnMatch(des1,des2,k=2)
matches_23= flann.knnMatch(des2,des3,k=2)
matches_34= flann.knnMatch(des3,des4,k=2)

# Filter matches using Lowe's ratio test and obtain good matches
matches_12= filter(matches_12)
matches_23= filter(matches_23)
matches_34= filter(matches_34)


# Compute homography between adjacent images if enough good matches are found
H_12=homography(matches_12,kp1,kp2)
H_23=homography(matches_23,kp2,kp3)
H_34=homography(matches_34,kp3,kp4)

# cv.drawMatchesKnn expects list of lists as matches.
matches_12_visualize = cv.drawMatchesKnn(img_1,kp1,img_2,kp2,matches_12[:200],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matches_12_visualize)
plt.show()

# cv.drawMatchesKnn expects list of lists as matches.
matches_23_visualize = cv.drawMatchesKnn(img_2,kp2,img_3,kp3,matches_23[:200],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matches_23_visualize)
plt.show()


# cv.drawMatchesKnn expects list of lists as matches.
matches_34_visualize = cv.drawMatchesKnn(img_3,kp3,img_4,kp4,matches_34[:200],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(matches_34_visualize)
plt.show()


img_34=attach(img_3,img_4,H_34)
plt.imshow(img_34)
plt.show()


img_234=attach(img_2,img_34,H_23)
plt.imshow(img_234)
plt.show()


img_1234=attach(img_1,img_234,H_12)
plt.imshow(img_1234)
plt.show()


cv.imwrite('output.png',img_1234)

