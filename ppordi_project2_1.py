import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from math import nan
from IPython.display import Image
from matplotlib import cm
from sklearn import datasets
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist
from IPython.display import Image

# Functions
def h_lines(edges):
    # An accumulator array has to be created where the each index respresents
    # A r,theta pair, the origin is in the middle of the frame
    # Using the rows and columns from the frame the maximal diagonal length is achieved
    # 0.5 is added to make sure that the end result is gonna be accurate
    diag = int(math.sqrt(edges.shape[0] ** 2 + edges.shape[1] ** 2) + 0.5)
    # Theta resolution in radians, r_res=1
    r_range=np.arange(-diag,diag,1)
    theta_range=np.arange(0,np.pi,np.pi/180)

    # Accumolator matrix, the dimensions are based on the r,theta range
    # The actual values are calculated based on the edge detection
    accumulator = np.zeros((len(r_range), len(theta_range)), dtype=np.uint64)
    cos = lambda x: np.cos(x)
    sin = lambda x: np.sin(x)

    for y,x in np.argwhere(edges):
        for j, theta in enumerate(theta_range):
            r = int(round(x * cos(theta) + y * sin(theta))) + diag
            accumulator[r, j] += 1

    
    # K-means to find the four local maximas, or centers in the accumolator matrix in our case
    data = (np.argwhere(accumulator > 40))
    # Clustering using 4 clusters and K-means, extracting the coordinates at the end
    k_means = KMeans(n_clusters=4,n_init=10)
    k_means.fit(data)
    centers = k_means.cluster_centers_

    lines = []
    for r, theta in centers:
        # Calculating the r,thetha pairs and the corresponding lines
        r -=diag
        theta = np.deg2rad(theta)
        x0, y0 = cos(theta) * r, sin(theta) * r
        x1, y1 = int(round(x0 + 10000 * (-sin(theta)))), int(round(y0 + 10000 * cos(theta)))
        x2, y2 = int(round(x0 - 10000 * (-sin(theta)))), int(round(y0 - 10000 * cos(theta)))
        lines.append([(x1, y1), (x2, y2)])
    return lines

def Homography(lines):
    # Create an empty list to store the intersection points
    intersections = []
    
    # Pairing the lines
    for i, line1 in enumerate(lines):
        for line2 in lines[i+1:]:
            
            # Endpoints of the two lines
            x1, y1 = line1[0]
            x2, y2 = line1[1]
            x3, y3 = line2[0]
            x4, y4 = line2[1]
            
            # Determinant for the equation
            determinant = (y4 - y3)*(x2 - x1) - (x4 - x3)*(y2 - y1)
            
            # In case that the lines are not parallel
            if determinant != 0:
                    
                A = ((x4 - x3)*(y1 - y3) - (y4 - y3)*(x1 - x3)) / determinant
                B = ((x2 - x1)*(y1 - y3) - (y2 - y1)*(x1 - x3)) / determinant
                
                # Check if the intersection point is inside both line segments
                if 0 <= A <= 1 and 0 <= B <= 1:
                    
                    # Intersection
                    intersection = (int(x1 + A*(x2 - x1)), int(y1 + A*(y2 - y1)))
                    intersections.append(intersection)
                    intersections.sort()

    

    # Convert the list of intersection points to a numpy array
    dst_points = np.array(intersections)
    
    # Keep only the points where both x and y are non-negative
    dst_points = dst_points[(dst_points >= 0).all(axis=1)]
    
    # Coordinates in real world in cm
    src_points=np.array([[0,21.6],[0,0],[27.9,21.6],[27.9,0]])

    K=[]
    for i in range(len(src_points)):
        x,y=src_points[i][0], src_points[i][1]
        x_d,y_d=dst_points[i][0], dst_points[i][1]
        K.append([x,y,1,0,0,0,-x_d*x,-x_d*y,-x_d])
        K.append([0,0,0,x,y,1,-y_d*x,-y_d*y,-y_d])
    K=np.array(K)
     # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(K)
    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
    H = np.reshape(V[-1], (3, 3))
    # Normalization
    H = (1 / H.item(8)) * H
    
    # return the array of intersection points
    return H, dst_points

def compute_camera_pose(H):
    # Define camera intrinsics matrix
    res=4
    fx, fy, cx, cy = 1382.58/res, 1383.57/res, 945.74/res, 527.05/res
    K = np.array([[fx, 0.00, cx],
                  [0.00, fy, cy],
                  [0.00, 0.00, 1.00]])


    # Compute the matrix Left_side
    Left_side = np.matmul(np.linalg.inv(K), H)

    # Compute the scale factor lambda
    l = np.linalg.norm(Left_side[:, :2]) / np.sqrt(2)

    # Compute the rotation matrix R
    r1 = Left_side[:, 0] / l
    r2 = Left_side[:, 1] / l
    r3 = np.cross(r1, r2)
    R = np.column_stack((r1, r2, r3))

    # Compute the translation vector t
    t = Left_side[:, 2] / l

    # Compute the norm of the translation vector and append it to a list
    norm_t = np.linalg.norm(t)
    translation = [norm_t]

    # Compute the yaw, pitch, and roll angles and append them to their respective lists
    yaw = [np.arctan2(R[1, 0], R[0, 0])]
    pitch = [np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))]
    roll = [np.arctan2(R[2, 1], R[2, 2])]

    # Return the rotation matrix R, translation vector t, and the lists of angles and translation norms
    return R, pitch, roll, yaw, translation


roll=[]
pitch=[]
yaw=[]
translation=[]
R=[]
# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv.VideoCapture('project2.avi')
if (vid_capture.isOpened() == False):
  print("Error opening the video file")
# Read fps and frame count
else:
  # Get frame rate information
  # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
  fps = vid_capture.get(5)
  #print('Frames per second : ', fps,'FPS')
 
  # Get frame count
  # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
  frame_count = vid_capture.get(7)
  #print('Frame count : ', frame_count)
  lst=[]

while(vid_capture.isOpened()):
    # vid_capture.read() methods returns a tuple, first element is a bool 
    # and the second is frame
    ret, frame = vid_capture.read()
    
    if ret == True:
        frame = cv.resize(frame, (480, 270))
        
        # Bluring, conversion to hsv, creating limits, creating the mask, creating the results using the mask
        blurred = cv.GaussianBlur(frame, (11, 11), 0)
        grey=cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(grey, threshold1=130, threshold2=150)

        lines=h_lines(edges)
        for line in lines:
          cv.line(frame, line[0], line[1], (255, 255, 255), 1)
        H,Intersections=Homography(lines)

        a,b,c,d,e=compute_camera_pose(H)
        R.append(a)
        roll.append(b)
        pitch.append(c)
        yaw.append(d)
        translation.append(e)

        for point in Intersections:
          cv.circle(frame, point, 3, (0, 255, 255), -1)
        cv.imshow('Lines', frame)

        key = cv.waitKey(30)
        if key == ord('q'):
            break
    else:
        break
 
# Release the video capture object
vid_capture.release()
cv.destroyAllWindows()


# Plotting the results
# Rotations
t=[x for x in range(len(translation))]
plt.title("Rotation")
plt.xlabel("Time [t]")
plt.ylabel("Angles [rad]")
plt.plot(t,roll,color='red',label='roll')
plt.plot(t,pitch,color='green',label='pitch')
plt.plot(t,yaw,color='blue',label='yaw')
plt.legend(loc='upper right')
plt.show()

# Translation
plt.title("Translation")
plt.xlabel("Time [t]")
plt.ylabel("Distance [cm]")
plt.plot(t,translation,color='green')
plt.show()