import cv2
import numpy as np
from scipy.spatial import distance as dist
import math
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--image', help='Enter image path', required=True)
args = vars(ap.parse_args())
print(args)


def midpoint(pt1, pt2):
    return ((pt1 + pt2) // 2)


img = cv2.pyrDown(cv2.imread(args['image'], cv2.IMREAD_UNCHANGED))

md_pt_y = midpoint(0, img.shape[0])
md_pt_x = midpoint(0, img.shape[1])

# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                  160, 255, cv2.THRESH_BINARY)

# find contours
contours, hier = cv2.findContours(
    threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if contours is None:
    print("No contours found !")


for c in contours:
    if cv2.contourArea(c) < 150 or cv2.contourArea(c) > 50000:
        continue

    print(str(cv2.contourArea(c)))
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)

    # draw a green rectangle to visualize the bounding rect
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)

    # convert all coordinates floating point values to int
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (255, 0, 0), thickness=2)

    # compute the center of the contour
    M = cv2.moments(box)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # draw center points
    cv2.circle(img, (cX, cY), 5, (127, 250, 169), -1)
    cv2.circle(img, (md_pt_x, md_pt_y), 5, (127, 250, 169), -1)

    cv2.putText(img, "Deviation is " + str(dist.euclidean((cX, cY), (md_pt_x, md_pt_y))),
                org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 255))

    # Angle calculation
    (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = box
    tltrX = midpoint(tl_x, tr_x)
    tltrY = midpoint(tl_y, tr_y)
    blbrX = midpoint(bl_x, br_x)
    blbrY = midpoint(bl_y, br_y)
    tlblX = midpoint(tl_x, bl_x)
    tlblY = midpoint(tl_y, bl_y)
    trbrX = midpoint(tr_x, br_x)
    trbrY = midpoint(tr_y, br_y)

    # compute the Euclidean distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    Angle = 0
    if(dA >= dB):
        rp1_x = tltrX
        rp1_y = tltrY
        rp2_x = blbrX
        rp2_y = blbrY
    else:
        rp1_x = tlblX
        rp1_y = tlblY
        rp2_x = trbrX
        rp2_y = trbrY

    gradient = (rp2_y - rp1_y)*1.0/(rp2_x - rp1_x)*1.0
    Angle = math.atan(gradient)
    Angle = Angle*57.2958

    if(Angle < 0):
        Angle = Angle + 180

cv2.line(img, (md_pt_x, 0),
         (md_pt_x, img.shape[0]), thickness=2, color=(255, 255, 255))

cv2.line(img, (0, md_pt_y),
         (img.shape[1], md_pt_y), thickness=2, color=(255, 255, 255))

if cX is not None and cY is not None:
    cv2.line(img, (cX, cY),
             (md_pt_x, md_pt_y), thickness=2, color=(255, 255, 0))

# Results of calculations can be printed in console or you can
# write them on the frame itself
print("Angle = " + str(Angle))

cv2.putText(img, "Angle is " + str(Angle),
            org=(50, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 255))

cv2.imshow("contours", img)
# cv2.imwrite('output6.png', img)
while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to break
        break

cv2.destroyAllWindows()
