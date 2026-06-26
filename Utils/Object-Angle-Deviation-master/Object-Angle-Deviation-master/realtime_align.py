# import libraries
import cv2
import numpy as np
import math
import time
from scipy.spatial import distance as dist
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--angle', default=10, help='Enter allowed angle deviaiton')
ap.add_argument('--deviation', default=35,
                help='Enter allowed deviaiton from center')
ap.add_argument('--video', default=0,
                help='Enter video path')
args = vars(ap.parse_args())
print(args)


def obj_midpoint(pt1, pt2):
    return ((pt1 + pt2) // 2)


'''
function that returns an aligned image based on criteria 
input is file_name of video 
enter 0 for real time feed
'''


def object_align(video_ip):

    cap = cv2.VideoCapture(video_ip)

    while (cap.isOpened()):
        _, frame = cap.read()
        # frame = cv2.pyrDown(cv2.imread(video_ip, cv2.IMREAD_UNCHANGED))
        # frame = cv2.imread(video_ip, cv2.IMREAD_UNCHANGED)
        if(frame is None):
            cap.release()
            break
        img = frame.copy()
        cv2.pyrDown(img)
        md_pt_y = obj_midpoint(0, img.shape[0])

        md_pt_x = obj_midpoint(0, img.shape[1])

        # threshold image
        ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                          160, 255, cv2.THRESH_BINARY)

        # find contours
        contours, hier = cv2.findContours(
            threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

            deviation = dist.euclidean((cX, cY), (md_pt_x, md_pt_y))

            cv2.putText(img, "Deviation is " + str(deviation),
                        org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 255))

            # Angle calculation
            (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y) = box
            tltrX = obj_midpoint(tl_x, tr_x)
            tltrY = obj_midpoint(tl_y, tr_y)
            blbrX = obj_midpoint(bl_x, br_x)
            blbrY = obj_midpoint(bl_y, br_y)
            tlblX = obj_midpoint(tl_x, bl_x)
            tlblY = obj_midpoint(tl_y, bl_y)
            trbrX = obj_midpoint(tr_x, br_x)
            trbrY = obj_midpoint(tr_y, br_y)

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

            cv2.line(img, (cX, cY),
                     (md_pt_x, md_pt_y), thickness=2, color=(255, 255, 0))

            # Results of calculations can be printed in console or you can
            # write them on the frame itself
            print("Angle = " + str(Angle))
            print("Deviation = "+str(deviation))
            cv2.putText(img, "Angle is " + str(Angle),
                        org=(50, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 255))

            if(Angle < int(args['angle']) and deviation < int(args['deviation'])):
                # cv2.imshow('contours', img)

                latest_img = frame.copy()

                cv2.drawContours(
                    latest_img, [box], 0, (255, 0, 0), thickness=2)

                cv2.circle(latest_img, (cX, cY), 5, (127, 250, 169), -1)
                cv2.circle(latest_img, (md_pt_x, md_pt_y),
                           5, (127, 250, 169), -1)

                cv2.line(latest_img, (md_pt_x, 0),
                         (md_pt_x, img.shape[0]), thickness=2, color=(255, 255, 255))

                cv2.line(latest_img, (0, md_pt_y),
                         (img.shape[1], md_pt_y), thickness=2, color=(255, 255, 255))

                cv2.line(latest_img, (cX, cY),
                         (md_pt_x, md_pt_y), thickness=2, color=(255, 255, 0))

                cv2.putText(latest_img, "Deviation is " + str(deviation),
                            org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 255))
                cv2.putText(latest_img, "Angle is " + str(Angle),
                            org=(50, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 255))

                cv2.putText(latest_img, "Image aligned successfully. Press p to proceed.", org=(
                    50, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2, color=(0, 255, 255))

                cv2.imshow('contours', latest_img)
                while True:
                    key = cv2.waitKey(1)
                    if key == ord('p'):
                        return frame
                        break

            cv2.imshow("contours", img)

        # Interrupt video processing
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to break
            cap.release()
            break


if __name__ == "__main__":
    aligned_object = object_align(args['video'])
    if(aligned_object is None):
        print("Object not aligned properly, returned None \nPress esc to exit")
    else:
        cv2.imshow('Aligned Object', aligned_object)

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to break
            break
    cv2.destroyAllWindows()
