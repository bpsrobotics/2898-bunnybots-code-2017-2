import cv2
from findCircleNShit import *
from colorsys import hsv_to_rgb
import time
from scipy import optimize
import numpy as np
import scipy as sp
import math

def pipeline(points):
    pointDistanceFilter = 130.0/4096.0
    normalAngleFilter = 0.04
    pointWalkDistanceFilter = 0.01

    frame = np.zeros((512, 512, 3))

    # Sets of points
    #[[[t, r], [t, r]], [[],[]]]
    discreteObjects = []

    def polarDist(point1, point2):
        p1x = point1[1] * math.cos(point1[0])
        p1y = point1[1] * math.sin(point1[0])
        p2x = point2[1] * math.cos(point2[0])
        p2y = point2[1] * math.sin(point2[0])
        return math.sqrt((p2x-p1x)**2 + (p2y-p1y)**2)

    def nothing(x):
        pass

    # cv2.namedWindow('image')
    # cv2.createTrackbar('thingy','image',0,512,nothing)

    discreteObjects = []
    frame = np.zeros((512, 512, 3))
    # print("loop")
    # margin = cv2.getTrackbarPos('thingy', 'image')/4096.0
    queue = [points[0]]
    tardQueue = [points[0]]
    for i in points[1:]:
        dist = polarDist(tardQueue[-1], i)
        # print(dist)
        if dist >= pointDistanceFilter:
            # queue.append(i)
            pass
        else:
            discreteObjects.append(queue)
            queue = [i]
        tardQueue.append(i)
    if len(queue) > 0:
        discreteObjects.append(queue)

    filteredPoints = [x[0] for x in [y for y in discreteObjects]]

    # discreteObjects now contains a filtered point set or whatever, I don't even know
    # Basically all clumps of points there are in theory hella seperated
    # So now we have to actually bin everything ig

    # for i in range(len(discreteObjects)):
    #     color = (1.0/len(discreteObjects)) * i
    #     colorRGB = [int(x*255) for x in hsv_to_rgb(color, 1, 1)]
    #     # print(len(discreteObjects[i]))
    #     for j in discreteObjects[i]:
    #         angle = j[0]
    #         distance = j[1]*200
    #         x = distance * math.cos(angle)
    #         y = distance * math.sin(angle) + 256
    #         frame = cv2.circle(frame, (int(x), int(y)), 1, colorRGB, -1)
    for i in filteredPoints:
        angle = i[0]
        distance = i[1]*200
        x = distance*math.cos(angle)
        y = distance*math.sin(angle) + 256
        frame = cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

    foundNormals = getNormalPoints(filteredPoints)

    filteredNormals = []

    for i in foundNormals:
        # [normalSlope,
        if not abs(i[1]) <= normalAngleFilter:
            continue
        # if not abs(i[1]) <= 1:
        #     continue
        filteredNormals.append(i)

        angle = i[2][0]
        distance = i[2][1]*200
        x = distance * math.cos(angle)
        y = distance * math.sin(angle) + 256
        frame = cv2.arrowedLine(frame, (0, 256), (int(x), int(y)), (0, 255, 0), 1, 8, 0, 0.05)

    segmentedRegionProposals = []


    # pointWalkDistanceFilter = cv2.getTrackbarPos('thingy', 'image')/4096.0
    for i in filteredNormals:
        try:
            leftPointIndex = i[3]
            rightPointIndex = i[4]

            queue = []

            # walk left
            currentPoint1 = filteredPoints[leftPointIndex]
            currentPoint2 = filteredPoints[leftPointIndex-1]
            while polarDist(currentPoint1, currentPoint2) <= pointWalkDistanceFilter and leftPointIndex > 1:
                currentPoint1 = filteredPoints[leftPointIndex]
                currentPoint2 = filteredPoints[leftPointIndex-1]
                #print("distance", leftPointIndex, polarDist(currentPoint1, currentPoint2))
                queue.append(leftPointIndex)
                leftPointIndex -= 1

            # walk right
            currentPoint1 = filteredPoints[rightPointIndex]
            currentPoint2 = filteredPoints[rightPointIndex+1]
            while polarDist(currentPoint1, currentPoint2) <= pointWalkDistanceFilter and rightPointIndex < len(filteredPoints)-1:
                currentPoint1 = filteredPoints[rightPointIndex]
                currentPoint2 = filteredPoints[rightPointIndex+1]
                #print("distance", rightPointIndex, polarDist(currentPoint1, currentPoint2))
                queue.append(rightPointIndex)
                rightPointIndex += 1

        except:
            print("continuing")
            continue
        segmentedRegionProposals.append(queue)

    # print(len(segmentedRegionProposals))
    # print(segmentedRegionProposals[3])

    actualPoints = []
    for j in segmentedRegionProposals:
        for i in j:
            actualPoints.append(filteredPoints[i])
    # print("Unmerged", len(segmentedRegionProposals))

    # print("prop", segmentedRegionProposals)

    segmentedMergedRegionProposals = []
    # queue = []
    # for j in range(len(segmentedRegionProposals)):
    #     # print("j", sorted(segmentedRegionProposals[j]))
    #     # print("queue", sorted(queue))
    #     # If there's an intersection
    #     # if sorted(queue)[-1] >= sorted(segmentedRegionProposals[j])

    #     if not any(i in queue for i in segmentedRegionProposals[j]):
    #         [queue.append(x) for x in segmentedRegionProposals[j]]
    #     else:
            #segmentedMergedRegionProposals.append(queue)

    listLen = len(segmentedRegionProposals)
    index = 0
    while index < listLen-1:
        if setsOverlap(segmentedRegionProposals[index], segmentedRegionProposals[index+1]):
            for i in segmentedRegionProposals[index+1]:
                segmentedRegionProposals[index].append(i)
            segmentedRegionProposals.pop(index+1)
        else:
            index += 1
        listLen = len(segmentedRegionProposals)

    # print("Merged", len(segmentedRegionProposals))

    # print([i for i in [j for j in segmentedRegionProposals]])

    # actualPoints = []

    # frame = np.zeros((512, 512, 3))

    for i in actualPoints:
        # [normalSlope,>>> any(i in l1 for i in l2)

        # if not abs(i[1]) <= 1:
        #     continue
        angle = i[0]
        distance = i[1]*200
        x = distance * math.cos(angle)
        y = distance * math.sin(angle) + 256
        frame = cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), -1)


    #[[x, y, r, residue], [x, y, r, residue]]
    circles = []

    # First we find the closest point and then the two on either extremity of the set
    # Then we attempt to fit a circle to the points
    for i in segmentedRegionProposals:
        # realPoints = []
        realPoints = [filteredPoints[x] for x in i]

        #this shit's useless cause i'm retarded. readd if you want pretty circles or w/e

        # middlePoint = realPoints[np.argmax([x[1] for x in realPoints])]  # easier than figuring out numpy axis
        # leftPoint = realPoints[0]
        # rightPoint = realPoints[-1]
        # print("Left", leftPoint, "Middle", middlePoint, "Right", rightPoint)

        # for k in [leftPoint, middlePoint, rightPoint]:
        #     frame = cv2.circle(frame,
        #                        (
        #                            int(k[1] * math.cos(k[0]) * 200),
        #                            int(k[1] * math.sin(k[0]) * 200 + 256)
        #                        ), 3, (255, 0, 255), -1)
        # # now we fit a circle to the points mfers
        # leftPointCart = [leftPoint[1] * math.cos(leftPoint[0]), leftPoint[1] * math.sin(leftPoint[0])]
        # middlePointCart = [middlePoint[1] * math.cos(middlePoint[0]), middlePoint[1] * math.sin(middlePoint[0])]
        # rightPointCart = [rightPoint[1] * math.cos(rightPoint[0]), rightPoint[1] * math.sin(rightPoint[0])]

        cartesianPointSet = [[x[1]*math.cos(x[0]), x[1]*math.sin(x[0])] for x in realPoints]
        # print(cartesianPointSet)
        # print("len: ", len(cartesianPointSet))
        x = [i[0] for i in cartesianPointSet]
        y = [i[1] for i in cartesianPointSet]
        # print("x: ", x)
        x_m = np.mean(x)
        y_m = np.mean(y)
        # print("mean: ", x_m)
        u = x - x_m
        v = y - y_m
        # linear system defining the center (uc, vc) in reduced coordinates:
        #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
        #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
        Suv = sum(u * v)
        Suu = sum(u ** 2)
        Svv = sum(v ** 2)
        Suuv = sum(u ** 2 * v)
        Suvv = sum(u * v ** 2)
        Suuu = sum(u ** 3)
        Svvv = sum(v ** 3)
        A = np.array([[Suu, Suv], [Suv, Svv]])
        B = np.array([Suuu + Suvv, Svvv + Suuv]) / 2.0
        try:
            uc, vc = sp.linalg.solve(A, B)
        except:
            continue

        def calc_R(xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x-xc)**2 + (y-yc)**2)

        def f_2(c):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(*c)
            return Ri - Ri.mean()

        center_estimate = x_m, y_m
        l = x_m, y_m, x, y
        center_2, ier = sp.optimize.leastsq(f_2, center_estimate)

        xc_2, yc_2 = center_2
        Ri_2 = calc_R(*center_2)
        R_2 = Ri_2.mean()
        residu_2 = sum((Ri_2 - R_2) ** 2)

        xc, yc = xc_2, yc_2
        r = R_2
        err = residu_2
        # Center of circle is (xc, yc)
        # Radius of circle is r

        xc *= 200
        yc *= 200
        yc += 256
        r *= 200
        # print(xc, yc, r, residu_2)
        frame = cv2.circle(frame, (int(xc), int(yc)), int(r), (255, 255, 0), 1)

        # print(cartesianPointSet)

    return frame

if __name__ == "__main__":
    with open("points.csv", "r") as f:
        pts = [[float(y) for y in x.split(',')] for x in f.read().split('\n')[:-1]]
    frame = pipeline(pts)
    while True:
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    # cv2.imwrite('image.png', frame)
