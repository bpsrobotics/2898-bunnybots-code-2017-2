import numpy as np
import cv2
from findCircleNShit import *
import time
import math

with open("points.csv", "r") as f:
    points = [[float(y) for y in x.split(',')] for x in f.read().split('\n')[:-1]]

# print (points)

# proposals = getArcProposals(points)
# print (proposals)

# arcs = getArcPointSets(proposals, points)

# print(arcs[0])

# print(getNormalPoints(points)[0])

start = time.time()
frame = np.zeros((512, 512, 3))

normals = getNormalPoints(points)

prev = None
for i in points:
    angle = i[0]
    distance = i[1]*200
    x = distance * math.cos(angle)
    y = distance * math.sin(angle) + 256
    # print(x, y)
    # y = distance
    # x = 256-(angle * 180/3.14 * 9)

    if prev is not None:
        frame = cv2.line(frame, (prev[0], prev[1]), (int(x), int(y)), (255, 0, 0), 2)
    prev = (int(x), int(y))

tmp = []
prev = None
for i in normals:
    # [normalSlope,
    # if not abs(i[1]) <= 0.01:
    #     continue
    # if not abs(i[1]) <= 1:
    #     continue
    tmp.append(i)

    angle = i[2][0]
    distance = i[2][1]*200
    x = distance * math.cos(angle)
    y = distance * math.sin(angle) + 256

    # frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    frame = cv2.arrowedLine(frame, (0, 256), (int(x), int(y)), (0, 255, 0), 2, 8, 0, 0.05)

# List of all points with a low enough normal we should actually care about searching them
candidates = sorted(list(set([x[3] for x in tmp] + [x[4] for x in tmp])), key=lambda x: points[x][0])
# print(candidates)
# print([points[x][0] for x in candidates])

proposals = getArcProposals(points, candidates)
# print(proposals)
sets = getArcPointSets(proposals, points)
# print(sets.keys())

# {midpoint: [width1: whatever, width2: whatever]}
widthMidpointDerivUniformityThingy = {}

for i in sets.keys():
    widthMidpointDerivUniformityThingy[i] = {}
    arcWidthSizes = proposals[i]
    relaventPoints = sets[i]
    for j in arcWidthSizes:
        # print(j)
        toTakeSecondDeriv = getSubarcFromArc(j, relaventPoints[:-1])
        # print("SECOND DERIV: ", toTakeSecondDeriv)
        firstDeriv = getDerivatives(toTakeSecondDeriv)
        secondDeriv = getDerivatives(firstDeriv)
        allDerivs = np.asarray([x[1] for x in secondDeriv])
        normalicy = allDerivs - np.mean(allDerivs)
        sum = 0
        for x in normalicy:
            sum += 1/abs(x)
        widthMidpointDerivUniformityThingy[i][j] = sum

maxI, maxJ, maxD = 0, 0, 0
for i in widthMidpointDerivUniformityThingy.keys():
    if i > maxI:
        maxI = i
    for j in widthMidpointDerivUniformityThingy[i].keys():
        if j > maxJ:
            maxJ = j
        if widthMidpointDerivUniformityThingy[i][j] > maxD:
            maxD = widthMidpointDerivUniformityThingy[i][j]

frame2 = np.zeros((maxI+1, maxJ+1, 1), dtype=np.uint8)
for i in widthMidpointDerivUniformityThingy.keys():
    for j in widthMidpointDerivUniformityThingy[i].keys():
        frame2[i][j] = int((widthMidpointDerivUniformityThingy[i][j] / float(maxD))*255)
        #print(i, j, int((widthMidpointDerivUniformityThingy[i][j] / float(maxD))*255))

# print(frame2)
frame2 = cv2.applyColorMap(frame2, cv2.COLORMAP_JET)
print([x[4] for x in tmp])
print("MaxI: ", maxI)
print("MaxJ: ", maxJ)
print("MaxD: ", maxD)

while True:
    cv2.imshow("e", frame2)
    cv2.imshow("f", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

# csv = ""
# for i in widthMidpointDerivUniformityThingy.keys():
#     for j in widthMidpointDerivUniformityThingy[i].keys():
#         csv += str(i) + ", " + str(j) + ", " + str(widthMidpointDerivUniformityThingy[i][j]) + "\n"

print(time.time()-start)

# with open("3dgraph.csv", "w") as f:
#     f.write(csv)
