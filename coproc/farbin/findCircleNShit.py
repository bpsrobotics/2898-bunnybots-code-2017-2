import numpy as np
import scipy as sp


def getPoints(pts):
    points = pts
    # Type of points is [[angle, distance]]

def getNormalPoints(pts):
    margin = 1
    normals = []
    # [[normal, derivative, origin, pointL, pointR], [normal, derivative, origin, pointL, pointR]]
    for i in range(1, len(pts)-1):
        # Second order derivative (Xt+1 - Xt-1)/(2dt)
        pointR = pts[i+1]
        pointL = pts[i-1]
        dt = pointR[0] - pointL[0]
        dr = pointR[1] - pointL[1]
        drdt = dr/dt
        pivot = [(pointR[0]+pointL[0])/2, (pointR[1]+pointL[1])/2]  # Essentially the origin point of the deriv calc
        normal = -1/drdt
        normals.append([normal, drdt, pivot, i-1, i+1])
        # r(t) = pivot + t*normal
    return normals

def getArcProposals(pts, seedIndexes):
    divider = 4
    proposals = []  # [{'midpoint': midpoint, 'width': width}]
    # for arcWidth in range(5, len(pts) - len(pts) % 2, 2*divider):
    #     for arcMidpoint in range(arcWidth / 2, len(pts) - arcWidth / 2):
    #         proposals.append({'midpoint': arcMidpoint, 'width': arcWidth})

    proposals = {}

    for midpointIndex in seedIndexes:
        proposals[midpointIndex] = []
        totalNumPoints = len(pts)
        # Largest possible size (from left): midpointIndex
        # Largest possible size (from right): totalNumPoints - midpointIndex
        for arcWidth in range(5, min(midpointIndex - midpointIndex % 2, (totalNumPoints - midpointIndex) - (totalNumPoints - midpointIndex) % 2), 2):
            proposals[midpointIndex].append(arcWidth)
    return proposals


def getArcPointSets(proposals, pts):
    arcs = {}  # {midpoint: allPoints

    keys = list(proposals.keys())
    for i in range(len(keys)):
        try:
            pointSet = []
            # [[angle, distance]]
            midpoint = keys[i]
            maxWidth = proposals[keys[i]][-1]
            # Walk the width positive direction
            for walk in range(-maxWidth/ 2, maxWidth / 2 + 1):
                pointSet.append(pts[midpoint + walk])

            # Sort pointSet by angle just to make sure
            pointSet = sorted(pointSet, key=lambda point: point[0])
            arcs[keys[i]] = pointSet
        except:
            continue

    return arcs

def getSubarcFromArc(width, arc):
    assert width % 2 == 1
    assert len(arc) % 2 == 1
    arcWidth = len(arc)
    assert arcWidth >= width
    difference = arcWidth - width
    assert difference % 2 == 0
    points = arc[difference/2:arcWidth-(difference/2)]
    return points

def getDerivatives(pts):
    # pts is [[angle, distance]]
    derivs = []
    # angle, derivative
    for i in range(len(pts)-1):
        angle = (pts[i][0] + pts[i+1][0]) / 2
        deriv = (pts[i+1][1]-pts[i][1])/(pts[i+1][0]-pts[i][0])
        derivs.append([angle, deriv])
    return derivs

def setsOverlap(list1, list2):
    return any(i in list1 for i in list2)


from scipy import optimize
