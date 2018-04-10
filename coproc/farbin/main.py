#!/usr/bin/env python2
# license reeejkmoved for brevity
import rospy

import time
from sensor_msgs.msg import LaserScan
import cv2
import sys
import numpy as np
import math
from threading import Thread
import scipy
import scipy.interpolate
import os
import test2


def smooth(x, window_len=11, window='hanning'):
    """
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def getNormalized(arr):
    sum = 0
    for i in arr:
        sum += abs(i*i)
    sum/len(arr)
    sum = math.sqrt(sum)
    arr = arr/sum()

class renderer:
    def __init__(self):
        self.scanHistory = []
        self.frame = np.zeros((512, 512, 3), np.uint8)
        self.grads = np.asarray([])

    def recieve(self, scanNew):
        self.scanHistory.append(list(scanNew))
        if len(self.scanHistory) > 10:
            self.scanHistory.pop(0)

    def render(self):
        # print("Rendering")
        if (len(self.scanHistory) < 10):
            return
        self.frame = np.ones((512, 512, 3), np.uint8)*255
        max = 0
        for j in self.scanHistory[-1]:
            if j[1] > max:
                max = j[1]

        points = self.scanHistory[-1]
        frame = test2.pipeline(points)
        self.frame = frame

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            self.render()
            cv2.imwrite("frame2.png", self.frame)
            os.rename("frame2.png", "frame.png")
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            # cv2.imwrite("frame.jpg", self.frame)
            # time.sleep(.1)
            # print(len(self.grads))
        sys.exit(0)

instance = renderer().start()

def callback(scan):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", scan.scan_time)

    angleMin = scan.angle_min
    angleMax = scan.angle_max
    angleIncrement = scan.angle_increment

    rangeMin = scan.range_min
    rangeMax = scan.range_max

    ranges = scan.ranges

    # protocol: [angle, distance]
    scans = []

    i, index = angleMin, 0
    while i < angleMax:
        if rangeMin <= ranges[index] <= rangeMax:
            scans.append([i, ranges[index]])

        i += angleIncrement
        index += 1

    def derivative(i):
        point1 = scans[i]
        point2 = scans[i+1]
        point0 = scans[i-1]
        deriv1 = (point2[1]-point1[1])/(point2[0]-point1[0])
        deriv2 = (point1[1]-point0[1])/(point1[0]-point0[0])
        return (deriv1+deriv2)/2

    # derivs = [derivative(i) for i in xrange(1, len(scans)-1)]
    angles = np.asarray([i[0] for i in scans])
    distances = np.asarray([i[1] for i in scans])

    # grads = np.gradient(distances, angles)

    # interp = scipy.interpolate.UnivariateSpline(angles, distances, s=0, k=4)

    if (len(scans) > 200):
        instance.recieve(scans)

def listener():
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('lidar_listener', anonymous=True)

    rospy.Subscriber("scan", LaserScan, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    listener()
