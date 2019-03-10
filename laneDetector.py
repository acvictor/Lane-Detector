import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from math import *
import argparse

from collections import deque, defaultdict

QUEUE_LENGTH=50

def applySmoothing(image, kernel_size=3):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def convertGrayScale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def detectEdges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def filterRegion(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

def selectRegion(image):
    """
    Keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # Define the polygon by vertices
    a = [0, 1023]
    b  = [0, 630]
    c     = [720, 455]
    d = [1505, 511]
    e    = [2047, 689]
    f = [2047, 1023] 
    # Vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[a, b, c, d, e, f]], dtype = np.int32)
    return filterRegion(image, vertices)

def houghLines(image):
    """
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho = 1, theta = np.pi / 180, threshold = 20, minLineLength = 20, maxLineGap = 300)

'''
Use this function to see what the Hough lines look like
'''
def drawLines(image, lines, color = [255, 0, 0], thickness = 2, makeCopy = True):
    # Lines returned by cv2.HoughLinesP have shape (-1, 1, 4)
    if makeCopy:
        image = np.copy(image) # Don't want to modify the original
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def average_slope_intercept(lines):
    avgLines = defaultdict(list)
    weights = defaultdict(list)
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue # Ignore a vertical line
            slope = (y2 - y1) / float(x2 - x1)
            slope = floor(slope * 10) / 10

            if slope == 0:
                continue # Avoid division by zero

            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            avgLines[slope].append((slope, intercept))
            weights[slope].append((length))
    
    keys = []
    for key in sorted(avgLines):
        keys.append(key)

    newAvgLines = defaultdict(list)
    newWeights = defaultdict(list)
    for i in range(1, len(keys)):
        if abs(keys[i] - keys[i - 1]) <= .1:
            slope = (keys[i] + keys[i - 1]) / 2.0
            for (s, intercept) in avgLines[keys[i]]:
                newAvgLines[slope].append((s, intercept))
            for (s, intercept) in avgLines[keys[i - 1]]:
                newAvgLines[slope].append((s, intercept))
            for (l) in weights[keys[i]]:
                newWeights[slope].append((l))
            for (l) in weights[keys[i - 1]]:
                newWeights[slope].append((l))  
        else:
            if(i == 1):
                slope = keys[i - 1]
                for (s, intercept) in avgLines[keys[i - 1]]:
                    newAvgLines[slope].append((s, intercept))
                for (l) in weights[keys[i - 1]]:
                    newWeights[slope].append((l))
            slope = keys[i]
            for (s, intercept) in avgLines[keys[i]]:
                newAvgLines[slope].append((s, intercept))
            for (l) in weights[keys[i]]:
                newWeights[slope].append((l))
        
    count = 0
    for key in newAvgLines:
        if count == 2:
            break
        if count == 0:
            leftLane  = np.dot(newWeights[key],  newAvgLines[key]) / np.sum(newWeights[key])  if len(newWeights[key]) > 0 else None
        if count == 1:
            rightLane = np.dot(newWeights[key], newAvgLines[key])/ np.sum(newWeights[key]) if len(newWeights[key]) > 0 else None
        count = count + 1
    return leftLane, rightLane 

def makeLinePoints(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    
    slope, intercept = line
    
    # cv2.line requires integers
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))

def laneLines(image, lines):
    leftLane, rightLane = average_slope_intercept(lines)
    
    y1 = image.shape[0] # Bottom of the image
    y2 = y1 * 0.6       # Slightly lower than the middle

    leftLine  = makeLinePoints(y1, y2, leftLane)
    rightLine = makeLinePoints(y1, y2, rightLane)
    
    return leftLine, rightLine

    
def drawLaneLines(image, lines, color=[255, 0, 0], thickness=20):
    # Make a separate image to draw lines and combine with the orignal later
    lineImage = np.zeros_like(image)
    for line in lines:
        if line is not None:
            print(line)
            (x1, y1) , (x2, y2) = line
            cv2.line(lineImage, (x1, y1), (x2, y2), color, thickness)
    # image1 * alpha + image2 * beta + lambda
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, lineImage, 0.95, 0.0)
             
class LaneDetector:
    def __init__(self):
        self.leftLines  = deque(maxlen=QUEUE_LENGTH)
        self.rightLines = deque(maxlen=QUEUE_LENGTH)

    def process(self, img):
        retval, threshold = cv2.threshold(convertGrayScale(img), 115, 255, cv2.THRESH_BINARY)
        smooth            = applySmoothing(threshold)
        edges             = detectEdges(smooth)
        roi               = selectRegion(edges)
        lines             = houghLines(roi)
        leftLine, rightLine = laneLines(img, lines)
        im = drawLines(img, lines)

        def meanLine(line, lines):
            if line is not None:
                lines.append(line)

            if len(lines) > 0:
                line = np.mean(lines, axis=0, dtype=np.int32)
                line = tuple(map(tuple, line)) # cv2.line needs tuples to work
            return line

        leftLine  = meanLine(leftLine,  self.leftLines)
        rightLine = meanLine(rightLine, self.rightLines)

        return drawLaneLines(img, (leftLine, rightLine))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to image")
    parser.parse_args()
    args = parser.parse_args()
    if args.path:
        detector = LaneDetector()
        img = cv2.imread(args.path)
        res = detector.process(img)
        plt.imshow(res, cmap = 'gray')
        plt.title('Lanes'), plt.xticks([]), plt.yticks([])
        plt.show()
    else:
        print("Usage: python laneDetector.py --path filepath")


