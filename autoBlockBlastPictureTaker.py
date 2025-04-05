import os
from matplotlib import pyplot as plt
import cv2
import math


os.system("screencapture temp/screen.png")
img = cv2.imread("/Users/sagewong/git/Block-Blast-Data-Analyst/temp/screen.png", 1)
imgBoard = img[360:1190, 173:1000]
imgBoard = cv2.cvtColor(imgBoard, cv2.COLOR_BGR2GRAY)

import numpy as np
rows, cols = imgBoard.shape
print(f"rows: {rows}, columns: {cols}")
x = np.zeros(rows*cols)
for i in range(rows):
        for j in range(cols):
            x[i + j*rows] = imgBoard[i, j]
            if imgBoard[i, j] < 35:
                imgBoard[i, j] = 0
            else:
                imgBoard[i, j] = 255
corners = []
distances = []
for i in range(25, rows-50):
    for j in range(25, cols-50):
        passesTest = True
        for z in range(50):
            if imgBoard[i-25 + z, j] == 255:
                passesTest = False
                break
        for z in range(50):
            if imgBoard[i, j-25+z] == 255:
                passesTest = False
                break
        if passesTest:
            if len(corners) > 0:
                passesSecondTest = True
                for corner in corners:
                    if math.sqrt((corner[0] - i)**2 + (corner[1] - j)**2) < 25:
                        passesSecondTest = False
                        break
                if passesSecondTest:
                    distances.append(math.sqrt((corners[0][0] - i)**2 + (corners[0][1] - j)**2))
                    corners.append([i, j])
            else:
                corners.append([i, j])

# for i in corners:
#     imgBoard[i[0], i[1]] = 255

print(corners)

def duplicateRow(input, axis, direction, distance):
    smallestX = 10000
    for i in input:
        if i[0] < smallestX:
            smallestX = i[0]
    smallestY = 10000
    for i in input:
        if i[1] < smallestY:
            smallestY = i[1]
    largestX = 0
    for i in input:
        if i[0] > largestX:
            largestX = i[0]
    largestY = 0
    for i in input:
        if i[1] > largestY:
            largestY = i[1]
    print(f"smallestX: {smallestX} smallestY: {smallestY} largestX: {largestX} largestY: {largestY}")
    littleDudes = []
    if axis == "x" and direction == 1:
        for i in input:
            if abs(i[0] - largestX) < 5:
                littleDudes.append([i[0] + distance, i[1]])
    if axis == "x" and direction == -1:
        for i in input:
            if abs(i[0] - smallestX) < 5:
                littleDudes.append([i[0] - distance, i[1]])
    if axis == "y" and direction == 1:
        for i in input:
            if abs(i[1] - largestY) < 5:
                littleDudes.append([i[0], i[1] + distance])
    if axis == "y" and direction == -1:
        for i in input:
            if abs(i[1] - smallestY) < 5:
                littleDudes.append([i[0], i[1] - distance])
    for i in littleDudes:
        input.append(i)
    return input
print(corners)
output = []
output = duplicateRow(corners.copy(), "y", -1, min(distances))
output = duplicateRow(output.copy(), "y", 1, min(distances))
output = duplicateRow(output.copy(), "x", 1, min(distances))
output = duplicateRow(output.copy(), "x", -1, min(distances))

print(output)
# for i in output:
    # imgBoard[int(i[1]), int(i[0])] = 255

a = np.array(output)
sorted_indices = np.lexsort((a[:,1], a[:,0]))
sorted_points = a[sorted_indices]
# sorted_points = np.reshape(sorted_points, (2, -1))
sorted_points = sorted_points.reshape(-1, 9, 2)
finalCornerCoords = []
for group in range(sorted_points.shape[0]):
    for row in range(sorted_points[group].shape[0]):
        try:
            finalCornerCoords.append([sorted_points[group][row], sorted_points[group][row+1], sorted_points[group+1][row], sorted_points[group+1][row+1]])
        except:
            pass
print(len(finalCornerCoords))

count = 0
for coords in finalCornerCoords:
    a = imgBoard[int(min([i[0] for i in coords])):int(max([i[0] for i in coords])), int(min([i[1] for i in coords])):int(max([i[1] for i in coords]))]
    cv2.imwrite(f"imgBoard{count}.png", a)
    
    count += 1

