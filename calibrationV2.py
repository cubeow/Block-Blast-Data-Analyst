import pyautogui
from multiprocess import Process, Manager, set_start_method, Value
from ctypes import c_bool
import cv2
import os
import time
import numpy as np
from mss import mss
from PIL import Image
import os
from matplotlib import pyplot as plt
import cv2
import math
import ctypes

def prepare(blockNumber):
    img = cv2.imread("/Users/sagewong/git/Block-Blast-Data-Analyst/temp/screen.png", 1)
    imgBoard = img[360:1190, 173:1000]
    imgBoard = cv2.cvtColor(imgBoard, cv2.COLOR_BGR2GRAY)


    rows, cols = imgBoard.shape
    # print(f"rows: {rows}, columns: {cols}")
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

    # print(corners)

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
        # print(f"smallestX: {smallestX} smallestY: {smallestY} largestX: {largestX} largestY: {largestY}")
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
    # print(corners)
    output = []
    output = duplicateRow(corners.copy(), "y", -1, min(distances))
    output = duplicateRow(output.copy(), "y", 1, min(distances))
    output = duplicateRow(output.copy(), "x", 1, min(distances))
    output = duplicateRow(output.copy(), "x", -1, min(distances))
    # print(output)
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
    #print(finalCornerCoords)

    os.system("screencapture temp/screen2.png")
    if blockNumber == 1:
        screenCoord = [1258, 1500, 215, 445]
    if blockNumber == 2:
        screenCoord = [1258, 1500, 468, 710]
    if blockNumber == 3:
        screenCoord = [1258,1500,733,961]
    
    newImg = cv2.imread("temp/screen2.png")
    newImg = newImg[screenCoord[0]:screenCoord[1], screenCoord[2]:screenCoord[3]]

    # for row in range(newImg.shape[0]):
    #     for col in range(newImg.shape[1]):
    #         if row <= screenCoord[0] or row >= screenCoord[1]:
    #             newImg[row][col] = 0
    #         if col <= 215 or col >= 445:
    #             newImg[row][col] = 0
    cv2.imwrite("hi.png", newImg)
    #display("hi.png")
    rows, cols, _ = newImg.shape
    cv2.imwrite(f"temp/imgBoard1000.png", imgBoard)
    for i in range(rows):
        for j in range(cols):
            if np.linalg.norm(newImg[i, j]-[128,73,56]) < 10 or np.linalg.norm(newImg[i, j]-[119,60,46]) < 10:
                newImg[i, j] = [0, 0, 0]
    cv2.imwrite(f"temp/imgBoard1.png", newImg)

    kernel = np.ones((5,5), np.uint8)
    newImg = cv2.erode(newImg, kernel, iterations=2)
    cv2.imwrite(f"temp/imgBoard2.png", newImg)
    # display("temp/imgBoard2.png")
    imgBoard2 = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

    ret,thresh1 = cv2.threshold(imgBoard2,80,255,cv2.THRESH_BINARY)
    cv2.imwrite("temp/imgBoard3.png", thresh1)
    # display("temp/imgBoard3.png")
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgBoard, contours, -1, (0,255,0), 3)

    cv2.imwrite("temp/imgBoard4.png", imgBoard)
    # display("temp/imgBoard4.png")

    simplifiedContours = [i[0][0] for i in contours]
    minXDistance = 1000
    for i in range(len(simplifiedContours)):
        for j in range(i + 1, len(simplifiedContours)):
            distanceFound = abs(simplifiedContours[i][0] - simplifiedContours[j][0])
            if distanceFound < minXDistance and distanceFound>5:
                minXDistance = distanceFound
    minYDistance = 1000
    # print(minXDistance)
    for i in range(len(simplifiedContours)):
        for j in range(i + 1, len(simplifiedContours)):
            distanceFound = abs(simplifiedContours[i][1] - simplifiedContours[j][1])
            if distanceFound < minYDistance and distanceFound>5:
                minYDistance = distanceFound
    # print(minYDistance)
    distance = min(minXDistance, minYDistance)
    minX = min([i[0][0][0] for i in contours])
    # print("Minimum x found: " + str(minX))
    maxX = max([i[0][0][0] for i in contours])
    # print("Maximum x found: " + str(maxX))
    minY = min([i[0][0][1] for i in contours])
    # print("Minimum y found: " + str(minY))
    maxY = max([i[0][0][1] for i in contours])
    # print("Maximum y found: " + str(maxY))
    hi = np.zeros((round((maxY-minY)/distance)+1,round((maxX-minX)/distance)+1))
    for x in range(round((maxX-minX)/distance)+1):
        for y in range(round((maxY-minY)/distance)+1):
            xVal = minX + x*distance
            yVal = minY + y*distance
            for i in simplifiedContours:
                if math.sqrt((xVal-i[0])**2 + (yVal-i[1])**2) < 5:
                    hi[y][x] = 1
                    break
    return minX, minY, screenCoord, hi

def screenshot(filePath):
    os.system("screencapture temp/secondImage.png")
    secondImg = cv2.imread("temp/secondImage.png")

    secondImg = secondImg[380:1490, 100:1000]

    cv2.imwrite(f"{filePath}.png", secondImg)

def compareTwoImages(im1, im2):
    for row in range(int(im1.shape[0]*0.5), im1.shape[0]):
        for col in range(im2.shape[1]):
            if np.linalg.norm(im1[row][col]- im2[row][col])>0:
                return row, col

def dragBlockTo(blockNumber, Xx, Yy):
    screenshot("temp/image0")

    minX, minY, screenCoord, blockShape = prepare(blockNumber)
    mouseDist = -30

    count = 0
    for i in blockShape[0]:
        if i == 0:
            count += 1
        if i == 1:
            break
    Xx += count

    newMinX = minX+screenCoord[2]
    newMinY = minY+screenCoord[0]

    # print(f"NOT HOLD min x: {newMinX}")
    # print(f"NOT HOLD min y: {newMinY}")

    pyautogui.moveTo(newMinX/2,newMinY/2)

    pyautogui.click()

    time.sleep(0.1)

    pyautogui.dragTo(newMinX/2-mouseDist, newMinY/2-mouseDist, 1, button="left")
    screenshot("temp/image1")

    pyautogui.moveTo(newMinX/2,newMinY/2)
    pyautogui.mouseDown()
    screenshot("temp/image05")
    pyautogui.drag(0, 100, 0.2, button="left", mouseDownUp=False)
    pyautogui.mouseUp()
    initialHoldY, initialHoldX = compareTwoImages(cv2.imread("temp/image0.png"), cv2.imread("temp/image05.png"))
    initialHoldX += 100
    initialHoldY += 380
    # print(f"START COORDINATE: {initialHoldX}, {initialHoldY}")

    pyautogui.mouseUp()

    time.sleep(0.1)

    pyautogui.moveTo(newMinX/2,newMinY/2)

    row, col = compareTwoImages(cv2.imread("temp/image0.png"), cv2.imread("temp/image1.png"))

    row += 380
    col += 100

    # print(f"HOLD END x: {col}")
    # print(f"HOLD END y: {row}")

    desiredPosition = [200+Xx*97, 386+Yy*97]

    # print(f"HOLD DISTANCE x: {abs(initialHoldX-col)}")
    # print(f"HOLD DISTANCE y: {abs(initialHoldY-row)}")


    # print(f"GOAL COORDINATE: {desiredPosition}")

    mouseXPerPixel = mouseDist/abs(initialHoldX-col)
    mouseYPerPixel = mouseDist/abs(initialHoldY-row)

    # print(f"DISTANCE TO TRAVEL x: {desiredPosition[0] - initialHoldX}")
    # print(f"DISTANCE TO TRAVEL y: {desiredPosition[1] - initialHoldY}")

    XdistanceToTravel = (desiredPosition[0] - initialHoldX)*mouseXPerPixel
    YdistanceToTravel = (desiredPosition[1] - initialHoldY)*mouseYPerPixel

    pyautogui.moveTo(newMinX/2,newMinY/2)
    time.sleep(0.1)
    pyautogui.drag(-XdistanceToTravel, -YdistanceToTravel, 1, button="left")
    pyautogui.mouseUp()

dragBlockTo(2, 3, 3)