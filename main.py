import importlib
import pyautogui
from multiprocess import Process, Manager, set_start_method, Value
from ctypes import c_bool
import calculatePositions
importlib.reload(calculatePositions)
import pandas as pd
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

df = pd.read_csv("savedCalibrations.csv")
df['Block'] = df['Block'].astype(str)


directory = "records"
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

def prepare(blockNumber, blocksDone):
    img = cv2.imread("/Users/sagewong/git/Block-Blast-Data-Analyst/temp/screen.png", 1)
    cv2.imwrite(f"{blocksDone}.png", img)
    imgBoard = img[360:1190, 173:1200]
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

    os.system(f"screencapture records/{blocksDone}im0EntireScreen.png")
    if blockNumber == 0:
        screenCoord = [1258, 1500, 215, 445]
    if blockNumber == 1:
        screenCoord = [1258, 1500, 468, 710]
    if blockNumber == 2:
        screenCoord = [1258,1500,733,961]
    
    newImg = cv2.imread(f"records/{blocksDone}im0EntireScreen.png")
    newImg = newImg[screenCoord[0]:screenCoord[1], screenCoord[2]:screenCoord[3]]

    # for row in range(newImg.shape[0]):
    #     for col in range(newImg.shape[1]):
    #         if row <= screenCoord[0] or row >= screenCoord[1]:
    #             newImg[row][col] = 0
    #         if col <= 215 or col >= 445:
    #             newImg[row][col] = 0
    #display("hi.png")
    rows, cols, _ = newImg.shape
    # cv2.imwrite(f"temp/imgBoard1000.png", imgBoard)
    for i in range(rows):
        for j in range(cols):
            if np.linalg.norm(newImg[i, j]-[128,73,56]) < 10 or np.linalg.norm(newImg[i, j]-[119,60,46]) < 10:
                newImg[i, j] = [0, 0, 0]
    cv2.imwrite(f"records/{blocksDone}im1MakeBlack.png", newImg)

    kernel = np.ones((5,5), np.uint8)
    newImg = cv2.erode(newImg, kernel, iterations=2)
    cv2.imwrite(f"records/{blocksDone}im2Erode.png", newImg)
    # display("temp/imgBoard2.png")
    imgBoard2 = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

    ret,thresh1 = cv2.threshold(imgBoard2,85,255,cv2.THRESH_BINARY)
    cv2.imwrite(f"records/{blocksDone}im3threshold.png", thresh1)
    # display("temp/imgBoard3.png")
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgBoard, contours, -1, (0,255,0), 3)

    cv2.imwrite(f"records/{blocksDone}im4contours.png", imgBoard)
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

    secondImg = secondImg[380:1490,100:1200]

    cv2.imwrite(f"{filePath}.png", secondImg)

def compareTwoImages(im1, im2):
    differentOnes = []
    for row in range(int(im1.shape[0]*0.3), im1.shape[0]):
        for col in range(im2.shape[1]):
            if np.linalg.norm(im1[row][col]- im2[row][col])>0:
                differentOnes.append([row, col])
                if len([i for i in differentOnes if i[0] == row]) > 75:
                    # print([i for i in differentOnes if i[0] == row])
                    return [i for i in differentOnes if i[0] == row][0]

def dragBlockTo(blockNumber, Xx, Yy, blocksDone, minX, minY, screenCoord, blockShape):
    screenshot(f"records/{blocksDone}im7initialRecord")

    mouseDist = -40

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

    result = df[
        (df['Block'] == str(blockShape)) &
        (df['ScreenCoordX'] == screenCoord[0]) &
        (df['ScreenCoordY'] == screenCoord[2])
    ]

    matchFound = False

    if result.empty:
        time.sleep(0.1)

        pyautogui.dragTo(newMinX/2-mouseDist, newMinY/2-mouseDist, 0.2, button="left")
        screenshot(f"records/{blocksDone}im9initialDrag")

        pyautogui.moveTo(newMinX/2,newMinY/2)
        pyautogui.mouseDown()
        screenshot(f"records/{blocksDone}im8initialHold")
        pyautogui.drag(0, 100, 0.1, button="left", mouseDownUp=False)
        pyautogui.mouseUp()
        initialHoldY, initialHoldX = compareTwoImages(cv2.imread(f"records/{blocksDone}im7initialRecord.png"), cv2.imread(f"records/{blocksDone}im8initialHold.png"))
        row, col = compareTwoImages(cv2.imread(f"records/{blocksDone}im7initialRecord.png"), cv2.imread(f"records/{blocksDone}im9initialDrag.png"))
        initialHoldX += 100
        initialHoldY += 380
        row += 380
        col += 100
    else:
        matchFound = True
        print("Database match found!")
        initialHoldX = result.iloc[0]["InitialHoldX"]
        initialHoldY = result.iloc[0]["InitialHoldY"]
        row = result.iloc[0]["Row"]
        col = result.iloc[0]["Col"]

    

    # print(f"SCREEN COORD X: {screenCoord[0]}, Y: {screenCoord[2]}")
    # print(f"START COORDINATE: {initialHoldX}, {initialHoldY}")

    pyautogui.mouseUp()

    time.sleep(0.1)

    pyautogui.moveTo(newMinX/2,newMinY/2)

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
    pyautogui.drag(-XdistanceToTravel, -YdistanceToTravel, 0.2, button="left")
    pyautogui.mouseUp()
    return blockShape, screenCoord, initialHoldX, initialHoldY, row, col, matchFound

def elements_equal(a, b):
    return np.array_equal(a[0], b[0]) and a[1] == b[1]

def changeColor():
    time.sleep(0.1)
    pyautogui.moveTo(472, 101)
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.moveTo(381, 432)
    pyautogui.click()
    time.sleep(1)
    pyautogui.moveTo(300, 694)
    pyautogui.click()
    time.sleep(2)


pyautogui.moveTo(100, 100)
pyautogui.click()
moveToReset = 3
lastOutput = 0
lastBlocks = 0
blocksDone = 0
while True:
    try:
        output, blocks, coordsUsed, board = calculatePositions.calculate(blocksDone)
        print(board)
    except:
        print("YOU ARE COOKED")
        break
    if str(lastBlocks) == str(blocks):
        changeColor()
        lastOutput = []
        lastBlocks = []
    else:
        if blocksDone != 0 and len(blocks) == 3:
            for blockShapem, screenCoordXm, screenCoordYm, initialHoldXm, initialHoldYm, rowm, colm, matchFoundm in zip(blockShapeList, screenCoordXList, screenCoordYList, initialHoldXList, initialHoldYList, rowList, colList, matchFoundList):
                if matchFoundm == False:
                    new_row = {'Block': blockShapem, 'ScreenCoordX': screenCoordXm, 'ScreenCoordY': screenCoordYm, 'InitialHoldX': initialHoldXm, 'InitialHoldY': initialHoldYm, 'Row': rowm, 'Col':colm}
                    df.loc[len(df)] = new_row
                    df.to_csv("savedCalibrations.csv", index=False)
    blockShapeList = []
    screenCoordXList = []
    screenCoordYList = []
    initialHoldXList = []
    initialHoldYList = []
    rowList = []
    colList = []
    matchFoundList = []
    # print(output)
    output2 = calculatePositions.bestOption(output, moveToReset, board)
    print(output2)
    for index, i in enumerate(output2[:-1]):
        # print([j[1] for j in coordsUsed if np.array_equal(j[0], i[0])][0])
        blockShape, screenCoord, initialHoldX, initialHoldY, row, col, matchFound = dragBlockTo([j[1] for j in coordsUsed if np.array_equal(j[0], i[0])][0], i[2], i[1], blocksDone, [j[2] for j in coordsUsed if np.array_equal(j[0], i[0])][0], [j[3] for j in coordsUsed if np.array_equal(j[0], i[0])][0], [j[4] for j in coordsUsed if np.array_equal(j[0], i[0])][0], [j[0] for j in coordsUsed if np.array_equal(j[0], i[0])][0])
        blockShapeList.append(str(blockShape))
        screenCoordXList.append(screenCoord[0])
        screenCoordYList.append(screenCoord[2])
        initialHoldXList.append(initialHoldX)
        initialHoldYList.append(initialHoldY)
        rowList.append(row)
        colList.append(col)
        matchFoundList.append(matchFound)
        if i[3] > 0:
            time.sleep(2)
            moveToReset = index+1
        
        coordsUsed.pop(next(k for k, item in enumerate(coordsUsed) if elements_equal(item, [j for j in coordsUsed if np.array_equal(j[0], i[0])][0])))
        lastOutput = output.copy()
        lastBlocks = blocks.copy()
        blocksDone += 1
        

# dragBlockTo(2, 7, 0)
