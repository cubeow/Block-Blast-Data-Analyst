import os
from matplotlib import pyplot as plt
import cv2
import math
import copy
import numpy as np
import time
import random

def calculate(blocksDone):

    # os.system("screencapture temp/screen.png")
    img = cv2.imread("/Users/sagewong/git/Block-Blast-Data-Analyst/temp/screen.png", 1)
    imgBoard = img[360:1190, 173:1000]
    imgBoard = cv2.cvtColor(imgBoard, cv2.COLOR_BGR2GRAY)

    import numpy as np
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
    # print(len(finalCornerCoords))

    # count = 0
    # for coords in finalCornerCoords:
    #     a = imgBoard[int(min([i[0] for i in coords])):int(max([i[0] for i in coords])), int(min([i[1] for i in coords])):int(max([i[1] for i in coords]))]
    #     cv2.imwrite(f"imgBoard{count}.png", a)
        
    #     count += 1

    os.system("screencapture temp/screen2.png")
    # print("Screenshotting!!!")
    img = cv2.imread("/Users/sagewong/git/Block-Blast-Data-Analyst/temp/screen2.png", 1)
    cv2.imwrite(f"records/{blocksDone}.png", img)
    imgBoard = img[360:1190, 173:1000]
    imgBoard = cv2.cvtColor(imgBoard, cv2.COLOR_BGR2GRAY)
    count = 0
    bigList = np.zeros(64)
    for coords in finalCornerCoords:
        a = imgBoard[int(min([i[0] for i in coords])):int(max([i[0] for i in coords])), int(min([i[1] for i in coords])):int(max([i[1] for i in coords]))]
        if np.average(a.flatten()) > 50:
            bigList[count] = 1
        else:
            bigList[count] = 0
        count += 1
    board = np.reshape(bigList, (8, -1))
    # print(board)
    blocks = []
    #, [1258, 1500, 468, 710], [1258,1500,733,961]
    screenCoords = [[1258, 1500, 215, 445], [1258, 1500, 468, 710],  [1258,1500,733,961]]
    coordsUsed = []
    minXList = []
    minYList = []
    for index, coords in enumerate(screenCoords):
        imgBoard = img[coords[0]:coords[1], coords[2]:coords[3]]
        rows, cols, _ = imgBoard.shape
        # cv2.imwrite(f"temp/imgBoard1000.png", imgBoard)
        for i in range(rows):
            for j in range(cols):
                if np.linalg.norm(imgBoard[i, j]-[128,73,56]) < 10 or np.linalg.norm(imgBoard[i, j]-[119,60,46]) < 10:
                    imgBoard[i, j] = [0, 0, 0]
        cv2.imwrite(f"temp/CALCULATE1.png", imgBoard)

        kernel = np.ones((10,10), np.uint8)
        imgBoard = cv2.erode(imgBoard, kernel, iterations=2)
        cv2.imwrite(f"temp/CALCULATE2.png", imgBoard)
        # display("temp/imgBoard2.png")
        imgBoard2 = cv2.cvtColor(imgBoard, cv2.COLOR_BGR2GRAY)

        ret,thresh1 = cv2.threshold(imgBoard2,85,255,cv2.THRESH_BINARY)
        cv2.imwrite("temp/CALCULATE3.png", thresh1)
        # display("temp/imgBoard3.png")
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            cv2.drawContours(imgBoard, contours, -1, (0,255,0), 3)

            cv2.imwrite("temp/CALCULATE4.png", imgBoard)
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

            # print(simplifiedContours)

            hi = np.zeros((round((maxY-minY)/distance)+1,round((maxX-minX)/distance)+1))
            for x in range(round((maxX-minX)/distance)+1):
                for y in range(round((maxY-minY)/distance)+1):
                    xVal = minX + x*distance
                    yVal = minY + y*distance
                    for i in simplifiedContours:
                        if math.sqrt((xVal-i[0])**2 + (yVal-i[1])**2) < 5:
                            hi[y][x] = 1
                            break
            # print(hi)
            blocks.append(hi)
            coordsUsed.append([hi, index, minX, minY, coords])

    def permutations(blocks, fi):
        result = []
        if fi == len(blocks) - 1:
            return [blocks[:]]
        for i in range(fi, len(blocks)):
            blocks = swap(blocks, i, fi)
            result.extend(permutations(blocks, fi + 1))
            blocks = swap(blocks, i, fi)
        return result

    def swap(blocks, i, fi):
        temp = blocks[i]
        blocks[i] = blocks[fi]
        blocks[fi] = temp
        return blocks

    def waysToFit(figure, board):
        firstFit = []
        for rowNumber in range(board.shape[0]):
            # print(board[rowNumber])
            for cellNumber in range(board.shape[1]):
                # print(board[rowNumber][cellNumber])
                cellValue = board[rowNumber][cellNumber]
                if (cellValue == 1 and figure[0][0] == 0) or (cellValue == 0 and figure[0][0] == 1) or (cellValue == 0 and figure[0][0] == 0):
                    canFit = True
                    if cellNumber + figure.shape[1] > board.shape[0]:
                        canFit = False
                        break
                    if rowNumber + figure.shape[0] > board.shape[1]:
                        canFit = False
                        break
                    for figureRow in range(figure.shape[0]):
                        for figureColumn in range(figure.shape[1]):
                            if figure[figureRow][figureColumn] == 1:
                                if board[rowNumber + figureRow][cellNumber + figureColumn] == 1:
                                    canFit = False
                                    break
                    if canFit:
                        firstFit.append([figure, rowNumber, cellNumber])
        return firstFit
    def updateBoard(board, figure, x, y):
        board = board.copy()
        clearedRow = 0
        for row in range(figure.shape[0]):
            for column in range(figure.shape[1]):
                board[x+row][y+column] += figure[row][column]
        rowsToClear = []
        for row in range(board.shape[0]):
            for column in range(board.shape[1]):
                canClear = True
                if board[row][column] == 0:
                    canClear = False
                    break
            if canClear:
                rowsToClear.append(row)    
        colsToClear = []
        for column in range(board.shape[1]):
            for row in range(board.shape[0]):
                canClear = True
                if board[row][column] == 0:
                    canClear = False
                    break
            if canClear:
                colsToClear.append(column)
        for row in range(board.shape[0]):
            for column in range(board.shape[1]):
                if row in rowsToClear or column in colsToClear:
                    board[row][column] = 0
        if len(colsToClear) > 0 or len(rowsToClear) > 0:
            clearedRow = len(colsToClear) + len(rowsToClear)
        return board, clearedRow

    originalBoard = board.copy()

    allPermutations = permutations(blocks, 0)
    def tryOutAllMethods(board, permutation, fi):
        
        workingPaths = []
        pathTaken = []
        blockIndex = 0
        if len(permutation) == 3:
            for _, rowToFit, colToFit in waysToFit(permutation[blockIndex], board):
                board1, clearedRow = updateBoard(board, permutation[blockIndex], rowToFit, colToFit)
                pathTaken.append([permutation[blockIndex], rowToFit, colToFit, clearedRow])
                for _, rowToFit, colToFit in waysToFit(permutation[blockIndex + 1], board1):
                    board2, clearedRow = updateBoard(board1, permutation[blockIndex + 1], rowToFit, colToFit)
                    pathTaken.append([permutation[blockIndex + 1], rowToFit, colToFit, clearedRow])
                    for _, rowToFit, colToFit in waysToFit(permutation[blockIndex + 2], board2):
                        board3, clearedRow = updateBoard(board2, permutation[blockIndex + 2], rowToFit, colToFit)
                        pathTaken.append([permutation[blockIndex + 2], rowToFit, colToFit, clearedRow])
                        pathTaken.append(board3)
                        workingPaths.append(copy.deepcopy(pathTaken))
                        pathTaken = [pathTaken[0], pathTaken[1]]
                    pathTaken = [pathTaken[0]]
                pathTaken = []
            return workingPaths
        elif len(permutation) == 2:
            for _, rowToFit, colToFit in waysToFit(permutation[blockIndex], board):
                board1, clearedRow = updateBoard(board, permutation[blockIndex], rowToFit, colToFit)
                pathTaken.append([permutation[blockIndex], rowToFit, colToFit, clearedRow])
                for _, rowToFit, colToFit in waysToFit(permutation[blockIndex + 1], board1):
                    board2, clearedRow = updateBoard(board1, permutation[blockIndex + 1], rowToFit, colToFit)
                    pathTaken.append([permutation[blockIndex + 1], rowToFit, colToFit, clearedRow])
                    pathTaken.append(board2)
                    workingPaths.append(copy.deepcopy(pathTaken))
                    pathTaken = [pathTaken[0]]
                pathTaken=[]
            return workingPaths
        elif len(permutation) == 1:
            for _, rowToFit, colToFit in waysToFit(permutation[blockIndex], board):
                board1, clearedRow = updateBoard(board, permutation[blockIndex], rowToFit, colToFit)
                pathTaken.append([permutation[blockIndex], rowToFit, colToFit, clearedRow])
                pathTaken.append(board1)
                workingPaths.append(copy.deepcopy(pathTaken))
                pathTaken = []
            return workingPaths

    allSolutions = []
    for index, i in enumerate(allPermutations):
        if index == 3:
            pass
        results = tryOutAllMethods(board, copy.deepcopy(allPermutations[index]), 0)
        if results:
            allSolutions.append(results)
        # if len([i for i in results if i[2][3] == True]) > 0:
        #     return [i for i in results if i[2][3] == True][0]
        # else:
        #     pass
        #     # print(results)
    return allSolutions, blocks, coordsUsed, originalBoard
def bestOption(output, moveToReset, copyOfBoard):
    def waysToFit(figure, board, optimized=False):
            if optimized==False:
                firstFit = []
                for rowNumber in range(board.shape[0]):
                    # print(board[rowNumber])
                    for cellNumber in range(board.shape[1]):
                        # print(board[rowNumber][cellNumber])
                        cellValue = board[rowNumber][cellNumber]
                        if (cellValue == 1 and figure[0][0] == 0) or (cellValue == 0 and figure[0][0] == 1) or (cellValue == 0 and figure[0][0] == 0):
                            canFit = True
                            if cellNumber + figure.shape[1] > board.shape[0]:
                                canFit = False
                                break
                            if rowNumber + figure.shape[0] > board.shape[1]:
                                canFit = False
                                break
                            for figureRow in range(figure.shape[0]):
                                for figureColumn in range(figure.shape[1]):
                                    if figure[figureRow][figureColumn] == 1:
                                        if board[rowNumber + figureRow][cellNumber + figureColumn] == 1:
                                            canFit = False
                                            break
                            if canFit:
                                firstFit.append([figure, rowNumber, cellNumber])
                return firstFit
            else:
                howManyFits = 0
                for rowNumber in range(board.shape[0]):
                    # print(board[rowNumber])
                    for cellNumber in range(board.shape[1]):
                        # print(board[rowNumber][cellNumber])
                        cellValue = board[rowNumber][cellNumber]
                        if (cellValue == 1 and figure[0][0] == 0) or (cellValue == 0 and figure[0][0] == 1) or (cellValue == 0 and figure[0][0] == 0):
                            canFit = True
                            if cellNumber + figure.shape[1] > board.shape[0]:
                                canFit = False
                                break
                            if rowNumber + figure.shape[0] > board.shape[1]:
                                canFit = False
                                break
                            for figureRow in range(figure.shape[0]):
                                for figureColumn in range(figure.shape[1]):
                                    if figure[figureRow][figureColumn] == 1:
                                        if board[rowNumber + figureRow][cellNumber + figureColumn] == 1:
                                            canFit = False
                                            break
                            if canFit:
                                howManyFits += 1
                return howManyFits
    def countHoles(board, number):
        def traverseHoles(board, row, col):
            countHoles = 0
            if row-1 >= 0:
                if board[row-1][col] == number:
                    board[row-1][col] = abs(number-1)
                    countHoles += 1
                    countHoles += traverseHoles(board, row-1, col)
            if row+1 < board.shape[0]:
                if board[row+1][col] == number:
                    board[row+1][col] = abs(number-1)
                    countHoles += 1
                    countHoles += traverseHoles(board, row+1, col)
            if col-1 >= 0:
                if board[row][col-1] == number:
                    board[row][col-1] = abs(number-1)
                    countHoles += 1
                    countHoles += traverseHoles(board, row, col-1)
            if col+1 < board.shape[1]:
                if board[row][col+1] == number:
                    board[row][col+1] = abs(number-1)
                    countHoles += 1
                    countHoles += traverseHoles(board, row, col+1)
            return countHoles
        holes = []
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row][col] == number:
                    numberOfHoles = traverseHoles(board, row, col)
                    if numberOfHoles == 0:
                        holes.append(1)
                    else:
                        if numberOfHoles < 8:
                            holes.append(numberOfHoles)
        return holes
    def creviceCount(board):
        creviceCount = 0
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row][col] == 1:
                    if row+1 < board.shape[0]-2 and col+1 < board.shape[1]-2:
                        if board[row+1][col] != 1:
                            creviceCount += 1
                        if board[row+1][col+1] != 1:
                            creviceCount += 1
                        if board[row][col+1] != 1:
                            creviceCount += 1
        for row in range(board.shape[0]-1, 0, -1):
            for col in range(board.shape[1]-1, 0, -1):
                if board[row][col] == 1:
                    if row-1 >= 0 and col-1 >= 0:
                        if board[row-1][col] != 1:
                            creviceCount += 1
                        if board[row-1][col-1] != 1:
                            creviceCount += 1
                        if board[row][col-1] != 1:
                            creviceCount += 1
        for row in range(board.shape[0]-1, 0, -1):
            for col in range(board.shape[1]):
                if board[row][col] == 1:
                    if row-1 >= 0 and col+1 < board.shape[1]-2:
                        if board[row-1][col] != 1:
                            creviceCount += 1
                        if board[row-1][col+1] != 1:
                            creviceCount += 1
                        if board[row][col+1] != 1:
                            creviceCount += 1
        for row in range(board.shape[0]):
            for col in range(board.shape[1]-1, 0, -1):
                if board[row][col] == 1:
                    if row+1 < board.shape[0]-2 and col-1 >= 0:
                        if board[row+1][col] != 1:
                            creviceCount += 1
                        if board[row+1][col-1] != 1:
                            creviceCount += 1
                        if board[row][col-1] != 1:
                            creviceCount += 1
        return creviceCount
    def squareCheck(area):
        numBRFit = np.array([[0, 0, 1], 
                            [0, 0, 1], 
                            [1, 1, 1]])
        numBLFit = np.array([[1, 0, 0], 
                            [1, 0, 0], 
                            [1, 1, 1]])
        numTRFit = np.array([[1, 1, 1], 
                            [0, 0, 1], 
                            [0, 0, 1]])
        numTLFit = np.array([[1, 1, 1], 
                            [1, 0, 0], 
                            [1, 0, 0]])
        
        if np.array_equal(area, numBRFit) or np.array_equal(area, numBLFit) or np.array_equal(area, numTRFit) or np.array_equal(area, numTLFit):
            return 0
        
        rowIndicesFilled = np.array([])
        colIndicesFilled = np.array([])
        for row in range(area.shape[0]):
            for col in range(area.shape[1]):
                if area[row][col] == 1:
                    rowIndicesFilled = np.append(rowIndicesFilled, row)
                    colIndicesFilled = np.append(colIndicesFilled, col)
        if len(colIndicesFilled) == 0 or len(rowIndicesFilled) == 0:
            return 0
        minX = min(colIndicesFilled)
        maxX = max(colIndicesFilled)
        minY = min(rowIndicesFilled)
        maxY = max(rowIndicesFilled)
        count = 0
        for row in range(int(minY), int(maxY)+1):
            for col in range(int(minX), int(maxX)+1):
                if area[row][col] == 0:
                    count += 1
        return count

    def roughEdgesScore(board):
        count = 0
        for row in range(1, board.shape[0] - 1):
            for col in range(1, board.shape[1] - 1):
                count += squareCheck(board[row-1:row+2, col-1:col+2])
        return count
    def assignPoints(board, output):
        num2x2Fit = waysToFit(np.array([[1, 1], [1, 1]]), np.array(board), True)
        num3x3Fit = waysToFit(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), np.array(board), True)
        num5x1Fit = waysToFit(np.array([[1], [1], [1], [1], [1]]), np.array(board), True)
        num4x1Fit = waysToFit(np.array([[1], [1], [1], [1]]), np.array(board), True)
        num3x1Fit = waysToFit(np.array([[1], [1], [1]]), np.array(board), True)

        num1x5Fit = waysToFit(np.array([[1, 1, 1, 1, 1]]), np.array(board), True)
        num1x4Fit = waysToFit(np.array([[1, 1, 1, 1]]), np.array(board), True)
        num1x3Fit = waysToFit(np.array([[1, 1, 1]]), np.array(board), True)

        numBRFit = waysToFit(np.array([[0, 0, 1], 
                                      [0, 0, 1], 
                                      [1, 1, 1]]), np.array(board), True)
        numBLFit = waysToFit(np.array([[1, 0, 0], 
                                      [1, 0, 0], 
                                      [1, 1, 1]]), np.array(board), True)
        numTRFit = waysToFit(np.array([[1, 1, 1], 
                                      [0, 0, 1], 
                                      [0, 0, 1]]), np.array(board), True)
        numTLFit = waysToFit(np.array([[1, 1, 1], 
                                      [1, 0, 0], 
                                      [1, 0, 0]]), np.array(board), True)

        #                                                                                                                                                                                       Checks if a row clears                            Counts number of holes (Empty)        Num of spaces filled                Counts the number of holes (filled)   How many are two or one holes?                                                   Amount of standalone islands                                                     Rough edges count                  favors later clears
        return numBRFit*10 + numBLFit*10 + numTRFit*10 + numTLFit*10 + num2x2Fit*5 + num3x3Fit*20 + num5x1Fit*2 + num1x5Fit*2 + num4x1Fit*0.8 + num3x1Fit*0.5 + num1x4Fit*0.8 + num1x3Fit*0.5 + sum([i[3] for i in output[:-1] if i[3] > 0])*30 - len(countHoles(board.copy(), 0))*5 - abs(np.count_nonzero(board==1))*0.5- len(countHoles(board.copy(), 1))*10 - len([i for i in countHoles(board.copy(), 1) if i == 2 or i == 1 or i == 3])*20 - len([i for i in countHoles(board.copy(), 0) if i == 2 or i == 1 or i == 3])*20 - roughEdgesScore(board.copy()) * 0.5 + [index for index, i in enumerate(output[:-1]) if i[3]>0][0] if len([index for index, i in enumerate(output[:-1]) if i[3]>0]) > 0 else 0
    def assignPointsVisualizer(board, output):
        num2x2Fit = waysToFit(np.array([[1, 1], [1, 1]]), np.array(board), True)
        num3x3Fit = waysToFit(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]), np.array(board), True)
        num5x1Fit = waysToFit(np.array([[1], [1], [1], [1], [1]]), np.array(board), True)
        num4x1Fit = waysToFit(np.array([[1], [1], [1], [1]]), np.array(board), True)
        num3x1Fit = waysToFit(np.array([[1], [1], [1]]), np.array(board), True)

        num1x5Fit = waysToFit(np.array([[1, 1, 1, 1, 1]]), np.array(board), True)
        num1x4Fit = waysToFit(np.array([[1, 1, 1, 1]]), np.array(board), True)
        num1x3Fit = waysToFit(np.array([[1, 1, 1]]), np.array(board), True)

        numBRFit = waysToFit(np.array([[0, 0, 1], 
                                      [0, 0, 1], 
                                      [1, 1, 1]]), np.array(board), True)
        numBLFit = waysToFit(np.array([[1, 0, 0], 
                                      [1, 0, 0], 
                                      [1, 1, 1]]), np.array(board), True)
        numTRFit = waysToFit(np.array([[1, 1, 1], 
                                      [0, 0, 1], 
                                      [0, 0, 1]]), np.array(board), True)
        numTLFit = waysToFit(np.array([[1, 1, 1], 
                                      [1, 0, 0], 
                                      [1, 0, 0]]), np.array(board), True)

        #                                                                                                                                                                                       Checks if a row clears                            Counts number of holes (Empty)        Num of spaces filled                Counts the number of holes (filled)   How many are two or one holes?                                                   Amount of standalone islands                                                     Rough edges count                  favors later clears
        print("Positives: ")
        print(f"numBRFit*10: {numBRFit*10}")
        print(f"numBLFit*10: {numBLFit*10}")
        print(f"numTRFit*10: {numTRFit*10}")
        print(f"numTLFit*10: {numTLFit*10}")
        print(f"num2x2Fit*5: {num2x2Fit*5}")
        print(f"num3x3Fit*20: {num3x3Fit*20}")
        print(f"num5x1Fit*2: {num5x1Fit*2}")
        print(f"num1x5Fit*2: {num1x5Fit*2}")
        print(f"num4x1Fit*0.8: {num4x1Fit*0.8}")
        print(f"num3x1Fit*0.5: {num3x1Fit*0.5}")
        print(f"num1x4Fit*0.8 {num1x4Fit*0.8}")
        print(f"num1x3Fit*0.5: {num1x3Fit*0.5}")
        print(f"row clears*30: {sum([i[3] for i in output[:-1] if i[3] > 0])*30}")
        print("Negatives: ")
        print(f"numberOfIslands*5: {len(countHoles(board.copy(), 0))*5}")
        print(f"filledSquares*0.5: {abs(np.count_nonzero(board==1))*0.5}")
        print(f"numberOfIslands*10: {len(countHoles(board.copy(), 1))*10}")
        print(f"numberOfFilledIslands*20: {len([i for i in countHoles(board.copy(), 1) if i == 2 or i == 1 or i == 3])*20}")
        print(f"1or2or3Islands*20: {len([i for i in countHoles(board.copy(), 0) if i == 2 or i == 1 or i == 3])*20}")
        print(f"roughEdgesScore* 0.5: {roughEdgesScore(board.copy()) * 0.5}")
    numberOfSolutions = sum(len(row) for row in output)
    # print(f"number of possible solutions: {numberOfSolutions}")
    if numberOfSolutions > 50000:
        maxScore = 0
        bestArray = []
        # print("we doing it the easy way")
        for i in range(len(output)):
            for j in range(100):
                a = random.randint(0, len(output[i])-1)
                score = assignPoints(output[i][a][-1], output[i][a])
                if score > maxScore:
                    maxScore = score
                    bestArray = output[i][a]
        if bestArray != []:
            assignPointsVisualizer(bestArray[-1], bestArray)
            return bestArray

    firstMoveClear = []
    secondMoveClear = []
    thirdMoveClear = []
    for permutation in output:
        for solution in permutation:
            if len(solution) == 4:
                if solution[0][3] > 0:
                    firstMoveClear.append(solution)
                elif solution[1][3] > 0:
                    secondMoveClear.append(solution)
                elif solution[2][3] > 0:
                    thirdMoveClear.append(solution)
            elif len(solution) == 3:
                if solution[0][3] > 0:
                    secondMoveClear.append(solution)
                elif solution[1][3] > 0:
                    thirdMoveClear.append(solution)
            elif len(solution) == 2:
                if solution[0][3] > 0:
                    thirdMoveClear.append(solution)

    
    maxScore = -10000
    bestArray = []
    # print("MOVE TO RESET: " + str(moveToReset))
    if numberOfSolutions >= 5000 and np.count_nonzero(copyOfBoard==1) <= 40:
        # print("Playing offensively")
        if moveToReset == 3:
            for i in thirdMoveClear:
                score = assignPoints(i[-1], i)
                if score > maxScore:
                    maxScore = score
                    bestArray = i
            if bestArray != []:
                print(f"Max score: {maxScore}")
                assignPointsVisualizer(bestArray[-1], bestArray)

                return bestArray
            for i in secondMoveClear:
                score = assignPoints(i[-1], i)
                if score > maxScore:
                    maxScore = score
                    bestArray = i
            if bestArray != []:
                print(f"Max score: {maxScore}")
                assignPointsVisualizer(bestArray[-1], bestArray)

                return bestArray
            for i in firstMoveClear:
                score = assignPoints(i[-1], i)
                if score > maxScore:
                    maxScore = score
                    bestArray = i
            if bestArray != []:
                print(f"Max score: {maxScore}")
                assignPointsVisualizer(bestArray[-1], bestArray)

                return bestArray
            return output[0][0]
        elif moveToReset == 2:
            goodOnes = []
            for i in secondMoveClear:
                score = assignPoints(i[-1], i)
                if score > maxScore:
                    maxScore = score
                    bestArray = i
                if i[len(i)-2][3] > 0:
                    goodOnes.append(i)
            for i in firstMoveClear:
                score = assignPoints(i[-1], i)
                if score > maxScore:
                    maxScore = score
                    bestArray = i
                if i[len(i)-2][3] > 0 or i[len(i)-3][3] > 0:
                    goodOnes.append(i)
            maxGoodOnesScore = -100000
            goodOnesArray = []
            for i in goodOnes:
                score = assignPoints(i[-1], i)
                if score > maxGoodOnesScore:
                    maxGoodOnesScore = score
                    bestArray = i
            if goodOnesArray != []:
                return goodOnesArray
            if bestArray != []:
                print(f"Max score: {maxScore}")

                assignPointsVisualizer(bestArray[-1], bestArray)

                return bestArray
            for i in thirdMoveClear:
                score = assignPoints(i[-1], i)
                if score > maxScore:
                    maxScore = score
                    bestArray = i
            if bestArray != []:
                print(f"Max score: {maxScore}")
                assignPointsVisualizer(bestArray[-1], bestArray)

                return bestArray
            return output[0][0]
        elif moveToReset == 1:
            for i in firstMoveClear:
                score = assignPoints(i[-1], i)
                if score > maxScore:
                    maxScore = score
                    bestArray = i
                if i[len(i)-2][3] > 0 or i[len(i)-3][3] > 0:
                    return i
            for i in thirdMoveClear:
                score = assignPoints(i[-1], i)
                if score > maxScore:
                    maxScore = score
                    bestArray = i
            if bestArray != []:
                print(f"Max score: {maxScore}")

                assignPointsVisualizer(bestArray[-1], bestArray)

                return bestArray
            for i in secondMoveClear:
                score = assignPoints(i[-1], i)
                if score > maxScore:
                    maxScore = score
                    bestArray = i
            if bestArray != []:
                return bestArray
            return output[0][0]
    else:
        # print("Playing defensively")
        for permutation in output:
            for i in permutation:
                score = assignPoints(i[-1], i)
                if score > maxScore:
                    maxScore = score
                    bestArray = i
        print(f"Max score: {maxScore}")

        assignPointsVisualizer(bestArray[-1], bestArray)

        return bestArray
    # print("Playing defensively")
    # for permutation in output:
    #     for i in permutation:
    #         score = assignPoints(i[-1], i)
    #         if score > maxScore:
    #             maxScore = score
    #             bestArray = i
    # return bestArray
# allSolutions, blocks, coordsUsed, board = calculate()
# print(bestOption(allSolutions, 3, board))