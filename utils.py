import numpy as np
import cv2
from digit_classifier import classify_digit

def extract_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, 0.02 * peri, True)
    pts = np.float32([pt[0] for pt in approx])
    pts = reorder_points(pts)
    side = 450
    dst = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (side, side))

def reorder_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]])

def extract_board(warped):
    board = []
    side = warped.shape[0] // 9
    for y in range(9):
        row = []
        for x in range(9):
            cell = warped[y*side:(y+1)*side, x*side:(x+1)*side]
            if cv2.countNonZero(cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)) > 50:
                digit = classify_digit(cell)
            else:
                digit = 0
            row.append(digit)
        board.append(row)
    return board
