import cv2
from utils import extract_grid, extract_board
from solver import solve_sudoku

img = cv2.imread("images/sample_sudoku.png")
warped = extract_grid(img)
board = extract_board(warped)

print("Original Board:")
for row in board:
    print(row)

if solve_sudoku(board):
    print("\nSolved Board:")
    for row in board:
        print(row)
else:
    print("No solution found.")
    for row in board:
        print(row)
