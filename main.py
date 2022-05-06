import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pygame

pygame.mixer.init()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#importing all images
imgBackground = cv2.imread("rsc/bg.png")
imgGameOver = cv2.imread("rsc/gameOver.png")
imgBat1 = cv2.imread("rsc/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("rsc/bat2.png", cv2.IMREAD_UNCHANGED)
imgBall = cv2.imread("rsc/ball2.png", cv2.IMREAD_UNCHANGED)

#hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# variables
ballPos = [100, 100]
speedX = 10
speedY = 10
gameOver = False
score = [0, 0]


while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw, without draw is [draw=False] and remove 'img'

    #overlaying the background
    img = cv2.addWeighted(img, 0.1, imgBackground, 0.9, 0)

    #check for hands
    if hands:
        for hand in hands:
            x, y, w, h = hand["bbox"]
            h1, w1, _ = imgBat1.shape
            y1 = y - h1//2
            y1 = np.clip(y1, 20, 440)

            if hand["type"] == "Left":
                img = cvzone.overlayPNG(img, imgBat1, (59, y1))
                if 75 < ballPos[0] < 75 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    #sound
                    pygame.mixer.music.load("sounds/handle.mp3")
                    pygame.mixer.music.play(loops=0)
                    # ballPos[0] += 30
                    score[0] += 1

            if hand["type"] == "Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195, y1))
                if 1135 < ballPos[0] < 1135 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    #sound
                    pygame.mixer.music.load("sounds/handle.mp3")
                    pygame.mixer.music.play(loops=0)
                    # ballPos[0] -= 30
                    score[1] += 1

    #game over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[1] + score[0]).zfill(2), (580, 360), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 5)


    #if game not over the ball
    else:
        #move the ball up and down
        if ballPos[1] >= 530 or ballPos[1] <= 20:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        #draw the ball
        img = cvzone.overlayPNG(img, imgBall, ballPos)

        #score number
        cv2.putText(img, str(score[0]), (300, 670), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        cv2.putText(img, str(score[1]), (900, 670), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)


    #show webcam
    img[580:700, 52:260] = cv2.resize(imgRaw, (208, 120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballPos = [100, 100]
        speedX = 20
        speedY = 20
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.imread("rsc/gameOver.png")
    
