import tkinter as tk
import tkinter.font
import serial
import cv2
import numpy as np
import time
from PIL import ImageTk, Image
import warnings
import pdb
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore')
prevTime = 0

def video_play():
    global prevTime, judge
    ret, frame = cap.read()
    lz_x_r = 0
    lz_y_r = 0
    lz_x_g = 0
    lz_y_g = 0
    px_b = 0
    py_b = 0
    # 점 인식 함수 초기값 지정
    params = cv2.SimpleBlobDetector_Params()
    detector = cv2.SimpleBlobDetector_create(params)

    if ret:
        start_t = time.perf_counter()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        lower_r = np.array([165, 50, 80])
        upper_r = np.array([180, 255, 255])
        frame_r = cv2.inRange(frame_hsv, lower_r, upper_r)

        lower_g = np.array([30, 50, 130])
        upper_g = np.array([80, 255, 255])
        frame_g = cv2.inRange(frame_hsv, lower_g, upper_g)

        video_red = cv2.bitwise_and(frame,frame,mask=frame_r)
        video_green = cv2.bitwise_and(frame,frame,mask=frame_g)

        # 블롭 안잡히면 파라미터 문제일 수도 있음
        keypoints_r = detector.detect(video_red)
        keypoints_g = detector.detect(video_green)

        # 적색 레이저 인식
        for point in keypoints_r:
            lz_x_r = point.pt[0]
            lz_y_r = point.pt[1]
            lazer_x_red.set("lazer_x (red) : " + str(int(lz_x_r)))
            lazer_y_red.set("lazer_y (red) : " + str(int(lz_y_r)))
            cv2.circle(frame, (int(lz_x_r), int(lz_y_r)), 10, (0, 0, 255), 2)

        # 녹색 레이저 인식
        for point in keypoints_g:
            lz_x_g = point.pt[0]
            lz_y_g = point.pt[1]
            lazer_x_green.set("lazer_x (green) : " + str(int(lz_x_g)))
            lazer_y_green.set("lazer_y (green) : " + str(int(lz_y_g)))
            cv2.circle(frame, (int(lz_x_g), int(lz_y_g)), 10, (0, 255, 0), 2)

        # 골대 인식
        frame_gray_tsh = np.where(frame_gray < 120, frame_gray, 255)
        corners = cv2.goodFeaturesToTrack(frame_gray_tsh, 3, 0.5, 10)

        # 골대가 비디오 내부에 잡힐 경우 좌표 (px_b,py_b)을 출력
        if corners is not None:
            corners = np.int0(corners).reshape(
                corners.shape[0], corners.shape[2])

            xy_b_12 = corners[corners[:, 0].argsort(kind='mergesort')]
            xy_b_34 = corners[corners[:, 1].argsort(kind='mergesort')]
            x1_b, y1_b = xy_b_12[0]
            x2_b, y2_b = xy_b_12[-1]
            x3_b, y3_b = xy_b_34[0]
            x4_b, y4_b = xy_b_34[-1]

            px_b = ((x1_b * y2_b - y1_b * x2_b) * (x3_b - x4_b) - (x1_b - x2_b) * (x3_b * y4_b - y3_b * x4_b)) / \
                ((x1_b - x2_b) * (y3_b - y4_b) - (y1_b - y2_b) * (x3_b - x4_b))

            py_b = ((x1_b * y2_b - y1_b * x2_b) * (y3_b - y4_b) - (y1_b - y2_b) * (x3_b * y4_b - y3_b * x4_b)) / \
                ((x1_b - x2_b) * (y3_b - y4_b) - (y1_b - y2_b) * (x3_b - x4_b))

            # px_b = corners[:, 0].sum() / len(corners)
            # py_b = corners[:, 1].sum() / len(corners)

            if np.isnan(px_b) == False and np.isnan(py_b) == False:
                # cv2.circle(frame, (int(px_b), int(py_b)), 10, (255, 0, 0), 3)
                goal_x.set("goal_x : " + str(int(px_b)))
                goal_y.set("goal_y : " + str(int(py_b)))

            for i in corners:
                x_b, y_b = i.ravel()
                # cv2.circle(frame, (x_b, y_b), 3, 255, -1)
        else:
            # 골대가 영상에 잡히지 않을 때 좌표 (0,0) 지정
            goal_x.set("goal_x : 0")
            goal_y.set("goal_y : 0")

        # 비디오 FPS 출력
        terminate_t = time.perf_counter()
        fps = int(1. / (terminate_t - start_t))
        cv2.putText(frame, "FPS : " + str(fps), (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # img = Image.fromarray(video_red)
        # img = Image.fromarray(video_green)
        # img = Image.fromarray(video_red)
        # img = Image.fromarray(video_red)

        imgtk = ImageTk.PhotoImage(image=img)
        label2.imgtk = imgtk
        label2.configure(image=imgtk)
        label2.after(10, video_play)

    else:
        cap.release()
        return


# tkinter UI 설정부분
window = tk.Tk()
window.title('control')
window.geometry("1500x800+500+100")
window.resizable(False, False)

frm = tk.Frame(window, bg="white", width=500, height=400)
frm.place(x=0, y=0)
label2 = tk.Label(frm)
label2.grid()

goal_x = tk.StringVar(window)
goal_x.set("0")
goal_y = tk.StringVar(window)
goal_y.set("0")
lazer_x_red = tk.StringVar(window)
lazer_x_red.set("lazer_x (red) : 0")
lazer_y_red = tk.StringVar(window)
lazer_y_red.set("lazer_y (red) : 0")
lazer_x_green = tk.StringVar(window)
lazer_x_green.set("lazer_x (green) : 0")
lazer_y_green = tk.StringVar(window)
lazer_y_green.set("lazer_y (green) : 0")

font = tk.font.Font(size=20)
goal_x_label = tk.Label(window, textvariable=goal_x, font=font)
goal_x_label.place(x=750, y=100)
goal_y_label = tk.Label(window, textvariable=goal_y, font=font)
goal_y_label.place(x=750, y=130)

lazer_x_red_label = tk.Label(window, textvariable=lazer_x_red, font=font)
lazer_x_red_label.place(x=750, y=200)
lazer_y_red_label = tk.Label(window, textvariable=lazer_y_red, font=font)
lazer_y_red_label.place(x=750, y=230)

lazer_x_green_label = tk.Label(window, textvariable=lazer_x_green, font=font)
lazer_x_green_label.place(x=750, y=300)
lazer_y_green_label = tk.Label(window, textvariable=lazer_y_green, font=font)
lazer_y_green_label.place(x=750, y=330)

win = tk.StringVar(window)
win.set("")
win_label = tk.Label(window, textvariable=win, font=font)
win_label.place(x=750, y=400)

cap = cv2.VideoCapture(1)

video_play()
window.mainloop()
