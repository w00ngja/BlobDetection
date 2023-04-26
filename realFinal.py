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

judge = False
# ser = serial.Serial(port='COM3', baudrate=9600)
x_loc = 90
y_loc = 90
x_ser = ''
y_ser = ''


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
    params.blobColor = 0
    params.minThreshold = 0
    params.maxThreshold = 255
    params.minArea = 2
    params.minConvexity = 1
    params.maxArea = 255
    params.maxConvexity = 255
    params.minInertiaRatio = 1
    params.maxInertiaRatio = 255
    params.minCircularity = 1
    params.maxCircularity = 255
    detector = cv2.SimpleBlobDetector_create(params)

    if ret:
        start_t = time.perf_counter()

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 지금 마스킹 상태는 BGR 이미지에서 적색 영역, 녹색 영역 한 번 거르고 (video)
        # HSV 이미지에서 H 영역으로 분리한 (Intensity 무시) 영상과 AND 연산해주고 있음
        # 레이저가 겹칠때도 좌표에 영향을 안받으려면 ..

        mask_intensity = (
            ((frame_hsv[:, :, 2] > 230)*255).astype(np.uint8)
        )

        mask_red = (
            ((frame[:, :, 2] > 190)*255).astype(np.uint8)
        )

        mask_green = (
            ((frame[:, :, 1] > 180) * 255).astype(np.uint8)
        )

        lower_r = np.array([-10, 50, 0])
        upper_r = np.array([10, 100, 255])
        frame_r = cv2.inRange(frame_hsv, lower_r, upper_r)

        lower_g = np.array([30, 30, 0])
        upper_g = np.array([80, 80, 255])
        frame_g = cv2.inRange(frame_hsv, lower_g, upper_g)

        # video_red = cv2.bitwise_and(frame_r, mask_red)
        # video_green = cv2.bitwise_and(frame_g, mask_green)

        video_red = cv2.bitwise_and(frame_r, mask_intensity)
        video_green = cv2.bitwise_and(frame_g, mask_intensity)

        # red_sum_y = video_red.sum(0) > 1500
        # red_sum_x = video_red.sum(1) > 1500
        # red_y = np.where(red_sum_y == True)[0]
        # red_x = np.where(red_sum_x == True)[0]
        #
        # if np.any(np.isnan(red_y)) == True or np.any(np.isnan(red_x)) == True or red_y.size == 0 or red_x.size == 0:
        #     red_y = 0
        #     red_x = 0
        # else:
        #     red_y = int(red_y.mean())
        #     red_x = int(red_x.mean())
        #     cv2.circle(video_red, (red_x, red_y), 3, 255, -1)



        # frame[:, :, 1] = video_green
        # frame[:, :, 2] = video_red

        keypoints_g = detector.detect(video_green)
        keypoints_r = detector.detect(video_red)

        # 적색 레이저 인식 (lz_x_r,lz_y_r)
        for point in keypoints_r:
            lz_x_r = point.pt[0]
            lz_y_r = point.pt[1]
            lazer_x_red.set("lazer_x (red) : " + str(int(lz_x_r)))
            lazer_y_red.set("lazer_y (red) : " + str(int(lz_y_r)))
            cv2.circle(video_red, (int(lz_x_r), int(lz_y_r)), 3, (255,0,0), -1)

        # 녹색 레이저 인식 (lz_x_g, lz_y_g)
        for point in keypoints_g:
            lz_x_g = point.pt[0]
            lz_y_g = point.pt[1]
            lazer_x_green.set("lazer_x (green) : " + str(int(lz_x_g)))
            lazer_y_green.set("lazer_y (green) : " + str(int(lz_y_g)))
            cv2.circle(video_red, (int(lz_x_g), int(lz_y_g)), 3, (0,255,0), -1)


        # 골대 인식을 위한 코너 인식
        frame_gray_threshold = np.where(frame_gray < 120, frame_gray, 255)
        corners = cv2.goodFeaturesToTrack(frame_gray_threshold, 3, 0.5, 10)

        # 골대 인식 (px,py)
        if corners is not None:
            corners = np.int0(corners).reshape(
                corners.shape[0], corners.shape[2])

            xy_12 = corners[corners[:, 0].argsort(kind='mergesort')]
            xy_34 = corners[corners[:, 1].argsort(kind='mergesort')]
            x1, y1 = xy_12[0]
            x2, y2 = xy_12[-1]
            x3, y3 = xy_34[0]
            x4, y4 = xy_34[-1]

            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
                 ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
                 ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

            if np.isnan(px) == False and np.isnan(py) == False:
                cv2.circle(frame, (int(px), int(py)), 10, (0, 0, 255), 3)
                goal_x.set("goal_x : " + str(int(px)))
                goal_y.set("goal_y : " + str(int(py)))

            for i in corners:
                x, y = i.ravel()
                cv2.circle(frame, (x, y), 3, 255, -1)
        else:
            # 골대가 영상에 잡히지 않을 때 좌표 (0,0) 지정
            goal_x.set("goal_x : 0")
            goal_y.set("goal_y : 0")

        # 비디오 FPS 출력
        terminate_t = time.perf_counter()
        fps = int(1. / (terminate_t - start_t))
        cv2.putText(frame, "FPS : " + str(fps), (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        # 출력할 영상 지정 : 원본 영상은 frame, 테스트 원하는 영상 주석 풀고 사용하면 됨
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # img = Image.fromarray(frame_r)
        # img = Image.fromarray(frame_g)
        # img = Image.fromarray(mask_red)
        # img = Image.fromarray(mask_green)
        # img = Image.fromarray(mask_intensity)
        # img = Image.fromarray(video_red)
        # img = Image.fromarray(video_green)


        imgtk = ImageTk.PhotoImage(image=img)
        label2.imgtk = imgtk
        label2.configure(image=imgtk)

        if np.isnan(px) == False and np.isnan(py) == False:
            return int(px), int(py), int(lz_x_r), int(lz_y_r), int(lz_x_g), int(lz_y_g)
        else:
            return 0, 0, int(lz_x_r), int(lz_y_r), int(lz_x_g), int(lz_y_g)
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

# 캠 포트 설정 : 이상한 카메라가 뜬다면 0,1 등 바꿔보세요
cap = cv2.VideoCapture(1)

# video_play()
g_x, g_y, lz_x_r, lz_y_r, lz_x_g, lz_y_g = video_play()

window.mainloop()
