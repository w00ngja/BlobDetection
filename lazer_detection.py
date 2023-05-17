import tkinter as tk
import tkinter.font
import serial
import cv2
import numpy as np
import time
from time import localtime
from PIL import ImageTk, Image
import warnings

warnings.filterwarnings(action='ignore')
prevTime = 0
prevTime2 = 0

judge = False

isTimer = 0

def video_play():
    global prevTime,prevTime2, judge, isTimer
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

    # blob 탐지. 빨간색, 초록색 레이저를 잡아냄.
    detector = cv2.SimpleBlobDetector_create(params)

    if ret:
        # 원본영상(frame)에서 레이저 인식을 위한 HSV변환(frame_hsv)과 Grayscale변환(frame_gray)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        video_red = (
            # 적색(G, B, R)만 인식하여 화면에 출력하는 영상 (video_red)
            ((frame[:, :, 2] > 190)*255).astype(np.uint8)
        )

        video_green = (
            ((frame[:, :, 1] > 180)*255).astype(np.uint8)
        )

        # HSV영역에서 색상 범위를 지정하여 적, 녹 필터링 마스크 생성 (H : 채도, S : 농도 , V : 밝기)
        lower_r = np.array([-10, 50, 0])
        upper_r = np.array([10, 100, 255])
        # HSV로 빨강색 부분만 한번 더 추출
        frame_r = cv2.inRange(frame_hsv, lower_r, upper_r)

        # 녹색 필터 마스크
        lower_g = np.array([30, 30, 0])
        upper_g = np.array([80, 80, 255])
        # HSV로 녹색 부분만 한번 더 추출
        frame_g = cv2.inRange(frame_hsv, lower_g, upper_g)

        # RGB를 이용해 뽑아낸 것과 HSV를 이용해 뽑아낸 것을 and 연산으로 합침
        video_red = cv2.bitwise_and(frame_r, video_red)
        video_green = cv2.bitwise_and(frame_g, video_green)

        # 레이저 포인트 감지.
        keypoints_r = detector.detect(video_red)
        keypoints_g = detector.detect(video_green)

        # 적색 레이저 인식
        for point in keypoints_r:
            lz_x_r = point.pt[0]
            lz_y_r = point.pt[1]
            # tkinter에 레이저 좌표 출력
            lazer_x_red.set("lazer_x (red) : " + str(int(lz_x_r)))
            lazer_y_red.set("lazer_y (red) : " + str(int(lz_y_r)))
            cv2.circle(frame, (int(lz_x_r), int(lz_y_r)), 10, (0, 255, 0), 2)

        # 녹색 레이저 인식
        for point in keypoints_g:
            lz_x_g = point.pt[0]
            lz_y_g = point.pt[1]
            # tkinter에 레이저 좌표 출력
            lazer_x_green.set("lazer_x (green) : " + str(int(lz_x_g)))
            lazer_y_green.set("lazer_y (green) : " + str(int(lz_y_g)))
            cv2.circle(frame, (int(lz_x_g), int(lz_y_g)), 10, (0, 0, 255), 2)

        # 골대 인식 및 화면에 출력 (px_b, py_b)
        # 잘 잡히지 않을 시 goodFeaturesToTrack 파라미터 바꿔볼 것
        corners = cv2.goodFeaturesToTrack(
            frame_gray, 5, 0.04, 10, corners=None, mask=None, blockSize=None, useHarrisDetector=True, k=0.04)

        # 골대가 비디오 내부에 잡힐 경우 좌표 (px_b,py_b)을 출력
        if corners is not None:
            corners = np.int0(corners).reshape(
                corners.shape[0], corners.shape[2])

            px_b = corners[:, 0].sum() / len(corners)
            py_b = corners[:, 1].sum() / len(corners)

            if np.isnan(px_b) == False and np.isnan(py_b) == False:
                # 골대가 비디오에 잡힐 경우 좌표 출력
                cv2.circle(frame, (int(px_b), int(py_b)), 20, (255, 0, 0), 3)
                goal_x.set("goal_x : " + str(int(px_b)))
                goal_y.set("goal_y : " + str(int(py_b)))

            for i in corners:
                x_b, y_b = i.ravel()
                cv2.circle(frame, (x_b, y_b), 3, (255, 0, 0), -1)
        else:
            # 골대가 영상에 잡히지 않을 때 좌표 (0,0) 지정
            goal_x.set("goal_x : 0")
            goal_y.set("goal_y : 0")

        # 비디오 FPS 출력
        curTime = time.time()

        sec = curTime - prevTime
        mi = time.perf_counter()
        # time.sleep(1)
        if (isTimer):
            print(int(time.process_time()-isTimer))

        prevTime = curTime
        fps = int(1/(sec))
        cv2.putText(frame, "FPS : " + str(fps), (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        # print(isTimer)


        # 이미지 출력
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        imgtk = ImageTk.PhotoImage(image=img)
        label2.imgtk = imgtk
        label2.configure(image=imgtk)
        label2.after(10, video_play)
    else:
        cap.release()
        return

def timer():
    global isTimer
    isTimer = time.process_time()

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

button = tkinter.Button(window, overrelief="solid", width=15, command=timer, repeatdelay=1000, repeatinterval=100)
button.pack()
button.place(x=750, y=450)
# 캠 선택 : 거의 기본 캠(0)인데, 안나오면 1로 바꿔볼 것
cap = cv2.VideoCapture(0)


# 함수 실행 및 무한 루프
video_play()
window.mainloop()
