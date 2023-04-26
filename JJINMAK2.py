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


def motor_move(goal_x, goal_y, lazer_x, lazer_y):
    diff_x = goal_x - lazer_x  # 양수면 오른쪽으로 가야함, 음수면 왼쪽으로 가야함
    diff_y = goal_y - lazer_y  # 양수면 위로 가야함, 음수면 아래로 가야함
    if diff_x > 0:
        x_loc -= 1
    else:
        x_loc += 1
    if diff_y > 0:
        y_loc -= 1
    else:
        y_loc += 1

    ser_loc = str(x_loc) + " " + str(y_loc) + "\n"
    ser_loc = ser.encode('utf-8')
    ser.write(ser_loc)


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
    # params.minDistBetweenBlobs = 100
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

        # 적/녹 색상 추출을 위해 HSVscale로 변환 : frame_hsv
        # 골대 인식을 위해 Grayscale로 변환 : frame_gray
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, i = cv2.split(frame_hsv)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # a = np.where(frame_hsv[:, :, 2] < 150)[0]

        # 위에서 만든 필터 마스크와 원본 영상(frame)을 논리합 연산
        # 적색만 인식하여 화면에 출력하는 영상 (video_red)
        # 녹색만 인식하여 화면에 출력하는 영상 (vidoe_green)

        # pdb.set_trace()
        video_red = (
            ((frame[:, :, 2] > 150)*255).astype(np.uint8)
        )

        # sss = video_red.sum(0) > 1500
        # sss2 = video_red.sum(1) > 1500
        # a = np.where(sss == True)[0]
        # b = np.where(sss2 == True)[0]
        # if np.any(np.isnan(a)) == True or np.any(np.isnan(b)) == True or a.size == 0 or b.size == 0:
        #     video_red2 = video_red
        # else:
        #     a = int(a.mean())
        #     b = int(b.mean())
        #     video_red2 = np.zeros((1080, 1920))
        #     video_red2[b-10:b+10, a-10:a+10] = 255
        #     video_red2 = video_red2.astype(np.uint8)

        # pdb.set_trace()

        video_green = (
            ((frame[:, :, 1] > 180)*255).astype(np.uint8)
        )

        # ssss = video_green.sum(0) > 3000
        # ssss2 = video_green.sum(1) > 3000
        # aa = np.where(ssss == True)[0]
        # bb = np.where(ssss2 == True)[0]
        # if np.any(np.isnan(aa)) == True or np.any(np.isnan(bb)) == True or aa.size == 0 or bb.size == 0:
        #     video_green2 = video_green
        # else:
        #     aa = int(aa.mean())
        #     bb = int(bb.mean())
        #     video_green2 = np.zeros((480, 640))
        #     video_green2[bb-20:bb+20, aa-20:aa+20] = 255
        #     video_green2 = video_green2.astype(np.uint8)

        lower_r = np.array([-10, 50, 0])
        upper_r = np.array([10, 100, 255])
        # erode. dilate는 추출 성능을 올리기 위한 과정이니 무시해도 됨
        frame_r = cv2.inRange(frame_hsv, lower_r, upper_r)
        # frame_r = cv2.erode(frame_r, None, iterations=0)
        # frame_r = cv2.dilate(frame_r, None, iterations=0)

        # 녹색 필터 마스크
        # upper_g = np.array([70, 255, 255])
        # lower_g = np.array([40, 70, 100])
        lower_g = np.array([30, 30, 0])
        upper_g = np.array([80, 80, 255])
        frame_g = cv2.inRange(frame_hsv, lower_g, upper_g)
        # frame_g = cv2.erode(frame_g, None, iterations=0)
        # frame_g = cv2.dilate(frame_g, None, iterations=0)

        # 위에서 만든 필터 마스크와 원본 영상(frame)을 논리합 연산
        # 적색만 인식하여 화면에 출력하는 영상 (video_red)
        # 녹색만 인식하여 화면에 출력하는 영상 (vidoe_green)
        video_red = cv2.bitwise_and(frame_r, video_red)
        video_green = cv2.bitwise_and(frame_g, video_green)
        frame[:, :, 1] = video_green
        frame[:, :, 2] = video_red

        # pdb.set_trace()
        # np.add를 통해 적,녹 비디오 합쳐서 frame 변수에 저장

        # (실제 점 인식은 이거 안 쓰고 위에서 뽑은 마스크를 통해 진행)
        # np.add(video_red, video_green, frame)
        # frame = cv2.erode(frame, None, iterations=0)
        # frame = cv2.dilate(frame, None, iterations=0)
        keypoints_g = detector.detect(video_green)
        keypoints_r = detector.detect(video_red)

        # im_with_keypoints = cv2.drawKeypoints(
        #     frame_r, keypoints_r, video_red, (0, 2500, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # 적색 레이저 인식
        for point in keypoints_r:
            lz_x_r = point.pt[0]
            lz_y_r = point.pt[1]
            lazer_x_red.set("lazer_x (red) : " + str(int(lz_x_r)))
            lazer_y_red.set("lazer_y (red) : " + str(int(lz_y_r)))

        # 녹색 레이저 인식
        for point in keypoints_g:
            lz_x_g = point.pt[0]
            lz_y_g = point.pt[1]
            lazer_x_green.set("lazer_x (green) : " + str(int(lz_x_g)))
            lazer_y_green.set("lazer_y (green) : " + str(int(lz_y_g)))

        # 골대 인식
        sss = np.where(frame_gray < 120, frame_gray, 255)
        corners = cv2.goodFeaturesToTrack(sss, 3, 0.5, 10)

        # 골대가 비디오 내부에 잡힐 경우 좌표 (px_b,py_b)을 출력
        if corners is not None:
            corners = np.int0(corners).reshape(
                corners.shape[0], corners.shape[2])

            # xy_b_12 = corners[corners[:, 0].argsort(kind='mergesort')]
            # xy_b_34 = corners[corners[:, 1].argsort(kind='mergesort')]
            # x1_b, y1_b = xy_b_12[0]
            # x2_b, y2_b = xy_b_12[-1]
            # x3_b, y3_b = xy_b_34[0]
            # x4_b, y4_b = xy_b_34[-1]

            # px_b = ((x1_b * y2_b - y1_b * x2_b) * (x3_b - x4_b) - (x1_b - x2_b) * (x3_b * y4_b - y3_b * x4_b)) / \
            #     ((x1_b - x2_b) * (y3_b - y4_b) - (y1_b - y2_b) * (x3_b - x4_b))

            # py_b = ((x1_b * y2_b - y1_b * x2_b) * (y3_b - y4_b) - (y1_b - y2_b) * (x3_b * y4_b - y3_b * x4_b)) / \
            #     ((x1_b - x2_b) * (y3_b - y4_b) - (y1_b - y2_b) * (x3_b - x4_b))

            px_b = corners[:, 0].sum() / len(corners)
            py_b = corners[:, 1].sum() / len(corners)

            if np.isnan(px_b) == False and np.isnan(py_b) == False:
                cv2.circle(frame, (int(px_b), int(py_b)), 10, (0, 0, 255), 3)
                goal_x.set("goal_x : " + str(int(px_b)))
                goal_y.set("goal_y : " + str(int(py_b)))

            for i in corners:
                x_b, y_b = i.ravel()
                cv2.circle(frame, (x_b, y_b), 3, 255, -1)
        else:
            # 골대가 영상에 잡히지 않을 때 좌표 (0,0) 지정
            goal_x.set("goal_x : 0")
            goal_y.set("goal_y : 0")

        # 비디오 FPS 출력
        terminate_t = time.perf_counter()
        fps = int(1. / (terminate_t - start_t))
        cv2.putText(frame, "FPS : " + str(fps), (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        # 1000x600 사이즈의 비디오 출력 화면 지정 (출력원하는 img변수 주석 풀고 사용하면 됨)
        # 1. 빨 + 초 같이
        img = Image.fromarray(video_red)
        imgtk = ImageTk.PhotoImage(image=img)
        label2.imgtk = imgtk
        label2.configure(image=imgtk)
        if judge == False:
            label2.after(10, video_play)

        #     return int(px_b), int(py_b), int(lz_x_r), int(lz_y_r), int(lz_x_g), int(lz_y_g)
        # else:
        #     return 0, 0, int(lz_x_r), int(lz_y_r), int(lz_x_g), int(lz_y_g)
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

# if (lz_x_r != 0 and lz_y_r != 0) and (lz_x_g != 0 and lz_y_g != 0) and lz_x_r == lz_x_g and lz_y_r == lz_y_g:
#     win_g = tk.StringVar(window)
#     win_g.set("수비팀 승리!")
#     win_g_label = tk.Label(window, textvariable=win_g, font=font)
#     win_g_label.place(x=750, y=400)
#     judge = True

# if judge == False:
# motor_move(g_x, g_y, lz_x_r, lz_y_r, lz_x_g, lz_y_g)


window.mainloop()
