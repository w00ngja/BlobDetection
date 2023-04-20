import tkinter as tk
import tkinter.font
import serial
import cv2
import numpy as np
import timeit
from PIL import ImageTk, Image
import warnings

warnings.filterwarnings(action='ignore')
prevTime = 0

def video_play():
    global prevTime
    ret, frame = cap.read()

    # 점 인식 함수 초기값 지정
    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = 255
    params.minThreshold = 100
    params.maxThreshold = 255
    detector = cv2.SimpleBlobDetector_create(params)

    if ret:
        start_t = timeit.default_timer()

        # 적/녹 색상 추출을 위해 HSVscale로 변환 : frame_hsv
        # 골대 인식을 위해 Grayscale로 변환 : frame_gray
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_hsv = cv2.cvtColor(frame_hsv, cv2.COLOR_RGB2HSV)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 적색 필터 마스크
        # erode. dilate는 추출 성능을 올리기 위한 과정이니 무시해도 됨
        # frame_r = cv2.inRange(frame_hsv, (150, 100, 100), (180, 255, 255))
        frame_r = cv2.inRange(frame_hsv, (-5, 255, 255), (7, 255, 255))

        # frame_r = cv2.inRange(frame_hsv, (-10, 70, 50), (7, 255, 255))
        frame_r = cv2.erode(frame_r, None, iterations=0)
        frame_r = cv2.dilate(frame_r, None, iterations=0)

        # 녹색 필터 마스크
        frame_g = cv2.inRange(frame_hsv, (36, 25, 25), (70, 255, 255))
        frame_g = cv2.erode(frame_g, None, iterations=0)
        frame_g = cv2.dilate(frame_g, None, iterations=0)

        # 위에서 만든 필터 마스크와 원본 영상(frame)을 논리합 연산
        # 적색만 인식하여 화면에 출력하는 영상 (video_red)
        # 녹색만 인식하여 화면에 출력하는 영상 (vidoe_green)
        video_red = cv2.bitwise_and(frame,frame,mask=frame_r)
        video_green = cv2.bitwise_and(frame, frame, mask=frame_g)

        # np.add를 통해 적,녹 비디오 합쳐서 frame 변수에 저장
        # (실제 점 인식은 이거 안 쓰고 위에서 뽑은 마스크를 통해 진행)
        np.add(video_red, video_green, frame)
        frame = cv2.erode(frame, None, iterations=0)
        frame = cv2.dilate(frame, None, iterations=0)

        # 색깔 별 필터 마스크에서 키포인트 추출
        keypoints_g = detector.detect(frame_g)
        keypoints_r = detector.detect(frame_r)

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
            lazer_x_green.set("lazer_y (green) : " + str(int(lz_y_g)))

        # 골대 인식
        corners = cv2.goodFeaturesToTrack(frame_gray, 5, 0.04, 10)

        # 골대가 비디오 내부에 잡힐 경우 좌표 (px_b,py_b)을 출력
        if corners is not None:
            corners = np.int0(corners).reshape(corners.shape[0], corners.shape[2])

            xy_b_12 = corners[corners[:, 0].argsort(kind='mergesort')]
            xy_b_34 = corners[corners[:, 1].argsort(kind='mergesort')]
            x1_b, y1_b = xy_b_12[0];
            x2_b, y2_b = xy_b_12[-1];
            x3_b, y3_b = xy_b_34[0];
            x4_b, y4_b = xy_b_34[-1]

            px_b = ((x1_b * y2_b - y1_b * x2_b) * (x3_b - x4_b) - (x1_b - x2_b) * (x3_b * y4_b - y3_b * x4_b)) / \
                 ((x1_b - x2_b) * (y3_b - y4_b) - (y1_b - y2_b) * (x3_b - x4_b))

            py_b = ((x1_b * y2_b - y1_b * x2_b) * (y3_b - y4_b) - (y1_b - y2_b) * (x3_b * y4_b - y3_b * x4_b)) / \
                 ((x1_b - x2_b) * (y3_b - y4_b) - (y1_b - y2_b) * (x3_b - x4_b))

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
        terminate_t = timeit.default_timer()
        fps = int(1. / (terminate_t - start_t))
        cv2.putText(frame, "FPS : " + str(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        ### 1000x600 사이즈의 비디오 출력 화면 지정 (출력원하는 img변수 주석 풀고 사용하면 됨)
        # 1. 빨 + 초 같이
        # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 2. 빨강만
        # img = Image.fromarray(cv2.cvtColor(video_red, cv2.COLOR_BGR2RGB))

        #3. 초록만
        img = Image.fromarray(cv2.cvtColor(video_green, cv2.COLOR_BGR2RGB))

        imgtk = ImageTk.PhotoImage(image=img)
        label2.imgtk = imgtk
        label2.configure(image=imgtk)
        label2.after(10, video_play)
    else:
        cap.release()
        return


#tkinter UI 설정부분
window = tk.Tk()
window.title('control')
window.geometry("1000x600+500+100")
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

cap = cv2.VideoCapture(1)
video_play()

window.mainloop()




