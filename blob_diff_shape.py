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

        # 원본 영상은 위의 cap.read()를 통해 frame 변수에 저장되고,
        # 색상 추출을 위해 hsv scale로 변환한 비디오를 frame_hsv에 저장
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 적색 필터 마스크 (Lazer 인식에 사용)
        # erode. dilate는 추출 성능을 올리기 위한 과정
        frame_r = cv2.inRange(frame_hsv, (-10, 100, 100), (10, 255, 255))
        frame_r = cv2.erode(frame_r, None, iterations=0)
        frame_r = cv2.dilate(frame_r, None, iterations=0)

        # 청색 필터 마스크 (골대 인식에 사용)
        frame_b = cv2.inRange(frame_hsv, (110,100,100), (130, 255, 255))
        frame_b = cv2.erode(frame_b, None, iterations=0)
        frame_b = cv2.dilate(frame_b, None, iterations=0)

        # 위에서 만든 적색 필터 마스크와 영상을 논리합 연산을 통해 빨간 점만 추출
        video_red = cv2.bitwise_and(frame,frame,mask=frame_r)

        # 위에서 만든 청색 필터 마스크와 영상을 논리합 연산을 통해 파란 점만 추출
        video_blue = cv2.bitwise_and(frame,frame,mask=frame_b)

        # np.add를 통해 빨,파 비디오 합쳐서 frame(원본영상)으로 넘겼음
        # (실제 점 인식은 이거 안 쓰고 위에서 뽑은 마스크를 통해 진행)
        np.add(video_red,video_blue,frame)
        frame = cv2.erode(frame, None, iterations=0)
        frame = cv2.dilate(frame, None, iterations=0)

        # 색깔 별 필터 마스크에서 키포인트 추출
        keypoints_b = detector.detect(frame_b)
        keypoints_r = detector.detect(frame_r)

        # 적색 원형 레이저 인식 (공격)
        for point in keypoints_r:
            lz_x_r = point.pt[0]
            lz_y_r = point.pt[1]
            lazer_circle_x.set("lazer_x (red) : " + str(int(lz_x_r)))
            lazer_circle_y.set("lazer_y (red) : " + str(int(lz_y_r)))

        #----------------------------------------------------------------------
        # 청색 십자형 인식 (골대)
        keypoints_blue_cross = cv2.drawKeypoints(frame_b, keypoints_b, np.array([]), (0, 2500, 0),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        corners_b = cv2.goodFeaturesToTrack(frame_b, 5, 0.04, 10)

        # 청색 십자가 비디오 내부에 잡힐 경우 좌표 (px_b,py_b)을 출력
        if corners_b is not None:
            corners_b = np.int0(corners_b).reshape(corners_b.shape[0], corners_b.shape[2])

            xy_b_12 = corners_b[corners_b[:, 0].argsort(kind='mergesort')]
            xy_b_34 = corners_b[corners_b[:, 1].argsort(kind='mergesort')]
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

            for i in corners_b:
                x_b, y_b = i.ravel()
                cv2.circle(frame, (x_b, y_b), 3, 255, -1)
        else:
            goal_x.set("goal_x : 0")
            goal_y.set("goal_y : 0")

        #----------------------------------------------------------------------
        # 적색 십자형 레이저 인식 (수비)
        keypoints_red_cross = cv2.drawKeypoints(frame_r, keypoints_r, np.array([]), (0, 2500, 0),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        corners_r = cv2.goodFeaturesToTrack(frame_r, 5, 0.04, 10)

        # 적색 십자가 비디오 내부에 잡힐 경우 좌표 (px_r,py_r)을 출력
        if corners_r is not None:
            corners_r = np.int0(corners_r).reshape(corners_r.shape[0], corners_r.shape[2])

            xy_r_12 = corners_r[corners_r[:, 0].argsort(kind='mergesort')]
            xy_r_34 = corners_r[corners_r[:, 1].argsort(kind='mergesort')]
            x1_r, y1_r = xy_r_12[0]
            x2_r, y2_r = xy_r_12[-1]
            x3_r, y3_r = xy_r_34[0]
            x4_r, y4_r = xy_r_34[-1]

            px_r = ((x1_r * y2_r - y1_r * x2_r) * (x3_r - x4_r) - (x1_r - x2_r) * (x3_r * y4_r - y3_r * x4_r)) / \
                 ((x1_r - x2_r) * (y3_r - y4_r) - (y1_r - y2_r) * (x3_r - x4_r))

            py_r = ((x1_r * y2_r - y1_r * x2_r) * (y3_r - y4_r) - (y1_r - y2_r) * (x3_r * y4_r - y3_r * x4_r)) / \
                 ((x1_r - x2_r) * (y3_r - y4_r) - (y1_r - y2_r) * (x3_r - x4_r))

            if np.isnan(px_r) == False and np.isnan(py_r) == False:
                cv2.circle(frame, (int(px_r), int(py_r)), 10, (0, 0, 255), 3)
                lazer_cross_x.set("lazer_x (cross) : " + str(int(px_r)))
                lazer_cross_y.set("lazer_y (cross) : " + str(int(py_r)))

            for i in corners_r:
                x_r, y_r = i.ravel()
                cv2.circle(frame, (x_r, y_r), 3, 255, -1)
        else:
            lazer_cross_x.set("lazer_x (cross) : 0")
            lazer_cross_y.set("lazer_y (cross) : 0")


        # 비디오에 초당 프레임 출력
        terminate_t = timeit.default_timer()
        fps = int(1. / (terminate_t - start_t))
        cv2.putText(frame, "FPS : " + str(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        # 1000x600 사이즈의 비디오 출력 화면 지정 (출력원하는 img변수 주석 풀고 사용하면 됨)
        # 1. 파랑 + 빨강 같이
        # img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 2. 빨강만
        img = Image.fromarray(video_red)

        #3. 파랑만
        # img = Image.fromarray(video_blue)

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
lazer_cross_x = tk.StringVar(window)
lazer_cross_x.set("lazer_x (cross) : 0")
lazer_cross_y = tk.StringVar(window)
lazer_cross_y.set("lazer_y (cross) : 0")
lazer_circle_x = tk.StringVar(window)
lazer_circle_x.set("lazer_x (circle) : 0")
lazer_circle_y = tk.StringVar(window)
lazer_circle_y.set("lazer_y (circle) : 0")

font = tk.font.Font(size=20)
goal_x_label = tk.Label(window, textvariable=goal_x, font=font)
goal_x_label.place(x=750, y=100)
goal_y_label = tk.Label(window, textvariable=goal_y, font=font)
goal_y_label.place(x=750, y=130)

lazer_cross_x_label = tk.Label(window, textvariable=lazer_cross_x, font=font)
lazer_cross_x_label.place(x=750, y=200)
lazer_cross_y_label = tk.Label(window, textvariable=lazer_cross_y, font=font)
lazer_cross_y_label.place(x=750, y=230)

lazer_circle_x_label = tk.Label(window, textvariable=lazer_circle_x, font=font)
lazer_circle_x_label.place(x=750, y=300)
lazer_circle_y_label = tk.Label(window, textvariable=lazer_circle_y, font=font)
lazer_circle_y_label.place(x=750, y=330)

cap = cv2.VideoCapture(1)
video_play()

window.mainloop()









