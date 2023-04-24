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

    params = cv2.SimpleBlobDetector_Params()
    params.blobColor = 255
    params.minThreshold = 100
    params.maxThreshold = 255
    detector = cv2.SimpleBlobDetector_create(params)

    if ret:
        start_t = timeit.default_timer()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 빨간 점 필터 마스크 (keypoints 추출에 사용)
        frame_r = cv2.inRange(frame, (170, 50, 50), (180, 255, 255))
        frame_r = cv2.erode(frame_r, None, iterations=0)
        frame_r = cv2.dilate(frame_r, None, iterations=0)

        # 초록 점 필터 마스크 (keypoints 추출에 사용)
        frame_g = cv2.inRange(frame, (25, 57, 2), (102, 255, 255))
        frame_g = cv2.erode(frame_g, None, iterations=0)
        frame_g = cv2.dilate(frame_g, None, iterations=0)

        # 빨간 점만 추출
        dst1 = cv2.bitwise_and(frame,frame,mask=frame_r)

        # 초록 점만 추출
        dst2 = cv2.bitwise_and(frame,frame,mask=frame_g)

        # 합쳐서 frame으로 넘겼음 -> (초록 + 빨간 점 확인 위한 화면 출력용으로, 실제 키포인트 인식은 개별 마스크를 통해 진행)
        np.add(dst1,dst2,frame)
        frame = cv2.erode(frame, None, iterations=0)
        frame = cv2.dilate(frame, None, iterations=0)

        h, w = gray.shape[:2]
        keypoints = detector.detect(gray)
        keypoints_g = detector.detect(frame_g)
        keypoints_r = detector.detect(frame_r)

        # 초록 점 찾기
        for point in keypoints_g:
            lz_x_g = point.pt[0]
            lz_y_g = point.pt[1]
            lazer_g_x.set("lazer_x (green) : " + str(int(lz_x_g)))
            lazer_g_y.set("lazer_y (green) : " + str(int(lz_y_g)))

        # 빨간 점 찾기
        for point in keypoints_r:
            lz_x_r = point.pt[0]
            lz_y_r = point.pt[1]
            lazer_r_x.set("lazer_x (red) : " + str(int(lz_x_r)))
            lazer_r_y.set("lazer_y (red) : " + str(int(lz_y_r)))

        # im_with_keypoints_g = cv2.drawKeypoints(frame_g, keypoints_g, np.array([]), (0, 2500, 0),
        #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints_r = cv2.drawKeypoints(frame_r, keypoints_r, dst1, (0, 2500, 0),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 2500, 0),
                                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        corners = cv2.goodFeaturesToTrack(gray, 5, 0.04, 10)
        corners = np.int0(corners).reshape(corners.shape[0], corners.shape[2])

        xy_12 = corners[corners[:, 0].argsort(kind='mergesort')]
        xy_34 = corners[corners[:, 1].argsort(kind='mergesort')]
        x1, y1 = xy_12[0];
        x2, y2 = xy_12[-1];
        x3, y3 = xy_34[0];
        x4, y4 = xy_34[-1]

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
             ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
             ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        if np.isnan(px) == False and np.isnan(py) == False:
            ##            print(int(px), int(py))
            cv2.circle(im_with_keypoints, (int(px), int(py)), 10, (0, 0, 255), 3)
            cross_x.set("cross_x : " + str(int(px)))
            cross_y.set("cross_y : " + str(int(py)))
        for i in corners:
            x, y = i.ravel()
            cv2.circle(im_with_keypoints, (x, y), 3, 255, -1)

        terminate_t = timeit.default_timer()
        fps = int(1. / (terminate_t - start_t))

        cv2.putText(frame, "FPS : " + str(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        # frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 출력 화면 지정
        img = Image.fromarray(cv2.cvtColor(dst1, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        label2.imgtk = imgtk
        label2.configure(image=imgtk)
        label2.after(10, video_play)
    else:
        cap.release()
        return


window = tk.Tk()
window.title('control')
window.geometry("1000x600+500+100")
window.resizable(False, False)

frm = tk.Frame(window, bg="white", width=500, height=400)
frm.place(x=80, y=30)
label2 = tk.Label(frm)
label2.grid()

cross_x = tk.StringVar(window)
cross_x.set("0")
cross_y = tk.StringVar(window)
cross_y.set("0")
lazer_g_x = tk.StringVar(window)
lazer_g_x.set("lazer_x (green) : 0")
lazer_g_y = tk.StringVar(window)
lazer_g_y.set("lazer_y (green) : 0")
lazer_r_x = tk.StringVar(window)
lazer_r_x.set("lazer_x (red) : 0")
lazer_r_y = tk.StringVar(window)
lazer_r_y.set("lazer_y (red) : 0")

font = tk.font.Font(size=20)
cross_x_label = tk.Label(window, textvariable=cross_x, font=font)
cross_x_label.place(x=750, y=100)
cross_y_label = tk.Label(window, textvariable=cross_y, font=font)
cross_y_label.place(x=750, y=130)

lazer_g_x_label = tk.Label(window, textvariable=lazer_g_x, font=font)
lazer_g_x_label.place(x=750, y=200)
lazer_g_y_label = tk.Label(window, textvariable=lazer_g_y, font=font)
lazer_g_y_label.place(x=750, y=230)

lazer_r_x_label = tk.Label(window, textvariable=lazer_r_x, font=font)
lazer_r_x_label.place(x=750, y=300)
lazer_r_y_label = tk.Label(window, textvariable=lazer_r_y, font=font)
lazer_r_y_label.place(x=750, y=330)

cap = cv2.VideoCapture(1)
video_play()

window.mainloop()

