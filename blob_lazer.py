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
        h, w = gray.shape[:2]
        keypoints = detector.detect(gray)
        for point in keypoints:
            lz_x = point.pt[0]
            lz_y = point.pt[1]
            lazer_x.set("lazer_x : " + str(int(lz_x)))
            lazer_y.set("lazer_y : " + str(int(lz_y)))
#
        im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,2500,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        edges = cv2.Canny(gray, 100, 200 )
        corners = cv2.goodFeaturesToTrack(gray, 5, 0.04, 10)

#         corners = np.int0(corners).reshape(corners.shape[0],corners.shape[2])
#
#         xy_12 = corners[corners[:,0].argsort(kind='mergesort')]
#         xy_34 = corners[corners[:,1].argsort(kind='mergesort')]
#         x1,y1 = xy_12[0]; x2,y2 = xy_12[-1]; x3,y3 = xy_34[0]; x4,y4 = xy_34[-1]
#
#         px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1-x2)*(x3*y4 - y3*x4))/\
#              ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
#
#         py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1-y2)*(x3*y4 - y3*x4))/\
#              ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
#
#         if np.isnan(px) == False and np.isnan(py)== False:
# ##            print(int(px), int(py))
#             cv2.circle(im_with_keypoints, (int(px),int(py)),10,(0,0,255),3)
#             cross_x.set("cross_x : " + str(int(px)))
#             cross_y.set("cross_y : " + str(int(py)))
#         for i in corners:
#             x, y = i.ravel()
#             cv2.circle(im_with_keypoints, (x,y), 3, 255, -1)

        terminate_t = timeit.default_timer()
        fps = int(1./(terminate_t - start_t))

        
        cv2.putText(im_with_keypoints,"FPS : " + str(fps),(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))  
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image = img)
        label2.imgtk = imgtk
        label2.configure(image = imgtk)
        label2.after(10, video_play)
    else:

        cap.release()
        return

window = tk.Tk()
window.title('control')
window.geometry("1000x600+500+100")
window.resizable(False, False)

frm = tk.Frame(window, bg = "white", width = 500, height = 400)
frm.place(x = 80, y = 30)
label2 = tk.Label(frm)
label2.grid()

cross_x = tk.StringVar(window)
cross_x.set("0")
cross_y = tk.StringVar(window)
cross_y.set("0")
lazer_x = tk.StringVar(window)
lazer_x.set("lazer_x : 0")
lazer_y = tk.StringVar(window)
lazer_y.set("lazer_y : 0")

font = tk.font.Font(size = 20)
cross_x_label= tk.Label(window, textvariable = cross_x, font = font)
cross_x_label.place(x =750, y = 100)
cross_y_label = tk.Label(window, textvariable = cross_y, font = font)
cross_y_label.place(x =750, y = 130)
lazer_x_label= tk.Label(window, textvariable = lazer_x, font = font)
lazer_x_label.place(x =750, y = 200)
lazer_y_label = tk.Label(window, textvariable = lazer_y, font = font)
lazer_y_label.place(x =750, y = 230)


cap = cv2.VideoCapture(1)
video_play()


window.mainloop()









