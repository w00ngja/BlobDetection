import tkinter as tk
import serial
import cv2

class motor():

    x_loc = 90; y_loc = 90; x_ser = ''; y_ser = '';
    ser = serial.Serial(port = 'COM3',baudrate = 9600)
    
    def __init__(self):
        pass
    
    def x_loc_Up(self):
        self.x_loc -= 10
        window.label_x.config(text = "X = " + str(self.x_loc))
        self.serial_write(self.x_ser, self.y_loc, self.x_loc)
        
    def x_loc_Down(self):
        self.x_loc += 10
        window.label_x.config(text = "X = " + str(self.x_loc))
        self.serial_write(self.x_ser, self.y_loc, self.x_loc)
        

    def y_loc_Up(self):
        self.y_loc -= 10
        window.label_y.config(text = "Y = " + str(self.y_loc))
        self.serial_write(self.y_ser, self.y_loc, self.x_loc)
        
        
    def y_loc_Down(self):
        self.y_loc += 10
        window.label_y.config(text = "Y = " + str(self.y_loc))
        self.serial_write(self.y_ser, self.y_loc, self.x_loc)
        
    def serial_write(self, ser, loc1, loc2):
        ser = str(loc1) + " " + str(loc2) + "\n"
        ser = ser.encode('utf-8')
        self.ser.write(ser)
        print(ser)

class windowform():

    label_x,label_y = tk.Label(), tk.Label()
    button_up, button_down, button_left, button_right = 0,0,0,0
    x_loc, y_loc = 0, 0
    
    def __init__(self):

        self.window1 = tk.Tk()
        self.window1.title('control')
        self.window1.geometry("800x300+500+100")
        self.window1.resizable(False, False)

    def run(self):
        self.make_label(self.label_x, self.window1, "X = " + str(self.x_loc), 10, 500, 100)
        self.make_label(self.label_y,self.window1,"Y = " + str(self.y_loc),10,600,100)
        self.make_button(self.button_up, self.window1,"UP",motor_move.y_loc_Up,150,50)
        self.make_button(self.button_down, self.window1,"Down",motor_move.y_loc_Down,150,190)
        self.make_button(self.button_right, self.window1,"Left",motor_move.x_loc_Down,80,120)
        self.make_button(self.button_left, self.window1,"Right",motor_move.x_loc_Up,210,120)
        self.window1.mainloop()
        
    def make_label(self,label, window, string, font_size, xloc, yloc):
        label = tk.Label(window,text = string, font = font_size)
        label.place(x = xloc, y = yloc)

    def make_button(self, button, window, string, cmd, xloc, yloc):
        button = tk.Button(window, text = string, overrelief = "raised",width=10,height=2,\
                           command = cmd, repeatdelay = 1000, repeatinterval = 100)
        button.place(x = xloc, y = yloc)

motor_move = motor()
window = windowform()
window.run()
