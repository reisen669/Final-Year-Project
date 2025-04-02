# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:23:43 2020
@author: Hor Sui Lyn 1161300122

This script is for GUI of a software called F-Filter to detect pornographic video

"""

import tkinter as tkt
from tkinter.font import Font 
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
import threading
import os
import cv2
import matplotlib 
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import numpy as np
from pathlib import Path
import loadmodel
from keras.preprocessing import image
import shutil

#///////////////////////////////////////////////////////////////////////////////////
#                       CONSTANTS AND GLOBAL VARS 
#///////////////////////////////////////////////////////////////////////////////////
NAME = "F-Filter" 
NAME_X = 10
NAME_Y = 15

fileLocFrameHeight = 200
frContainerWidth = 500
entryShortWidth = 4
entryWidth = 50
btnWidth = 10

lfEquation   = "   Equation for % Plot                                                                                                                         "
DEF_THRES = "40.0"

fontType = "Calibri"
btnBg = "midnight blue"
btnFg = "white"

OPTIONS = ["Method 1: Traditional CNN as Feature Extractor and Classifier",
           "Method 2: Fine-tuned CNN as Feature Extractor and Classifier",
           "Method 3: Traditional CNN as Feature Extractor and SVM as Classifier",
           "Method 4: Fine-tuned CNN as Feature Extractor and SVM as Classifier"
          ] 
FTYPES = [('MP4', '*.mp4'),   
          ('All files', '*') 
         ]
#          ('AVI files', '*.avi'), 
#          ('FLV', '*.flv'),  # semicolon trick
#          ('WMV', '*.wmv'), 
##          ('MOV', '*.mov;*.qt'),

LABEL = ["Non Porn",
         "Porn"
        ]
COL = 4
FPS = 25

global filepath 
global duration
global threshold
global method
global percentage
global vlabel

debug = True


### Load file dialog to allow selection of video film
def btnBrowse_clicked():
    if debug:
        print("Loading file dialog...") 
        
        fpath = askopenfilename(filetypes=FTYPES)
        if fpath:
            if debug:
                print("File selected: ", fpath)
            filepath.set(fpath)   
        else:   # fpath is empty string
            print('cancelled')
        
#///////////////////////////////////////////////////////////////////////////////////
#                       MASTER
#///////////////////////////////////////////////////////////////////////////////////
master = tkt.Tk()
master.title(NAME)
master.configure(padx=NAME_X, pady=NAME_Y)
master.resizable(height=False, width=False)
master.iconbitmap('f_filter.ico')

filepath = tkt.StringVar()
duration = tkt.StringVar()
threshold = tkt.StringVar()
method = tkt.StringVar() 
percentage = tkt.StringVar()
vlabel = tkt.StringVar()

# set default values
filepath.set("")
duration.set("10")
threshold.set(DEF_THRES)
method.set(OPTIONS[3]) 
percentage.set("-")
vlabel.set("-")

class SomeThread(threading.Thread):
    def __init__(self, path, wframe):
        threading.Thread.__init__(self)
        self.path = path
        self.wframe = wframe
        self.extractFrame()
        self.framenum = 0
    
    def extractFrame(self):
        vidObj = cv2.VideoCapture(self.path) # Path to video file 
        count = 0           # Used as counter variable
        keyf = 1
        success = 1         # checks whether frames were extracted 
        allimages = []
            
        fname = filepath.get().split("/")
        dirname = fname[-1].split(".")
        if debug:
            print(dirname[0])
            print("Creating folder...")
        if os.path.exists(Path(dirname[0])):
            shutil.rmtree(Path(dirname[0]))
        os.mkdir(Path(dirname[0]))
        self.path = dirname[0] # video file name without extension as folder name
        
        # read frames in the video
        while success: 
            success, image = vidObj.read() # read video to extract frames  
            count += 1  # count total number of frames in video
    
            if (count % (int(duration.get())*FPS)) == 0: # create list of key frames/frames to be extracted               
                # convert BGR images (OpenCV) into RGB
                image_rgb = np.fliplr(image.reshape(-1,3)).reshape(image.shape)
                allimages.append(image_rgb)
                cv2.imwrite(os.path.join(self.path , "frame%d.jpg" % keyf), image) # Saves the frames  
                keyf = keyf + 1
            
        print("Total number of frames: ", len(allimages))
        self.framenum = 0
        if debug:        
            self.dimension = allimages[0].shape
            print("Dimension of each image: ", allimages[0].shape)
            print(len(allimages), " frames") # total number of extracted frame
            self.framenum = len(allimages)
        
        if len(allimages) >= COL:
            col = COL
        else:
            col = len(allimages)
        self.wframe.plotBeforeFrame(allimages, math.ceil(len(allimages)/col), col, len(allimages))
        
            
class wholeframe:   
    def __init__(self,frame1):
        ### Define Font
        self.lfTitleFont = Font(family=fontType, size=14, weight="bold")
        self.btnFont = Font(family=fontType, size=12, weight="bold")
        self.enlblFont = Font(family=fontType, size=12)

        #///////////////////////////////////////////////////////////////////////////////////
        #                       CONTROLS
        #///////////////////////////////////////////////////////////////////////////////////
        self.frame1=frame1   
        # on the left side
        self.entFilePath   = tkt.Entry(self.frame1, textvariable=filepath, font=self.enlblFont, 
                                       width=entryWidth, state=tkt.DISABLED) 
        self.btnBrowse     = tkt.Button(self.frame1, text="Browse", font=self.btnFont, 
                                        width=btnWidth, bd=2, relief=tkt.RAISED,
                                       bg=btnBg, fg=btnFg, command=btnBrowse_clicked) 
        self.lblMethod     = tkt.Label(self.frame1, text="Method: ", font=self.enlblFont, 
                                       padx=10, pady=5) 
        self.optMethod     = tkt.OptionMenu(self.frame1, method, *OPTIONS)
        # on the right side 
        self.lblThreshold  = tkt.Label(self.frame1, text="Threshold: ", font=self.enlblFont, 
                                       padx=5, pady=5) 
        self.entThreshold  = tkt.Entry(self.frame1, textvariable=threshold, font=self.enlblFont, 
                                       justify=tkt.CENTER, width=entryShortWidth)
        self.lblPerSymbol1 = tkt.Label(self.frame1, text="%", font=self.enlblFont, pady=5)
        self.lblDuration   = tkt.Label(self.frame1, text="Duration(s): ", font=self.enlblFont, 
                                       padx=5, pady=5) 
        self.entDuration   = tkt.Entry(self.frame1, textvariable=duration, font=self.enlblFont, 
                                       justify=tkt.CENTER, width=entryShortWidth) 
        self.btnRun        = tkt.Button(self.frame1, text="Run", font=self.btnFont, width=btnWidth, 
                                        bd=2, relief=tkt.RAISED, bg=btnBg, fg=btnFg,
                                       command=self.btnRun_clicked)
        self.lblVideoClass = tkt.Label(self.frame1, text="Video Label: ", font=self.btnFont, 
                                       pady=5)
        self.entVideoClass = tkt.Entry(self.frame1, textvariable=vlabel, font=self.btnFont, 
                                       justify=tkt.CENTER, width=entryShortWidth + 10, 
                                       state="readonly")
        self.lblPercentage = tkt.Label(self.frame1, text="Percentage: ", font=self.enlblFont, 
                                       pady=5)
        self.entPercentage = tkt.Entry(self.frame1, textvariable=percentage, font=self.enlblFont, 
                                       justify=tkt.CENTER, width=entryShortWidth, state="readonly")
        self.lblPerSymbol2 = tkt.Label(self.frame1, text="%", font=self.enlblFont, pady=5)
        # sections at the bottom
        self.lblBefore     = tkt.Label(self.frame1, text="Before", font=self.lfTitleFont, 
                                       justify=tkt.CENTER) 
        self.frContainer1  = tkt.Frame(self.frame1, width=frContainerWidth, borderwidth=1, 
                                       relief=tkt.SOLID)
        self.lblAfter      = tkt.Label(self.frame1, text="After", font=self.lfTitleFont, padx=25, 
                                       justify=tkt.CENTER) 
        self.frContainer2  = tkt.Frame(self.frame1, width=frContainerWidth, bd=0.5,  
                                       relief=tkt.SOLID)
        
        #///////////////////////////////////////////////////////////////////////////////////
        #                       POSITIONING
        #///////////////////////////////////////////////////////////////////////////////////
        ### Controls in Master
        # on the left side [column 0 to 7]
        self.entFilePath.grid(row=0, column=0, padx=5, pady=5, columnspan=6, sticky="w")
        self.btnBrowse.grid(row=0, column=6, padx=5, pady=2, columnspan=2)
        self.lblMethod.grid(row=1, column=0, sticky="w")
        self.optMethod.grid(row=1, column=1, padx=1, columnspan=7, sticky="e")
        # on the right side [column 8 to 15]
        self.lblThreshold.grid(row=0, column=8, sticky="e")
        self.entThreshold.grid(row=0, column=9, padx=1, sticky="w")
        self.lblPerSymbol1.grid(row=0, column=10, sticky="w")
        self.lblDuration.grid(row=0, column=11, padx=1, sticky="e")
        self.entDuration.grid(row=0, column=12, padx=(0,20), sticky="w")
        self.btnRun.grid(row=0, column=12, padx=(0,10), columnspan=4, sticky="e") 
        self.lblVideoClass.grid(row=1, column=8, sticky="e")
        self.entVideoClass.grid(row=1, column=9, columnspan=3,sticky="w")
        self.lblPercentage.grid(row=1, column=12, padx=(0,10), columnspan=2, sticky="e")
        self.entPercentage.grid(row=1, column=14, sticky="e")
        self.lblPerSymbol2.grid(row=1, column=15, sticky="w")
        # sections at the bottom
        self.lblBefore.grid(row=2, column=0, columnspan=7, pady=(10,0), sticky="ew")
        self.frContainer1.grid(row=3, column=0, padx=(10,0), pady=(0,15), columnspan=7, sticky="e")   
        self.lblAfter.grid(row=2, column=8, columnspan=6, pady=(10,0), sticky="ew")
        self.frContainer2.grid(row=3, column=8, padx=(25,0), pady=(0,15), columnspan=5, sticky="e")
        
    #///////////////////////////////////////////////////////////////////////////////////
    #                       METHOD 
    #///////////////////////////////////////////////////////////////////////////////////    
    def plotBeforeFrame(self, allimages, row, col, fcount):
        count = 0
        try: 
            self.canvas1.get_tk_widget().pack_forget()
        except AttributeError: 
            print("No more subplot to be cleared")
            pass
        
        fig1 = Figure(dpi=100) 
        while count < fcount:
            ax = fig1.add_subplot(row, col, count+1)
            ax.imshow(allimages[count]/255., aspect='auto')
            ax.set_title("Frame "+str(count+1), fontsize=5)
            ax.axis('off')
            count = count + 1
        
        fig1.tight_layout(pad=0, w_pad=0.1, h_pad=0.1)
        print("Finished plotting!")

        self.canvas1 = FigureCanvasTkAgg(fig1, self.frContainer1) 
        self.canvas1.get_tk_widget().pack(side=tkt.TOP, fill=tkt.BOTH, expand=1)
        self.canvas1.draw_idle() # draw

    def plotAfterFrame(self, allimages, row, col, fcount):
        count = 0
        try: 
            self.canvas2.get_tk_widget().pack_forget()
        except AttributeError: 
            print("No more subplot to be cleared")
            pass
        
        fig2 = Figure(dpi=100) 
        while count < fcount:
            ax = fig2.add_subplot(row, col, count+1)
            ax.imshow(allimages[count]/255., aspect='auto')
            ax.set_title("Frame "+str(count+1), fontsize=5)
            ax.axis('off')
            count = count + 1
        
        fig2.tight_layout(pad=0, w_pad=0.1, h_pad=0.1)
        print("Finished plotting!")

        self.canvas2 = FigureCanvasTkAgg(fig2, self.frContainer2) 
        self.canvas2.get_tk_widget().pack(side=tkt.TOP, fill=tkt.BOTH, expand=1)
        self.canvas2.draw_idle()
        
    def filterFrame(self, result, num, path, dimension):
        count = 0
        filtered = 0
        afterimages = []
        for img_file in os.listdir(path):
            try:
                jpg = "frame" + str(count+1) + ".jpg"
                img = image.load_img(os.path.join(path, jpg), target_size=(dimension[1], dimension[0]))
                x = image.img_to_array(img)
                
                # output original frame if "Non Porn", else output black frame
                if result[count] == 1:
                    x = np.zeros(x.shape, dtype='float')
                    filtered = filtered + 1 # count number of filtered frame
                afterimages.append(x)
                
                count = count + 1
                
            except Exception as e:
                print("Exception: ", e)
                pass
        
        # set percentage accordingly
        if debug:
            print("Percentage: ", float(filtered/num), " (", filtered, "/", num, ")")
        percentage.set(str((float("{0:.3f}".format(filtered/num)))*100))
        
        # set video label accordingly
        if percentage.get() >= threshold.get():
            vlabel.set(LABEL[1]) 
        else:
            vlabel.set(LABEL[0])
        if count >= COL:
            col = COL
        else:
            col = count
        self.plotAfterFrame(afterimages, math.ceil(count/col), col, count)
        
    def btnRun_clicked(self):
        if debug:
            print("Running... ")
        
        # Output error message if no file is selected
        if filepath.get() == "":
            messagebox.showerror("Error! ", "Please select a file.")
            return
        
        # Output error message if an invalid threshold is input
        try:
            s = float(threshold.get())
            if (s >= 100.0) or (s <= 0.0):
                messagebox.showerror("Error! ", "Please enter a valid threshold " + 
                                     "between 0.0 and 100.0 exclusive of the " + 
                                     "boundary values.")
                return
        except:
            # if threshold is not of the correct data type 
            messagebox.showerror("Error! ", "Please enter a valid threshold " + 
                                 "between 0.0 and 100.0 exclusive of the " + 
                                 "boundary values.")
            return   
        
        # Output error message if duration is not entered or if invalid value is entered
        try:
            if duration.get() != "" and int(duration.get()) > 0:
                temp = float(duration.get())
            else:
                messagebox.showerror("Error! ", "Please enter a valid interval for "
                                     + "frame to be extracted.")
                return
        except:
            messagebox.showerror("Error! ", "Please enter a valid interval for "
                                     + "frame to be extracted.")
            return
        
        # Predict/Filter using thread
        t = SomeThread(filepath.get(), self) 
        t.start()
        print("***responsive***")
        
        
#       load model based on selected method 
        m = 0
        if method.get() == OPTIONS[0]:
            m = 1
        elif method.get() == OPTIONS[1]:
            m = 2
        elif method.get() == OPTIONS[2]:
            m = 3
        elif method.get() == OPTIONS[3]:
            m = 4
            
        print("Processing using Method ", m, "...")
        print("Path:", t.path)
        result = loadmodel.load(m, t.path) # t.path is a string
        
        
        # Output filtered frames
        self.filterFrame(result, len(result), t.path, t.dimension)
        
    
    

aframe=tkt.Frame(master)
wholeframe(aframe)
aframe.pack()  # packs a frame which given testme packs frame 1 in testme

#///////////////////////////////////////////////////////////////////////////////////
#                       LOOP TO ENSURE RESPONSIVENESS
#///////////////////////////////////////////////////////////////////////////////////    
master.mainloop() 