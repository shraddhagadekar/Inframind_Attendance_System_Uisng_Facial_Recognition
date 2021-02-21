import tkinter as tk
import cv2
from tkinter import messagebox
import csv
import numpy as np
import pandas as pd
import os
import  datetime,time
from PIL import Image

window = tk.Tk()
window.geometry('500x650+400+50')
window.title("Attendance System")

message =tk.Label(window, text="Attendance Tracking System" ,bg="white" ,fg="black" ,width=21,height=1,font=('times',30,'italic bold'))
message.place(x=10,y=10)

message = tk.Label(window, text=" " ,bg="white"  ,fg="black"  ,width=60  ,height=13,font=('times', 10,'italic bold'))
message.place(x=30, y=130)

message = tk.Label(window, text="New Users" ,bg="white"  ,fg="black"  ,width=25  ,height=2,font=('times', 20, 'italic bold'))
message.place(x=40, y=130)

lbl = tk.Label(window, text="Enter ID:",width=13  ,height=1  ,fg="black"  ,bg="white" ,font=('times', 13, ' bold '))
lbl.place(x=45, y=195)

txt = tk.Entry(window,width=20  ,bg="white" ,fg="black",font=('times', 15, ' bold '))
txt.place(x=183, y=195)
txt.focus()

lbl2 = tk.Label(window, text="Enter Name:",width=13  ,fg="black"  ,bg="white"  ,height=2 ,font=('times', 13, ' bold '))
lbl2.place(x=45, y=226)

txt2 = tk.Entry(window,width=20  ,bg="white"  ,fg="black",font=('times', 15, ' bold '))
txt2.place(x=183, y=235)

message = tk.Label(window, text=" " ,bg="black"  ,fg="black"  ,width=60  ,height=7,font=('times', 10, 'italic bold'))
message.place(x=30, y=365)

lbl3 = tk.Label(window, text="Attendance : ",width=13  ,fg="white"  ,bg="black"  ,height=2 ,font=('times', 13,'bold '))
lbl3.place(x=30, y=395)

message2 = tk.Label(window, text="" ,fg="white"   ,bg="black",activeforeground = "green",width=28  ,height=5  ,font=('times', 13, ' bold '))
message2.place(x=163, y=365)

def TakeImages():
    id=txt.get()
    name=txt2.get()
    if not id:
        messagebox.showerror('Error','Provide Proper INT Value',icon='error')
    elif not name:
        messagebox.showerror('Error','Provide Proper Value',icon='error')

    elif(id.isdigit() and name.isalpha()):  #type of validation while taking name and id
        cam=cv2.VideoCapture(0)
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        num=0
        while (True):
            check,frame=cam.read()
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray_img, 1.3, 5)
            for x, y, w, h, in faces:
                cv2.rectangle(frame,(x, y), (x + w, y + h), (0, 255, 0), 4)
                num += 1
                # saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainedImages\ " + name + "." + id + '.' + str(num) + ".jpg", gray_img[y:y + h, x:x + w])
                # display the frame
                cv2.imshow('Facial Recognition', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif num > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        row = [id, name]
        f = open("StudentDetails.csv", 'a+', newline='')
        w = csv.writer(f)
        w.writerow(row)
        f.close()
        messagebox.showinfo('Done','Your Face Images are added !!',icon='info')

    else:
        messagebox.showinfo('Value','Provide INT value for id and Str value for Name',icon='info')

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Idd = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Idd)
    return faces, Ids

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, key = getImagesAndLabels("TrainedImages")
    recognizer.train(faces, np.array(key))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    clear()
    tk.messagebox.showinfo('Completed','Your model has been trained successfully!!')

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Reading the trained model
    recognizer.read("TrainingImageLabel\Trainner.yml")
    face_Cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # getting the name from "userdetails.csv"
    df = pd.read_csv("StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_Cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 60):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id) + "-" + aa
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
            else:
                Id = 'Unknown'
                tt = str(Id)

            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('Press q to mark attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance\Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    res = attendance
    message2.configure(text=res)


def quit_window():
    MsgBox = tk.messagebox.askquestion(title='Quit',message='Are you sure you want to exit the application?',icon='warning')
    if MsgBox == 'yes':
        tk.messagebox.showinfo("Greetings", "Thank You")
        window.destroy()

def clear():
    txt.delete(0, 'end')
    txt2.delete(0, 'end')

takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="white"  ,bg="grey"  ,width=10  ,height=1, activebackground = "aqua" ,font=('times', 15, ' bold '))
takeImg.place(x=100, y=280)

trainImg = tk.Button(window, text="Trained", command=TrainImages  ,fg="white"  ,bg="blue"  ,width=10  ,height=1, activebackground = "gold" ,font=('times', 15, ' bold '))
trainImg.place(x=260, y=280)

trackImg = tk.Button(window, text="Mark Attendance", command=TrackImages  ,fg="white"  ,bg="green"  ,width=14  ,height=1, activebackground = "lime" ,font=('times', 15, ' bold '))
trackImg.place(x=20, y=70)

quitWindow = tk.Button(window, text="QUIT", command=quit_window  ,fg="white"  ,bg="red"  ,width=10  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
quitWindow.place(x=80, y=500)

quitWindow = tk.Button(window, text="CLEAR", command=clear  ,fg="white"  ,bg="red"  ,width=10  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
quitWindow.place(x=300, y=500)

window.mainloop()