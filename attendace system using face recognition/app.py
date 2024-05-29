import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from PIL import Image

class FaceRecognitionAttendance:
    def __init__(self, master):
        self.master = master
        self.master.title("Attendance System Using Face Recognition")
        self.master.geometry("800x600")
        self.master.configure(background='Light Grey')

        # Add a heading
        self.heading = ttk.Label(master, text="Attendance System Using Face Recognition", font=("Arial", 16))
        self.heading.pack(padx=10, pady=10)
        self.heading.configure(foreground='Black', background='Light Grey', borderwidth=5, relief="groove", padding=(20,10,20,10))
        
        # Load images and encode faces
        self.path = 'student_images'
        self.images, self.classNames = self.load_images(self.path)
        self.encoded_face_train = self.find_encodings(self.images)

        # Set up the webcam feed
        self.cap = cv2.VideoCapture(0)
        self.video_frame = ttk.Label(master)
        self.video_frame.pack(padx=10, pady=10)
        self.video_frame.configure(borderwidth=5, relief="groove", background='Black')

        # Label for displaying attendance messages
        self.attendance_message = ttk.Label(master, text="", font=("Arial", 12))
        self.attendance_message.pack(padx=10, pady=10)

        # Start updating the video feed
        self.update_video()

        # Add a copyright notice
        self.copyright = ttk.Label(master, text=f"© salman {datetime.now().year}", font=("Arial", 10))
        self.copyright.pack(side="bottom", anchor="se")
        self.copyright.configure(background='Dark Grey')

    def load_images(self, path):
        images = []
        classNames = []
        for class_name in os.listdir(path):
            class_path = os.path.join(path, class_name)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    if os.path.isfile(img_path):
                        curImg = cv2.imread(img_path)
                        images.append(curImg)
                        classNames.append(class_name)
        return images, classNames

    def find_encodings(self, images):
        encode_list = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                encoded_face = encodings[0]
                encode_list.append(encoded_face)
            else:
                print("Faces not found in some images. Please Take clear images of faces with less noise at background.")
        return encode_list

    def mark_attendance(self, name):
        with open('Attendance.csv', 'r+') as f:
            my_data_list = f.readlines()
            name_list = [line.split(',')[0] for line in my_data_list]
            now = datetime.now()
            date = now.strftime('%d-%B-%Y')
            if name not in name_list or not any(name in line and date in line for line in my_data_list):
                time = now.strftime('%I:%M:%S:%p')
                f.writelines(f'{name}, {time}, {date}\n')
                self.attendance_message.config(text="Attendance Taken",background='Light Grey', foreground='Green', borderwidth=5, relief="groove", padding=(5,5,5,5))
            else:
                self.attendance_message.config(text="Attendance Taken for Today!!!",background='Light Grey', foreground='Green', borderwidth=5, relief="groove", padding=(5,5,5,5))

    def update_video(self):
        success, img = self.cap.read()
        if success:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            faces_in_frame = face_recognition.face_locations(imgS)
            encoded_faces = face_recognition.face_encodings(imgS, faces_in_frame)

            for encode_face, faceloc in zip(encoded_faces, faces_in_frame):
                matches = face_recognition.compare_faces(self.encoded_face_train, encode_face)
                face_dist = face_recognition.face_distance(self.encoded_face_train, encode_face)
                match_index = np.argmin(face_dist)

                if matches[match_index]:
                    name = self.classNames[match_index].upper().lower()
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    self.mark_attendance(name)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        # Schedule the next update (e.g., after 10 milliseconds)
        self.master.after(10, self.update_video)

    def close(self):
        self.cap.release()
        self.master.destroy()

def main():
    root = tk.Tk()
    app = FaceRecognitionAttendance(root)

    root.protocol("WM_DELETE_WINDOW", app.close)  # Ensure proper closure of the webcam
    root.mainloop()

if __name__ == "__main__":
    main()
