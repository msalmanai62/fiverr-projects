from tkinter import ttk, filedialog, messagebox
import tkinter as tk
import pandas as pd
import customtkinter
from PIL import ImageTk, Image
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import threading
from train import trainn, preprocesss

class LoginWindow(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("Dark")
        customtkinter.set_default_color_theme("blue")

        self.geometry("800x450")
        self.title("Network Intrusion Detection System")

        self.bg = ImageTk.PhotoImage(Image.open("background.png"))
        self.background = customtkinter.CTkLabel(self, image=self.bg, text=" ")
        self.background.pack()

        self.username = customtkinter.CTkEntry(self, width=220, placeholder_text="Enter your username", font=("Arial", 12), text_color="#FFFFFF", fg_color="#000000")
        self.username.place(x=290, y=130)

        self.password = customtkinter.CTkEntry(self, width=220, placeholder_text="Enter your password", font=("Arial", 12), text_color="#FFFFFF", show="*", fg_color="#000000")
        self.password.place(x=290, y=165)

        self.login_button = customtkinter.CTkButton(self, text="Log In", font=("Arial Bold", 12), text_color="#FFFFFF", width=70, corner_radius=6, command=self.button_function)
        self.login_button.place(x=365, y= 220)

    def button_function(self):
        if self.username.get() == "admin" and self.password.get() == "admin":
            self.destroy()  # destroy current window and creating new one
            home = HomeWindow()
            home.mainloop()

class HomeWindow(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.train_data = None
        self.test_data = None
        self.data = None
        self.model = None

        customtkinter.set_appearance_mode("Dark")
        customtkinter.set_default_color_theme("blue")

        self.geometry("800x450")
        self.title("Welcome")

        self.bg = ImageTk.PhotoImage(Image.open("background.png"))
        self.background = customtkinter.CTkLabel(self, image=self.bg, text=" ")
        self.background.pack()

        self.data_frame = customtkinter.CTkFrame(master=self, width=235, height=130, fg_color="#000000")
        self.data_frame.place(relx=0.10, rely=0.3)

        self.model_frame = customtkinter.CTkFrame(master=self, width=160, height=130, fg_color="#000000")
        self.model_frame.place(relx=0.10, rely=0.6)

        self.training_frame = customtkinter.CTkFrame(master=self, width=400, height=130, fg_color="#000000",)
        self.training_frame.place(relx=0.4, rely=0.3)

        self.progress_feature = customtkinter.DoubleVar()
     
        label_data_frame = customtkinter.CTkLabel(self.data_frame, text="Data Input", font=("Arial Bold", 14))
        label_data_frame.place(x=80, y=10)

        load_train_button = customtkinter.CTkButton(self.data_frame, font=("Arial", 11), fg_color="#0C2155", text="Train Data", command=self.load_train_data)
        load_train_button.place(x=47, y=52)

        load_test_button = customtkinter.CTkButton(self.data_frame, font=("Arial", 11), fg_color="#0C2155", text="Test Data", command=self.load_test_data)
        load_test_button.place(x=47, y=85)

        models = ["Random Forest", "KNN", "Logistic Regression", "Decision Tree"]
        customtkinter.CTkLabel(self.model_frame, text="Select Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.model_selection = customtkinter.CTkComboBox(self.model_frame, values=models)
        self.model_selection.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        label_training_frame = customtkinter.CTkLabel(self.training_frame, text="Model Training & Testing", font=("Arial Bold", 14))
        label_training_frame.place(x=110, y=10)

        label_training_f_frame = customtkinter.CTkLabel(self.training_frame, text="Preprocessing and Training: ", font=("Arial", 11))
        label_training_f_frame.place(x=20, y=50)

        self.feature_progressbar = ttk.Progressbar(self.training_frame, orient="horizontal", length=300, mode="determinate", variable=self.progress_feature)
        self.feature_progressbar.place(x=250, y=85)

        self.train_button = customtkinter.CTkButton(self.training_frame, text="Train", font=("Arial Bold", 12), text_color="#FFFFFF", fg_color="#0C2155", width=70, corner_radius=6, command=self.train_test_model)
        self.train_button.place(x=200, y=90)
        
        self.preprocess_button = customtkinter.CTkButton(self.training_frame, text="Preprocess", font=("Arial Bold", 12), text_color="#FFFFFF", fg_color="#0C2155", width=90, corner_radius=6, command=self.preprocess_data)
        self.preprocess_button.place(x=310, y=90)

        self.output_text = customtkinter.CTkTextbox(self, height=5, width=120, corner_radius=6, fg_color="#000000", font=("Arial", 12))
        self.output_text.place(relx=0.4, rely=0.6)

        self.accuracy_text = customtkinter.CTkTextbox(self, height=5, width=150, corner_radius=6, fg_color="#000000", font=("Arial", 12))
        self.accuracy_text.place(relx=0.555, rely=0.6)

        self.time_text = customtkinter.CTkTextbox(self, height=5, width=150, corner_radius=6, fg_color="#000000", font=("Arial", 12))
        self.time_text.place(relx=0.725, rely=0.6)


    def load_train_data(self):
        file_path = filedialog.askopenfilename(title="Select Train Data", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.train_data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Train data loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading train data: {str(e)}")

    def load_test_data(self):
        file_path = filedialog.askopenfilename(title="Select Test Data", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.test_data = pd.read_csv(file_path)
                # self.test_data = self.test_data.drop('class', axis=1)
                messagebox.showinfo("Success", "Test data loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading test data: {str(e)}")

    def preprocess_data(self):
        if self.train_data is None or self.test_data is None:
            messagebox.showerror("Error", "Please load both train and test data.")
            return
        preprocessing_thread = threading.Thread(target=self.background_preprocessing_process)
        preprocessing_thread.start()

    def train_test_model(self):
        # Check if data is loaded
        if self.train_data is None or self.test_data is None:
            messagebox.showerror("Error", "Please load both train and test data.")
            return
        elif self.data is None:
            messagebox.showerror("Error", "Please preprocess the data before training.")
            return
        # Start training in a new thread
        training_thread = threading.Thread(target=self.background_training_process)
        training_thread.start()

    def background_preprocessing_process(self):
        try:
            if self.train_data is None or self.test_data is None:
                raise ValueError("Please load both train and test data.")
            self.output_text.delete(1.0, 'end')
            self.output_text.insert(customtkinter.END, 'Preprocessing....')
            self.data = preprocesss(train_data=self.train_data)

            self.progress_feature.set(0)
            # Simulating  progress
            for i in range(101):
                self.progress_feature.set(i)
                self.update()
                time.sleep(0.005)

            self.output_text.delete(1.0, 'end')
            self.output_text.insert(customtkinter.END, 'Preprocessing Success')
        except Exception as e:
            self.show_error_on_gui(str(e))

    def background_training_process(self):
        try:
            if self.data is None:
                raise ValueError("Please preprocess the data before training.")
            selected_model = self.model_selection.get()
            if selected_model == "Random Forest":
                self.model = RandomForestClassifier()
            elif selected_model == "KNN":
                self.model = KNeighborsClassifier()
            elif selected_model == "Logistic Regression":
                self.model = LogisticRegression()
            elif selected_model == "Decision Tree":
                self.model = DecisionTreeClassifier()

            self.output_text.delete(1.0, 'end')
            self.output_text.insert(customtkinter.END, 'Training...')

            self.progress_feature.set(0)
            # Simulating progress
            for i in range(101):
                self.progress_feature.set(i)
                self.update()
                time.sleep(0.005)
    
            accuracy, training_time = trainn(modell=self.model, data=self.data)
            self.output_text.delete(1.0, 'end')
            self.output_text.insert(customtkinter.END, 'Training Success')
            accuracy, training_time = round(accuracy*100, 2), round(training_time,3)
            self.update_gui_with_results(training_time, accuracy)
        except Exception as e:
            self.show_error_on_gui(str(e))

    def update_gui_with_results(self, training_time, accuracy):
        if self.winfo_exists():  # Check if the window is still open
            # Display evaluation results
            # self.output_text.delete(1.0, 'end')
            # self.output_text.insert(customtkinter.END, 'Training Success')
            self.accuracy_text.delete(1.0, 'end')
            self.accuracy_text.insert(tk.END, f"Accuracy: {accuracy}\n")
            self.time_text.delete(1.0, 'end')
            self.time_text.insert(tk.END, f"Training Time: {training_time} S\n")
            print(accuracy, training_time)

    def show_error_on_gui(self, error_message):
        if self.winfo_exists():  # Check if the window is still open
            messagebox.showerror("Training Error", error_message)

if __name__ == '__main__':
    login_window = LoginWindow()
    login_window.mainloop()
