import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import json
import os
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Conv3D, MaxPooling3D, Dropout
import sys
print(sys.executable)
print(sys.path)

import imageio
import moviepy

class FightingStyleAI:
    def __init__(self):
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(30, 33*4)),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam', 
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        
    def train(self, video_path, labels):
        # Training implementation
        return self.model.fit(np.random.random((100, 30, 132)), labels, 
                            epochs=5, batch_size=32)
    
    def predict_defense(self, video_path):
        # Prediction implementation
        return self.model.predict(np.random.random((1, 30, 132)))

class FightingAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fighting Style AI Analyzer")
        self.root.geometry("1200x800")
        
        self.fighting_ai = FightingStyleAI()
        self.setup_gui()
        
    def setup_gui(self):
        # Left panel for controls
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        # Video input section
        ttk.Button(control_frame, text="Load Training Video", 
                  command=self.load_video).grid(row=0, column=0, pady=5)
        ttk.Button(control_frame, text="Start Training", 
                  command=self.start_training).grid(row=1, column=0, pady=5)
        ttk.Button(control_frame, text="Load Test Video", 
                  command=self.load_test_video).grid(row=2, column=0, pady=5)
        ttk.Button(control_frame, text="Analyze Defense", 
                  command=self.analyze_defense).grid(row=3, column=0, pady=5)
        
        # Save/Load model buttons
        ttk.Button(control_frame, text="Save Model", 
                  command=self.save_model).grid(row=4, column=0, pady=5)
        ttk.Button(control_frame, text="Load Model", 
                  command=self.load_model).grid(row=5, column=0, pady=5)
        
        # Right panel for video display
        video_frame = ttk.LabelFrame(self.root, text="Video Display", padding="10")
        video_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew")
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.grid(row=0, column=0)
        
        # Bottom panel for output logs
        log_frame = ttk.LabelFrame(self.root, text="Processing Logs", padding="10")
        log_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")
        
        self.log_text = tk.Text(log_frame, height=10)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
    def load_test_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.test_video = file_path
            self.log_message(f"Loaded test video: {file_path}")
            self.display_video_frame()
        
        
    def load_video(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if file_path:
            self.current_video = file_path
            self.log_message(f"Loaded video: {file_path}")
            self.display_video_frame()
            
    def display_video_frame(self):
        cap = cv2.VideoCapture(self.current_video)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        cap.release()
        
    def start_training(self):
        if hasattr(self, 'current_video'):
            self.log_message("Starting training process...")
            dummy_labels = np.random.randint(0, 10, size=(100, 10))
            history = self.fighting_ai.train(self.current_video, dummy_labels)
            self.log_message("Training completed!")
            
    def analyze_defense(self):
        if hasattr(self, 'current_video'):
            self.log_message("Analyzing defense moves...")
            predictions = self.fighting_ai.predict_defense(self.current_video)
            self.log_message(f"Predicted defense moves: {predictions}")
            
    def save_model(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("H5 files", "*.h5")]
        )
        if file_path:
            save_model(self.fighting_ai.model, file_path)
            self.log_message(f"Model saved to: {file_path}")
            
    def load_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("H5 files", "*.h5")]
        )
        if file_path:
            self.fighting_ai.model = load_model(file_path)
            self.log_message(f"Model loaded from: {file_path}")
            
    def save_pose(self, landmarks, pose_name):
        poses_dir = "saved_poses"
        if not os.path.exists(poses_dir):
            os.makedirs(poses_dir)
            
        pose_data = {
            "name": pose_name,
            "landmarks": landmarks
        }
        
        file_path = os.path.join(poses_dir, f"{pose_name}.json")
        with open(file_path, 'w') as f:
            json.dump(pose_data, f)
            
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)

def main():
    root = tk.Tk()
    app = FightingAIGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
