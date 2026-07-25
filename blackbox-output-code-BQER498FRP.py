"""
Facial Recognition Safety & Attendance System
Real-time face detection, recognition, unknown person alerts, and auto-attendance
"""

import face_recognition
import cv2
import numpy as np
import pandas as pd
import os
import csv
from datetime import datetime
from playsound import playsound
import tkinter as tk
from tkinter import messagebox
import threading

class FacialRecognitionSystem:
    def __init__(self):
        # Known students data
        self.known_students = {
            "student1.jpg": "John Doe",
            "student2.jpg": "Jane Smith", 
            "student3.jpg": "Mike Johnson",
            "student4.jpg": "Sarah Wilson"
        }
        
        # Load known face encodings
        self.known_encodings = []
        self.student_names = []
        self.load_known_faces()
        
        # Attendance DataFrame
        self.attendance_file = "attendance.csv"
        self.init_attendance_csv()
        
        # System state
        self.camera = None
        self.running = False
        
        print("🚀 Facial Recognition Safety & Attendance System Initialized!")
        print(f"📊 Known Students: {len(self.student_names)}")
    
    def load_known_faces(self):
        """Load and encode known student faces from images folder"""
        print("🔄 Loading known faces from images/ folder...")
        
        images_folder = "images"
        if not os.path.exists(images_folder):
            print("❌ Create 'images/' folder and add student1.jpg to student4.jpg")
            return
        
        for filename, name in self.known_students.items():
            image_path = os.path.join(images_folder, filename)
            
            if os.path.exists(image_path):
                # Load image
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)
                
                if encoding:
                    self.known_encodings.append(encoding[0])
                    self.student_names.append(name)
                    print(f"✅ Loaded: {name}")
                else:
                    print(f"⚠️ No face found in {filename}")
            else:
                print(f"❌ Missing image: {image_path}")
    
    def init_attendance_csv(self):
        """Initialize attendance CSV file"""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Name', 'Date', 'Time', 'Status'])
            print("📄 attendance.csv created")
    
    def mark_attendance(self, name):
        """Mark attendance for recognized student"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Check if already marked today
        df = pd.read_csv(self.attendance_file)
        today_records = df[(df['Name'] == name) & (df['Date'] == date_str)]
        
        if today_records.empty:
            new_record = pd.DataFrame({
                'Name': [name],
                'Date': [date_str],
                'Time': [time_str],
                'Status': ['Present']
            })
            
            new_record.to_csv(self.attendance_file, mode='a', header=False, index=False)
            print(f"✅ Attendance marked: {name} - {date_str} {time_str}")
            return True
        else:
            print(f"ℹ️  {name} already marked attendance today")
            return False
    
    def alert_unknown_person(self):
        """Show alert for unknown person"""
        print("🚨 UNKNOWN PERSON DETECTED – ALERT SECURITY!")
        
        # Play alert sound (optional)
        try:
            playsound('alert.wav')  # Add alert.wav file or comment this line
        except:
            pass
        
        # Show popup alert
        root = tk.Tk()
        root.withdraw()  # Hide main window
        messagebox.showwarning("SECURITY ALERT", "🚨 UNKNOWN PERSON DETECTED!\nAlert Security Immediately!")
        root.destroy()
    
    def start_camera(self):
        """Start real-time face recognition"""
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            print("❌ Cannot access camera")
            return
        
        self.running = True
        print("📹 Camera started. Press 'q' to quit, 's' to save screenshot")
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find faces in current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Display frame with annotations
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Check if face matches known students
                matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
                name = "Unknown"
                
                # Find best match
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                    name = self.student_names[best_match_index]
                    self.mark_attendance(name)
                    color = (0, 255, 0)  # Green for known
                else:
                    # Unknown person alert
                    self.alert_unknown_person()
                    color = (0, 0, 255)  # Red for unknown
                
                # Draw rectangle and label
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            # Instructions on screen
            cv2.putText(frame, "Press 'q' to quit, 's' to screenshot", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Facial Recognition Safety & Attendance System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
        
        self.stop_system()
    
    def stop_system(self):
        """Clean shutdown"""
        self.running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("👋 System stopped")

def main():
    """Main function"""
    try:
        system = FacialRecognitionSystem()
        system.start_camera()
    except KeyboardInterrupt:
        print("\n👋 System interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("📊 Check attendance.csv for records")

if __name__ == "__main__":
    main()