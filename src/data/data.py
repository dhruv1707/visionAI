import torch
import os
from pathlib import Path
import cv2


class Data():
    def __init__(self, path_to_vid):
        # super.__init__(Data)
        self.path_to_vid = path_to_vid
    
    def process_video(self):
        
        def extract_frames(self, output_dir, frame_interval=1):
            
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            cap = cv2.VideoCapture()
            
        print(f"Traversing: {self.path_to_vid}")
        print("Is dir?", os.path.isdir(self.path_to_vid))
        patterns = [".mp4", ".mov", ".png", ".jpg", ".jpeg"]
        for filepath in Path(self.path_to_vid).rglob("*"):
            if filepath.suffix.lower() in patterns:
                filepath.extract_frames(output_dir=output_dir, frame_interval=10)
            

if __name__=="__main__":
    path_to_vid = "/Users/dhruvmehrottra007/Desktop/Beerbiceps - Assignments"
    data = Data(path_to_vid)
    data.process_video()