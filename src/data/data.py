import torch
import os
from pathlib import Path
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy
from moviepy import VideoFileClip
import logging
from audio_extract import extract_audio
import datetime
import subprocess, shlex
import audio

logger = logging.getLogger(__name__)

class Data():
    def __init__(self, path_to_vid):
        # super.__init__(Data)
        self.path_to_vid = path_to_vid
    
    
    # def chunk_video(self, path_to_vid, output_dir):

    #     def extract_subclip_reencode(input_path, t1, t2, output_path):
    #         clip = VideoFileClip(input_path).subclipped(t1, t2)
    #         clip.write_videofile(
    #         output_path,
    #         codec="libx264",     # enforce H.264
    #         audio=False,  
    #         remove_temp=True,
    #         threads=4,           # parallelize
    #         ffmpeg_params=["-ss", str(t1), "-to", str(t2)],  # precise seeking
    #         )
    #         clip.close()

    #     def split_video(video_path, chunk_duration, output_dir):
    #         if not os.path.exists(output_dir):
    #             os.makedirs(output_dir)
            
    #         clip_duration = VideoFileClip(video_path).duration
    #         print(clip_duration, path_to_vid)

    #         start_time = 0
    #         chunk_number = 1

    #         while start_time < clip_duration:
    #             end_time = min(start_time + chunk_duration, clip_duration)
    #             output_path = os.path.join(output_dir, f"chunk_{chunk_number}.mp4")
    #             extract_subclip_reencode(video_path, start_time, end_time, output_path)
    #             ffmpeg_extract_subclip(video_path, start_time, end_time, output_path)
    #             start_time = end_time
    #             chunk_number += 1
    #             print(f"Extracted chunk {chunk_number} into {output_path}")
            
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
        
    #     print(f"Traversing: {self.path_to_vid}")
    #     print("Is dir?", os.path.isdir(self.path_to_vid))
        
    #     patterns = [".mp4", ".mov"]
        
    #     for filepath in Path(self.path_to_vid).rglob("*"):
    #         if filepath.suffix.lower() in patterns:
    #             save_dir = os.path.join(output_dir, filepath.name)
    #             try:
    #                 split_video(filepath, chunk_duration=30, output_dir=save_dir)
    #             except Exception as e:
    #                 logger.exception(f"Failed to process {filepath}: {e}")
    #                 continue
    #     return
    
    def seconds_to_timecode(self, seconds):
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def has_audio(self, path):
        cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0", path
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return bool(proc.stdout.strip())
    
    def extract_audio_for_chunk(self, video_path, chunk_start, chunk_duration, chunk_dir):
        audio_file = os.path.join(chunk_dir , f"audio_{chunk_start:.0f}.mp3")
        if not video_path.suffix.lower() in {".mp4", ".mov"}:
            return
        if not self.has_audio(str(video_path)):
            logger.warning(f"No audio track in {video_path}, skipping audio extraction.")
            return
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_start_time = self.seconds_to_timecode(chunk_start)
        try:
            extract_audio(
            input_path=str(video_path),
            output_path=str(audio_file),
            start_time=chunk_start_time,
            duration=min(chunk_duration, VideoFileClip(str(video_path)).duration - chunk_start))
        except Exception as e:
            logger.exception(f"Audio extraction failed for {video_path} @ {chunk_start}s: {e}")
            
                
    def process_video(self, video_path, output_dir):
        
        def extract_frames_and_audio(video_path, output_dir, n_seconds, chunk_duration):
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error reading the video file: {self}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            interval = int(fps * n_seconds)
            if video_path.suffix.lower() in [".mp4", ".mov"]:
                clip = VideoFileClip(video_path)
                duration_seconds = clip.duration
                print(f"Duration of clip: {duration_seconds}")

            frame_count = 0
            frames_in_chunk  = 0
            chunk_count = 1
            chunk_start = 0
            current_chunk_dir = ""
            tot_frames = 0

            current_chunk_dir = os.path.join(output_dir, f"chunk_{chunk_count:03d}")
            os.makedirs(current_chunk_dir, exist_ok=True)
            self.extract_audio_for_chunk(video_path, chunk_start, chunk_duration, current_chunk_dir)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % interval == 0:
                    if frames_in_chunk >= 5:
                        chunk_count += 1
                        chunk_start += chunk_duration
                        frames_in_chunk = 0
                        current_chunk_dir = os.path.join(output_dir, f"chunk_{chunk_count:03d}")
                        os.makedirs(current_chunk_dir, exist_ok=True)
                        self.extract_audio_for_chunk(video_path, chunk_start, chunk_duration, current_chunk_dir)
                        audio_transcription = audio.Audio(current_chunk_dir)
                        audio_transcription.transcribe_audio()
                        print(f"Created chunk dir {current_chunk_dir}")

                    frame_path = os.path.join(current_chunk_dir, f"frame_{frames_in_chunk:03d}.png")
                    print(f"Storing frame #{frames_in_chunk} in {frame_path}")
                    cv2.imwrite(str(frame_path), frame)
                    frames_in_chunk += 1
                    tot_frames += 1
                
                frame_count += 1
                        # chunk_name = f"chunk_{chunk_count}"
                        # audio_file_name = f"audio_{chunk_count}.mp3"
                        # current_chunk_dir = os.path.join(output_dir, chunk_name)
                        # os.makedirs(current_chunk_dir, exist_ok=True)
                        # print(f"Created chunk dir {output_dir}")
                        # if video_path.suffix.lower() in [".mp4", ".mov"]:
                        #     if not self.has_audio(str(video_path)):
                        #         logger.warning(f"No audio track in {video_path}, skipping audio extraction.")
                        #     else:
                        #         audio_output_path = os.path.join(current_chunk_dir, audio_file_name)
                                
                        #         if not os.path.exists(audio_output_path):
                        #             chunk_start_time = self.seconds_to_timecode(chunk_start)
                        #             remaining = duration_seconds - chunk_start
                        #             print(f"Extracting audio from {chunk_start_time} to {min(chunk_start+chunk_duration, duration_seconds)} of {video_path}. Output path: {audio_output_path}")
                        #             dur = min(remaining, chunk_duration)
                        #             os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
                        #             try:
                        #                 if dur == remaining:
                        #                     print("Dur = remaining")
                        #                     extract_audio(input_path=video_path, output_path=audio_output_path, start_time=chunk_start_time)
                        #                 else:
                        #                     extract_audio(input_path=video_path, output_path=audio_output_path, start_time=chunk_start_time, duration=dur)
                        #             except Exception as e:
                        #                 logger.exception(f"Failed to process {filepath}: {e}")
                        #                 break

                        #             chunk_start = min(chunk_duration + chunk_start, duration_seconds)
                        # chunk_count += 1
                        # saved_count += 1
            cap.release()
            print(f"Extracted {tot_frames} frames to {output_dir}")
            return

            
        print(f"Traversing: {video_path}")
        print("Is dir?", os.path.isdir(video_path))
        patterns = [".mp4", ".mov", ".png", ".jpg", ".jpeg"]
        for filepath in Path(video_path).rglob("*"):
            if filepath.suffix.lower() in patterns:
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                save_dir = os.path.join(output_dir, base_name)
                print("Output dir: ", output_dir)
                extract_frames_and_audio(video_path=filepath, output_dir=save_dir, n_seconds=6, chunk_duration=30)
                
                

if __name__=="__main__":
    path_to_vid = "/Users/dhruvmehrottra007/Desktop/Beerbiceps - Assignments"
    output_dir = "src/output/"
    output_dir_chunk = "src/chunks"
    data = Data(path_to_vid)
    # data.chunk_video(path_to_vid, output_dir=output_dir_chunk)
    data.process_video(path_to_vid, output_dir=output_dir)