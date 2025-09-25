import torch
import os
from pathlib import Path
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy
from moviepy import VideoFileClip
import logging
from audio_extract import extract_audio
from botocore.exceptions import ClientError
import datetime
import subprocess, shlex
import audio
import json
from concurrent.futures import ThreadPoolExecutor
import imageio.v2 as iio

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
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
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
        EPS = 0.02
        if not video_path.suffix.lower() in {".mp4", ".mov"}:
            return
        if not self.has_audio(str(video_path)):
            logger.warning(f"No audio track in {video_path}, skipping audio extraction.")
            return
        os.makedirs(chunk_dir, exist_ok=True)
        # chunk_start_time = self.seconds_to_timecode(chunk_start)
        a_total = VideoFileClip(str(video_path)).audio.duration
        remaining_audio = a_total - float(chunk_start)
        eff = min(float(chunk_duration), float(remaining_audio)) - EPS
        if eff <= 0:
            logger.info(f"Effective duration <= 0 for {video_path} at {chunk_start}s; skipping.")
            return
        try:
            extract_audio(
            input_path=str(video_path),
            output_path=str(audio_file),
            start_time=self.seconds_to_timecode(chunk_start),
            duration=float(eff))
        except Exception as e:
            logger.exception(f"Audio extraction failed for {video_path} @ {chunk_start}s: {e}")
            
                
    def process_video(self, video_path, output_dir, key, s3):
        
        def extract_frames_and_audio(video_path, output_dir, n_seconds, chunk_duration, bucket):
            
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
                print(f"Duration of clip: {duration_seconds}")\
                
            with VideoFileClip(str(video_path)) as clip:

                frame_count = 0
                frames_in_chunk  = 0
                chunk_count = 1
                chunk_start = 0
                current_chunk_dir = ""
                tot_frames = 0
                t = 0.0
                frames_per_chunk = max(1, int(chunk_duration // n_seconds))

                current_chunk_dir = os.path.join(output_dir, f"chunk_{chunk_count:03d}")
                os.makedirs(current_chunk_dir, exist_ok=True)

                self.extract_audio_for_chunk(video_path, chunk_start, chunk_duration, current_chunk_dir)

                if duration_seconds < 30:
                    audio_transcription = audio.Audio(current_chunk_dir)
                    audio_transcription.transcribe_audio(s3, current_chunk_dir, bucket)
                    print(f"Created chunk dir {current_chunk_dir}")

                while t < duration_seconds + 1e-6:
                    if frames_in_chunk >= frames_per_chunk:
                        chunk_count += 1
                        chunk_start += chunk_duration
                        frames_in_chunk = 0
                        current_chunk_dir = os.path.join(output_dir, f"chunk_{chunk_count:03d}")
                        os.makedirs(current_chunk_dir, exist_ok=True)
                        self.extract_audio_for_chunk(video_path, chunk_start, chunk_duration, current_chunk_dir)
                        meta = {
                                "video_id":   os.path.splitext(os.path.basename(video_path))[0],
                                "chunk_start": chunk_start,
                                "chunk_end":   min(chunk_start+chunk_duration, duration_seconds)
                            }
                        metadata_path = os.path.join(current_chunk_dir, "metadata.json")
                        with open(metadata_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2)
                        audio_transcription = audio.Audio(current_chunk_dir)
                        audio_transcription.transcribe_audio(s3, current_chunk_dir, bucket)
                        print(f"Created chunk dir {current_chunk_dir}")

                    frame_path = os.path.join(current_chunk_dir, f"frame_{frames_in_chunk:03d}.jpg")
                    frame = clip.get_frame(t)
                    print(f"Storing frame #{frames_in_chunk} in {frame_path}")
                    try:
                        iio.imwrite(frame_path, frame, quality=90)
                    except TypeError:
                    # backend fallback if 'quality' not supported
                        iio.imwrite(frame_path, frame)

                    try:
                        s3.upload_file(frame_path, bucket, frame_path)
                    except ClientError as e:
                        logging.error(e)
                        
                    frames_in_chunk += 1
                    tot_frames += 1
                    frame_count += 1
                    t += float(n_seconds)

            print(f"Extracted {tot_frames} frames to {output_dir}")
            return

            
        print(f"Traversing: {video_path}")
        print("Is dir?", os.path.isdir(video_path))
        patterns = [".mp4", ".mov", ".png", ".jpg", ".jpeg"]
        for filepath in Path(video_path).rglob("*"):
            print("Filepath: ", filepath)
            if filepath.suffix.lower() in patterns:
                base_name = key
                save_dir = os.path.join(output_dir, base_name)
                print("Output dir: ", output_dir)
                extract_frames_and_audio(video_path=filepath, output_dir=save_dir, n_seconds=6, chunk_duration=30, bucket=output_dir)
                
                

if __name__=="__main__":
    path_to_vid = "/Users/dhruvmehrottra007/Desktop/Beerbiceps - Assignments/LA9"
    output_dir = "/Users/dhruvmehrottra007/Desktop/VisionAI/src/output"
    output_dir_chunk = "src/chunks"
    data = Data(path_to_vid)
    # data.chunk_video(path_to_vid, output_dir=output_dir_chunk)
    data.process_video(path_to_vid, output_dir=output_dir)