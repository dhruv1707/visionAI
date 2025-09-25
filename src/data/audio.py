import torch
import whisperx
import os
from botocore.exceptions import ClientError
import logging

class Audio():
    def __init__(self, path_to_data, batch_size=16):
        
        self.path_to_data = path_to_data
        self.model = whisperx.load_model("medium", device="cpu", compute_type="float32")
        self.batch_size = batch_size
    
    def transcribe_audio(self, s3, current_chunk_dir, bucket):
        print("Path to data: ", self.path_to_data)
        for dirpath, dirnames, filenames in os.walk(self.path_to_data):
            # print(f"Current directory: {dirpath}")
            # print(f"Subdirectories: {dirnames}")
            # print(f"Files: {filenames}")

            for file in filenames:
                if file.lower().endswith(".mp3"):
                    print("File inside audio: ", file)
                    audio = whisperx.load_audio(os.path.join(dirpath, file))
                    result = self.model.transcribe(audio, batch_size=self.batch_size)
                    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
                    result = whisperx.align(result["segments"], model_a, metadata, audio, "cpu", return_char_alignments=False)
                    print("Result inside audio: ", result)
                    transscript_file_name = "transcript.txt"
                    audio_path = os.path.join(current_chunk_dir, transscript_file_name)
                    with open(audio_path, 'w') as file:
                        # print(result['segments'])
                        for i in range(len(result['segments'])):
                            file.write(result['segments'][i]['text'])
                    print("Uploading audio transcript to S3")
                    try:
                        s3.upload_file(audio_path, bucket, audio_path)
                    except ClientError as e:
                        logging.error(e)
                    print(f"File '{transscript_file_name}' created and written to successfully.")