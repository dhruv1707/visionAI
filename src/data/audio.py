import torch
import whisperx
import os

class Audio():
    def __init__(self, path_to_data, batch_size=16):
        
        self.path_to_data = path_to_data
        self.model = whisperx.load_model("medium", device="cpu", compute_type="float32")
        self.batch_size = batch_size
    
    def transcribe_audio(self):
        print("Path to data: ", self.path_to_data)
        for dirpath, dirnames, filenames in os.walk(self.path_to_data):
            # print(f"Current directory: {dirpath}")
            # print(f"Subdirectories: {dirnames}")
            # print(f"Files: {filenames}")

            for file in filenames:
                if file.lower().endswith(".mp3"):
                    audio = whisperx.load_audio(os.path.join(dirpath, file))
                    result = self.model.transcribe(audio, batch_size=self.batch_size)
                    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
                    result = whisperx.align(result["segments"], model_a, metadata, audio, "cpu", return_char_alignments=False)
                    file_name = f"{self.path_to_data}/transcript.txt"
                    with open(file_name, 'w') as file:
                        file.write(result['segments'][0]['text'])
                    print(f"File '{file_name}' created and written to successfully.")
            # if os.path.isfile(file_path) and file_path.lower().endswith(".mp3"):
            #     print(file_path)

# if __name__=="__main__":
#     path_to_data = "src/output/"
#     audio_transcription = Audio(path_to_data=path_to_data, batch_size=16)
#     audio_transcription.transcribe_audio()