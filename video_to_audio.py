import subprocess
import os
from tqdm import tqdm  # Import tqdm for the progress bar

def extract_audio_from_video(video_path, audio_output_path):
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vn',
        '-acodec', 'copy',
        audio_output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio extracted successfully for {video_path}")
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode() if e.stderr else "No error message"
        print(f"Error extracting audio from {video_path}: {error_message}")
        return video_path, error_message
    return None, None

def process_directory(video_folder, audio_folder):
    errors = []
    video_files = [f for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]
    for video_file in tqdm(video_files, desc="Extracting audio"):  # Adding tqdm progress bar
        video_path = os.path.join(video_folder, video_file)
        audio_file_name = os.path.splitext(video_file)[0] + '.wav'
        audio_output_path = os.path.join(audio_folder, audio_file_name)
        result, error = extract_audio_from_video(video_path, audio_output_path)
        if result:
            errors.append((result, error))

    return errors

# Folder paths for videos and audios
video_folder = '/home/yz1031/yinghui/MHC/MultiHateClip_video_English'
audio_folder = '/home/yz1031/yinghui/MHC/multihateclip_english_audio'

errors = process_directory(video_folder, audio_folder)

if errors:
    print("Errors occurred with the following files:")
    for error in errors:
        print(f"{error[0]}: {error[1]}")
else:
    print("All videos processed successfully.")
