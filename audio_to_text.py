import os
import whisper
import torch
from tqdm import tqdm

def transcribe_audio(file_path):
    """
    使用OpenAI Whisper模型将音频文件转换为文字。
    :param file_path: 音频文件的路径
    :return: 转录的文本
    """
    model = whisper.load_model("base")  # 根据需要，可以选择不同的模型大小
    if torch.cuda.is_available():  # 检查是否有可用的GPU
        model = model.to("cuda")  # 将模型转移到GPU
    result = model.transcribe(file_path, language="zh")
    return result['text']

def process_audio_files(input_folder, output_folder):
    """
    处理指定文件夹中的所有音频文件，并将转录的文本保存为文本文件。
    :param input_folder: 包含音频文件的文件夹路径
    :param output_folder: 输出文本文件的文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.mp3') or f.endswith('.wav')]
    for filename in tqdm(audio_files, desc="Transcribing audio files"):
        file_path = os.path.join(input_folder, filename)
        text = transcribe_audio(file_path)
        text_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
        with open(text_file_path, 'w') as text_file:
            text_file.write(text)
        print(f"Processed {filename}, saved transcription to {text_file_path}")

# 设置文件夹路径
input_dir = '/home/yz1031/yinghui/MHC/multihateclip_Chinese_audio'
output_dir = '/home/yz1031/yinghui/MHC/Chinese_text'

# 处理音频文件
process_audio_files(input_dir, output_dir)
