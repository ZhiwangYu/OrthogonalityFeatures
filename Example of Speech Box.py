# -*- coding: utf-8 -*-
"""
此段程序借助.TextGrid文本标记，将整段音频切分成音素单位
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from praatio import textgrid
import os
import soundfile as sf

# 假设你已经有了音节标记的数据，其中包含了每个音节的开始和结束时间
# 这里仅作示例，你需要根据你的数据进行相应的处理



DATASET_PATH = 'data/LJSpeech-1.1/LJSpeech-1.1/wavs_all/'



def get_wav_filenames_without_extension(directory):
    return [f[:-9] for f in os.listdir(directory) if f.endswith('.TextGrid')]

# 使用方法
wav_filenames = get_wav_filenames_without_extension(DATASET_PATH)

def ensure_directory_exists(file_path):
    """
    确保目录存在，如果不存在则创建。
    Args:
        file_path (str): 文件路径。
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"已创建目录：{directory}")
    else:
        print(f"目录已存在：{directory}")
        
# 创建新的文件夹
"""
sptgdirectory = os.path.join('data/LJSpeech-1.1','spectrogram')  # 请替换为你的文件夹路径
os.makedirs(sptgdirectory, exist_ok=True)
"""
audiogdirectory = os.path.join('data/LJSpeech-1.1','audio_segment_all')  # 请替换为你的文件夹路径
os.makedirs(audiogdirectory, exist_ok=True)


for file in wav_filenames:
    # 加载TextGrid文件
    tgFile = DATASET_PATH+file+'.TextGrid'
    wavFile = DATASET_PATH+file+'.wav'
    y, sr = librosa.load(wavFile, sr=None)
    tg = textgrid.openTextgrid(tgFile, includeEmptyIntervals=False)
    try:
        PhoneTier = tg.getTier('phones')
    except:
        try:
            PhoneTier = tg.getTier('utt - phones')
        except:
            PhoneTier = tg.getTier('label - phones')

    for Interval in PhoneTier.entries:
        start = Interval[0]
        end = Interval[1]
        label = Interval[2]
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_segment = y[start_sample:end_sample]
        """
        ######################## 语谱图标签目录 ###########################
        labeldirctectory = os.path.join(sptgdirectory, label)
        os.makedirs(labeldirctectory, exist_ok=True)
        ensure_directory_exists(labeldirctectory)
        """
        ######################## 语音片段标签目录 #########################
        audiolabeldirctectory = os.path.join(audiogdirectory, label)
        os.makedirs(audiolabeldirctectory, exist_ok=True)
        ensure_directory_exists(audiolabeldirctectory)
        
        # 保存语音片段
        audio_filename = f"{audiolabeldirctectory}/{file}_{start}_{end}_{label}.wav"
        sf.write(audio_filename, audio_segment, sr)
        
        """
        # 3. 生成语谱图
        n_mels = 128
        hop_length = 256
        spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_mels=n_mels, hop_length=hop_length)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        

        # 绘制并保存语谱图
        plt.figure(figsize=(8, 4))
        plt.imshow(spectrogram_db, origin='lower', aspect='auto', cmap='viridis')
        plt.axis('off')  # 移除坐标轴
        plt.savefig(f'{labeldirctectory}/{file}_{start}_{end}_{label}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        """


