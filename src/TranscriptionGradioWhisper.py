from multiprocessing.sharedctypes import Value
from faster_whisper import WhisperModel
import ctranslate2
model_path = "whisper-large-v2-ct2"
import whisper
import gradio as gr 
from datetime import timedelta
from srt import Subtitle
import srt
import requests
import os
from tqdm import tqdm
# ct2-transformers-converter --model openai/whisper-large-v2 --output_dir whisper-large-v2-ct2 実行
# 高速モデル https://github.com/guillaumekln/faster-whisper
# driverいるかもhttps://teratail.com/questions/344120
# https://www.kkaneko.jp/tools/win/cuda110.html#S3

def modelset():
    filepath = "model/"
    filename = "large-v2.pt"
    
    if os.path.exists(path=filepath+filename) == False:    
        # モデルダウンロード
        print("download model")
        url = "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"
        file_size = int(requests.head(url).headers["Content-Length"])
        res = requests.get(url, stream=True)
        
        os.makedirs("model/", exist_ok=True)
        
        pbar = tqdm(total=file_size, unit="B", unit_scale=True)
        with open(filepath+filename, 'wb') as file:
            for chunk in res.iter_content(chunk_size=1024):
                file.write(chunk)
                pbar.update(len(chunk))
            pbar.close()
            
    
    # if os.path.exists(path=filepath+filename) != False:        
    # モデル変換
    print("convert model")
    
    ctranslate2._ini
    
    ct2 = ctranslate2.Translator("model/large-v2.pt",device="cpu")

    ct2.converters.Converter().convert(output_dir="whisper-large-v2-ct2", quantization="int16", force=True, vmap="whisper-large-v2-ct2/vocab.txt")
    model = WhisperModel(model_path, device="cpu", compute_type="int8")




def speechRecognitionModel(input):     
    segments, _ = model.transcribe(input, beam_size=2, word_timestamps=False)    
    out_text = []

    # segment情報から発言の開始/終了時間とテキストを抜き出し、srt形式で編集する
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        start = segment.start
        end = segment.end
        text = segment.text
        out_line = Subtitle(index=1,\
                            start=timedelta(seconds=timedelta(seconds=start).seconds,\
                            microseconds=timedelta(seconds=start).microseconds),\
                            end=timedelta(seconds=timedelta(seconds=end).seconds,\
                            microseconds=timedelta(seconds=end).microseconds),\
                            content=text,\
                            proprietary='')
        out_text.append(out_line)

    with open("sample" + ".csv", mode="w", encoding="utf-8") as f:
        origin = srt.compose(out_text)
        origin = origin.replace(",",".")
        origin = origin.replace("\n",",")
        origin = origin.replace(",,","\n")
        f.write(origin)
    
    return result

modelset()
gr.Interface(
    title = 'Whisper Sample App', 
    fn=speechRecognitionModel, 
    inputs=[
        # 音声ファイル
        # gr.Audio(type="filepath")
        # 動画ファイル
        gr.Video(type="filepath")
        # マイク入力
        # gr.inputs.Audio(source="microphone", type="filepath")        
    ],
    outputs=[
        "textbox"
    ],
    live=True).launch()