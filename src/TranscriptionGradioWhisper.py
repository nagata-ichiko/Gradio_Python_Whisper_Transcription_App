from multiprocessing.sharedctypes import Value
from faster_whisper import WhisperModel
model_path = "whisper-large-v2-ct2/"
import whisper
import gradio as gr 
from datetime import timedelta
from srt import Subtitle
import srt

# ct2-transformers-converter --model openai/whisper-large-v2 --output_dir whisper-large-v2-ct2 実行
# 高速モデル https://github.com/guillaumekln/faster-whisper
# driverいるかもhttps://teratail.com/questions/344120
# https://www.kkaneko.jp/tools/win/cuda110.html#S3
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
