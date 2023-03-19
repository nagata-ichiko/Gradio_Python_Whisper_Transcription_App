from multiprocessing.sharedctypes import Value
from faster_whisper import WhisperModel
model_path = "whisper-large-v2-ct2/"
import whisper
import gradio as gr 
from datetime import timedelta
from srt import Subtitle
import srt

# モデル選択、下に行くほどデカくて遅いが高精度
# model = whisper.load_model("tiny")
# model = whisper.load_model("base")
# model = whisper.load_model("small")
# model = whisper.load_model("medium")
# model = whisper.load_model("large")

# 高速モデル https://github.com/guillaumekln/faster-whisper
# ct2-transformers-converter --model openai/whisper-large-v2 --output_dir whisper-large-v2-ct2 実行
model = WhisperModel(model_path, device="cpu", compute_type="int8")

def speechRecognitionModel(input): 
    # 30秒データに変換
    # audio = whisper.load_audio(input)
    # audio = whisper.pad_or_trim(audio)

    # mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # _, probs = model.detect_language(mel)
    # print(f"言語: {max(probs, key=probs.get)}")

    # # 文字起こし
    # options = whisper.DecodingOptions(fp16=False)
    # result = whisper.decode(model, mel, options)
    
    
    # segments, _ = model.transcribe(input, word_timestamps=True)

    # for segment in segments:
    #     for word in segment.words:
    #         print("[%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
    
    result = model.transcribe(input, language="ja")
    
    seginfo = result["segments"]
    out_text = []

    # segment情報から発言の開始/終了時間とテキストを抜き出し、srt形式で編集する
    for data in seginfo:
        start = data["start"]
        end = data["end"]
        text = data["text"]
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
    # return ""

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
