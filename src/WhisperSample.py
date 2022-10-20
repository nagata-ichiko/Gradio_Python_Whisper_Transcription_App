from multiprocessing.sharedctypes import Value
import whisper
import gradio as gr 

model = whisper.load_model("base")

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
    
    #　時間制限なし、しかし画面表示が行われない
    result = model.transcribe(input, verbose=True, language="ja")
    
    return result

gr.Interface(
    title = 'Whisper Sample App', 
    fn=speechRecognitionModel, 
    inputs=[
        # 音声ファイル
        gr.Audio(type="filepath")
        # 動画ファイル
        # gr.Video(type="filepath")
        # マイク入力
        # gr.inputs.Audio(source="microphone", type="filepath")        
    ],
    outputs=[
        "textbox"
    ],
    live=True).launch()