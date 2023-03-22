# Gradio_Python_Whisper_Transcription_App

Gradio と Whisper を使用した文字起こしサンプルアプリです。

# 実行環境

1.Gradio インストール

```
pip3 install gradio
```

2.FFmpeg インストール

```
brew install ffmpeg
```

3.Whisper インストール

```
pip3 install git+https://github.com/openai/whisper.git
```

4.足りないパッケージなどあればインストール

5. TranscriptionGradioWhisper.py を実行

読み込んだ動画から文字を起こし、CSV 形式で出力します。

---

Faster 手順
https://github.com/guillaumekln/faster-whisper

whisper 更新

```
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
```
