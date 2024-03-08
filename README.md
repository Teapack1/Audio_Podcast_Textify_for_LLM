<h1 align="center">YouTube Playlist Diarization</h1>

Audio diarization app that was developed to download and diarize many podcast episodes at once. Takes an URL of YouTube playlist and diarizes audio of every video in it. Outputs are appended in single dataset.csv file, in format "speaker", "text", "title", "start time", "end time". 

I'd like to thank [@MahmoudAshraf97]([https://github.com/m-bain](https://github.com/MahmoudAshraf97) for developing an accurate and really well working diarization pipeline that i could build on,

<img src="https://github.blog/wp-content/uploads/2020/09/github-stars-logo_Color.png" alt="drawing" width="25"/> **Please, star the project on github (see top-right corner) if you appreciate my contribution to the community!**

## What is it
This application is a fork designed to download and diarize audio from YouTube playlists (or any other site suported by [yt_dlp](https://github.com/yt-dlp/yt-dlp)), diarize the audio. It uses `yt_dlp` for downloading and a DiarizePipeline class for audio diarization. Below you'll find the installation instructions and a basic usage example.

Combines Whisper ASR capabilities with Voice Activity Detection (VAD) and Speaker Embedding to identify the speaker for each sentence in the transcription generated by Whisper. First, the vocals are extracted from the audio to increase the speaker embedding accuracy, then the transcription is generated using Whisper, then the timestamps are corrected and aligned using WhisperX to help minimize diarization error due to time shift. The audio is then passed into MarbleNet for VAD and segmentation to exclude silences, TitaNet is then used to extract speaker embeddings to identify the speaker for each segment, the result is then associated with the timestamps generated by WhisperX to detect the speaker for each word based on timestamps and then realigned using punctuation models to compensate for minor time shifts.


## Installation
`FFMPEG` and `Cython` are needed as prerequisites to install the requirements

```bash
pip install --use-pep517 -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

```

## Usage
Clone the repositary.

`FFMPEG` and `Cython` are needed as prerequisites to install the requirements

Basic usage - insert only playlist URL
```
python main.py -url "https://www.youtube.com/playlist?list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4"
```
Basic usage - add some options
```
python main.py -url "https://www.youtube.com/playlist?list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4" -min 72 -batch 4 -device cpu
```

## Command Line Options

- `-url` **(required)**: URL of the YouTube playlist for downloading.
- `-min` **(optional)**: Start video number in the playlist for downloading. Default is `1`.
- `-max` **(optional)**: End video number in the playlist for downloading. Default is `999`.
- `-batch` **(optional)**: Batch size for diarization processing. Default is `4`.
- `-device` **(optional)**: Device for processing (`cuda` for GPU, `cpu` for CPU). Default is `cuda`.
- `-mtypes` **(optional)**: Precision settings for processing in dictionary format, e.g., `{"cpu": "int8", "cuda": "float16"}`.


## Acknowledgements
Original fork of [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)
This work is based on [OpenAI's Whisper](https://github.com/openai/whisper) , [Faster Whisper](https://github.com/guillaumekln/faster-whisper) , [Nvidia NeMo](https://github.com/NVIDIA/NeMo) , [yt_dlp](https://github.com/yt-dlp/yt-dlp), and [Facebook's Demucs](https://github.com/facebookresearch/demucs)
