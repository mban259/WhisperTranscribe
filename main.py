import queue
import threading
from queue import Queue
from typing import Final, Literal
import whisper
import torch
import soundcard as sc
import numpy as np

SAMPLE_RATE: Final[int] = 16000
INTERVAL: Final[int] = 3
BUFFER_SIZE: Final[int] = 4096
MODEL_SIZE: Literal["tiny", "base", "small", "medium", "large"] = "medium"

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use {device} device")

print(f"Loading {MODEL_SIZE} model")
model: whisper.Whisper = whisper.load_model(MODEL_SIZE).to(device)
print("done")

b: np.ndarray = np.full(200, 1 / 200, dtype=np.float32)
q: Queue[np.ndarray] = queue.Queue()


def transcribe() -> None:
    print("Start transcription")
    while True:
        item: np.ndarray = q.get()
        if np.abs(item).max() > 0.001:
            item = whisper.pad_or_trim(item)
            mel: torch.Tensor = whisper.log_mel_spectrogram(item, device=device)
            probs: dict[str, float]
            _, probs = model.detect_language(mel)
            result: whisper.DecodingResult = whisper.decode(model, mel, whisper.DecodingOptions(fp16=(device.type != "cpu")))
            print(f"{max(probs, key=probs.get)}: {result.text}")


thread_transcribe: threading.Thread = threading.Thread(target=transcribe, daemon=True)
thread_transcribe.start()

with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE,
                                                                                          channels=1) as mic:
    audio: np.ndarray = np.empty(SAMPLE_RATE * INTERVAL + BUFFER_SIZE, dtype=np.float32)
    n: int = 0
    print("Start recording")
    while True:
        while n < SAMPLE_RATE * INTERVAL:
            data: np.ndarray = mic.record(BUFFER_SIZE)
            audio[n:n + len(data)] = data.reshape(-1)
            n += len(data)

        # 後半4/5以降から二乗の移動平均最小のところを見つける
        m: int = n * 4 // 5
        vol: np.ndarray = np.convolve(audio[m:n]**2, b, "same")
        m += vol.argmin()
        q.put(audio[:m])

        # 余ったところは次回に
        audio[:n - m] = audio[m:n].copy()
        n = n - m
