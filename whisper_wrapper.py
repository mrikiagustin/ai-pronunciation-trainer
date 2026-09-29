import torch 
from transformers import pipeline
from ModelInterfaces import IASRModel
from typing import Union
import numpy as np 

class WhisperASRModel(IASRModel):
    def __init__(self, model_name="openai/whisper-base", language="en"):
        self.asr = pipeline("automatic-speech-recognition", model=model_name, return_timestamps="word")
        self._transcript = ""
        self._word_locations = []
        self.sample_rate = 16000
        self.language = language

    def processAudio(self, audio:Union[np.ndarray, torch.Tensor]):
        # 'audio' can be a path to a file or a numpy array of audio samples.
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        result = self.asr(
            audio[0],
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )
        self._transcript = result["text"]
        self._word_locations = []
        previous_end = 0.0
        for word_info in result.get("chunks", []):
            start, end = word_info.get("timestamp", (None, None))
            start = previous_end if start is None else float(start)
            end = start + 1.0 if end is None else float(end)
            previous_end = end
            self._word_locations.append(
                {
                    "word": word_info["text"],
                    "start_ts": start * self.sample_rate,
                    "end_ts": end * self.sample_rate,
                    "tag": "processed",
                }
            )

    def getTranscript(self) -> str:
        return self._transcript

    def getWordLocations(self) -> list:
        
        return self._word_locations
