import json
import pyaudio
from vosk import Model, KaldiRecognizer

model = Model('model')
rec = KaldiRecognizer(model,16000)
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16,channels=1,rate=16000,input=True,frames_per_buffer=8000)
stream.start_stream()

def listen():
    while True:
        data = stream.read(8000,exception_on_overflow=False)
        if (rec.AcceptWaveform(data)) and (len(data) > 0):
            answer = json.loads(rec.Result())
            if answer['text']:
                yield answer['text']

while True:
    sayed_text = ''
    for text in listen():
        sayed_text.join(text)
    if sayed_text == 'пока':
        break