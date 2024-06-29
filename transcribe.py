import pyaudio
import queue
import whisper
import numpy as np

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024  # Whisper processes data in chunks, adjust size as needed for latency

# Initialize PyAudio
p = pyaudio.PyAudio()

# Initialize Whisper model
model = whisper.load_model("base")  # Use the "base" model for faster processing

# Queue to hold audio data
audio_queue = queue.Queue()

def callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

# Open stream using callback
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

# Start the stream
stream.start_stream()

try:
    print("Recording and transcribing. Press Ctrl+C to stop.")
    while True:
        # Check if there is data in the queue
        if not audio_queue.empty():
            data = audio_queue.get()
            # Convert audio data to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16)
            # Process with Whisper
            mel = model.log_mel_spectrogram(audio_data).unsqueeze(0)
            # Decoding loop for streaming
            result = model.decode(mel, stream_state=None)  # Stream state can be managed for longer contexts
            print(result.text)
except KeyboardInterrupt:
    pass
finally:
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PyAudio
    p.terminate()
