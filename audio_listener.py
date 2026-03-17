import time
import queue
import sounddevice as sd
import numpy as np
import webrtcvad
import scipy.io.wavfile as wavfile
import threading

class AudioListener:
    def __init__(self, sample_rate=16000, frame_duration_ms=30):
        # Webrtcvad only supports 8000, 16000, 32000, 48000 Hz
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        
        # VAD aggressiveness: 0 (least aggressive) to 3 (most aggressive)
        self.vad = webrtcvad.Vad(3)
        self.audio_queue = queue.Queue()
        self.listening = False

        # Tuning parameters for speech segmentation
        self.speech_padding_frames = 15  # Frames to keep before/after speech
        self.silence_threshold_frames = 30 # Number of consecutive silent frames to consider speech ended (e.g. 30 * 30ms = 900ms)
        
    def _audio_callback(self, indata, frames, time_info, status):
        """This is called continuously by sounddevice for each audio block."""
        if status:
            print(f"SoundDevice Status: {status}")
        # Convert float32 to int16 for webrtcvad
        audio_int16 = (indata[:, 0] * 32767).astype(np.int16)
        self.audio_queue.put(audio_int16)

    def listen_and_record(self, output_filename="temp_speech.wav"):
        """
        Listens to the microphone continuously.
        When speech is detected, it records until a period of silence.
        Returns the path to the recorded WAV file, or None if stopped.
        """
        self.listening = True
        print("[Audio] Listening for speech...")

        # Start stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.frame_size,
            callback=self._audio_callback
        )
        stream.start()

        ring_buffer = queue.deque(maxlen=self.speech_padding_frames)
        triggered = False
        voiced_frames = []
        silent_frames_run = 0

        try:
            while self.listening:
                frame = self.audio_queue.get()
                
                # Check if frame contains speech
                try:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                except Exception as e:
                    # Sometimes size mismatch happens if queue returns partial frame at the very end
                    continue 

                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    # If ~90% of the ring buffer is speech, trigger recording
                    if num_voiced > 0.9 * ring_buffer.maxlen:
                        triggered = True
                        print("[Audio] Speech detected! Recording...")
                        for f, s in ring_buffer:
                            voiced_frames.append(f)
                        ring_buffer.clear()
                else:
                    voiced_frames.append(frame)
                    ring_buffer.append((frame, is_speech))
                    
                    if not is_speech:
                        silent_frames_run += 1
                    else:
                        silent_frames_run = 0

                    # Stop recording if we hit the silence threshold
                    if silent_frames_run > self.silence_threshold_frames:
                        print("[Audio] Silence detected. Stopping recording.")
                        break
                        
        except KeyboardInterrupt:
            print("[Audio] Stopped by user.")
        finally:
            stream.stop()
            stream.close()

        if len(voiced_frames) == 0:
            return None

        # Concatenate and save
        audio_data = np.concatenate(voiced_frames)
        wavfile.write(output_filename, self.sample_rate, audio_data)
        print(f"[Audio] Saved recording to {output_filename}")
        return output_filename

if __name__ == "__main__":
    # Simple test
    listener = AudioListener()
    filename = listener.listen_and_record("test_mic.wav")
    print(f"Test finished. File: {filename}")
