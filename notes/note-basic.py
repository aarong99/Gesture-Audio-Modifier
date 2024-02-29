import pyaudio
import numpy as np
import wave
import time
from pydub import AudioSegment

def pitch_shift(sound, n):
    octaves = n / 12
    new_sample_rate = int(sound.frame_rate * (2.0 ** octaves))
    pitch_shifted_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})
    return pitch_shifted_sound.set_frame_rate(sound.frame_rate)

def play_audio(audio):
    chunk = 1024
    wf = wave.open(audio, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = b''.join(frames)

    sound = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")

    # pitch down one octave
    pitch_down_sound = pitch_shift(sound, -12)
    
    # pitch up one octave
    pitch_up_sound = pitch_shift(sound, 12)

    # two new notes that act as a major harmony
    major_harmony_sound_1 = pitch_shift(sound, 4)
    
    major_harmony_sound_2 = pitch_shift(sound, 7)

    # two new notes that act as a minor harmony
    minor_harmony_sound_1 = pitch_shift(sound, -3)
    
    minor_harmony_sound_2 = pitch_shift(sound, -7)

     # reverb effect
    reverb_effect_sound = sound + AudioSegment.from_file("reverb.wav")

    # play audio based on user input
    user_input_number = int(input("Enter a number between 1-5: "))
    if user_input_number == 1:
        play_audio(pitch_down_sound.export(format='wav'))
    elif user_input_number == 2:
        play_audio(pitch_up_sound.export(format='wav'))
    elif user_input_number == 3:
        play_audio(major_harmony_sound_1.overlay(major_harmony_sound_2).export(format='wav'))
    elif user_input_number == 4:
        play_audio(minor_harmony_sound_1.overlay(minor_harmony_sound_2).export(format='wav'))
    elif user_input_number == 5:
        play_audio(reverb_effect_sound.export(format='wav'))

if __name__ == "__main__":
   main()