from pygame import mixer 
from pydub import AudioSegment
from termcolor import colored
from threading import Thread

import io, os, re
import time, wave, json

import webrtcvad, requests, pyaudio
import numpy as np
import traceback

# PUT YOUR ELEVEN LABS API KEY BELOW
API_KEY = ""
if not API_KEY:
    print(colored("\nNo API key found in environment variables.", "yellow"))
    print(colored("Please enter your ElevenLabs API key (from https://elevenlabs.io/subscription):", "light_green"))
    API_KEY = input().strip()

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

clear()
time.sleep(1)

# Verify API key before proceeding
try:
    print(colored("\nVerifying API key...", "blue"))
    headers = {
        "xi-api-key": API_KEY
    }
    response = requests.get("https://api.elevenlabs.io/v1/user", headers=headers)
    if response.status_code != 200:
        print(colored("\nError: Invalid API key or insufficient permissions.", "red"))
        print(colored("Please make sure you have a valid API key from https://elevenlabs.io/subscription", "yellow"))
        exit(1)
    print(colored("API key verified successfully!", "green"))
except Exception as e:
    print(colored(f"\nError verifying API key: {str(e)}", "red"))
    exit(1)

voices_map = {

}

def non_blocking(func):
    def wrapper(*args, **kwargs):
        t = Thread(target=func, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
        return t
    return wrapper


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def get_voices(API_KEY):
    headers = {
        "xi-api-key": API_KEY
    }

    parsed_voices = { }
    voices = requests.get("https://api.elevenlabs.io/v1/voices", headers=headers).json()

    voices = voices['voices']
    
    print(colored("Available voices: ", "blue"))

    for voice in voices:
        category = voice["category"]
        name = remove_emojis(voice["name"]).strip()
        voices_map[name] = voice["voice_id"]
        print(colored(f"    - {name} ({category})", "light_blue"))

    return parsed_voices

get_voices(API_KEY)

while True: 
    voice_input = input(colored("Select your voice: ", "light_green"))
    voice_id = voices_map.get(voice_input)
    if voice_id:
        break
    else:
        print(colored("Invalid voice", "red"))

# Remove transcription option since we're not using it
chunk_transcription_enabled = False

clear()

# Initialize audio
print("\nInitializing audio...")
input_audio = pyaudio.PyAudio()
output_audio = pyaudio.PyAudio()

# List all audio devices
print("\nAvailable Output Devices:")
vb_cable_output_index = None
for i in range(output_audio.get_device_count()):
    dev_info = output_audio.get_device_info_by_index(i)
    if dev_info['maxOutputChannels'] > 0:  # Only show output devices
        print(f"Device {i}: {dev_info['name']}")
        # Look for VB-Audio Cable Input (this is where we send output to)
        if "CABLE Input" in dev_info['name']:
            vb_cable_output_index = i
            print(colored(f"Found VB-Audio Cable: {dev_info['name']}", "green"))

if vb_cable_output_index is None:
    print(colored("Error: Could not find VB-Audio Cable input device", "red"))
    exit(1)

print(colored(f"\nUsing VB-Audio Cable (Device {vb_cable_output_index}) for output", "green"))

# Audio settings
RATE = 48000  # Standard sample rate that works well with VB-Cable
OUTPUT_RATE = 48000  # Keep output rate same as input
CHANNELS = 2  # Stereo
CHUNK = 1152  # 24ms of audio at 48kHz (48000 * 0.024 = 1152 samples)
RECORD_SECONDS = 2  # Reduced for faster response

# VAD settings (webrtcvad requires 16kHz)
VAD_RATE = 16000
VAD_CHUNK = 480  # 30ms of audio at 16kHz

# Speech detection parameters
SPEECH_THRESHOLD_PERCENT = 20  # Reduced from 20 to make it more sensitive
MIN_SILENT_CHUNKS = RATE // CHUNK  # One second of silence
MIN_AUDIO_LEVEL = 900  # Minimum audio level to consider as valid input

# Noise gate settings
NOISE_GATE_THRESHOLD = 100  # Threshold below which audio is considered noise
NOISE_GATE_ATTACK = 0.01  # Attack time in seconds
NOISE_GATE_RELEASE = 0.05  # Release time in seconds

ELEVENLABS_STABILITY = 0.4
ELEVENLABS_SIMILARITY_BOOST = 0.5

def apply_noise_gate(audio_data):
    """Apply noise gate to audio data"""
    # Convert to float32 for processing
    audio_float = audio_data.astype(np.float32) / 32768.0
    
    # Calculate envelope
    envelope = np.abs(audio_float)
    
    # Apply attack and release
    attack_samples = int(NOISE_GATE_ATTACK * RATE)
    release_samples = int(NOISE_GATE_RELEASE * RATE)
    
    # Create gate mask
    gate_mask = np.zeros_like(envelope)
    gate_mask[envelope > (NOISE_GATE_THRESHOLD / 32768.0)] = 1
    
    # Smooth the gate mask
    for i in range(1, len(gate_mask)):
        if gate_mask[i] > gate_mask[i-1]:  # Attack
            gate_mask[i] = min(1.0, gate_mask[i-1] + 1.0/attack_samples)
        else:  # Release
            gate_mask[i] = max(0.0, gate_mask[i-1] - 1.0/release_samples)
    
    # Apply gate
    gated_audio = audio_float * gate_mask
    
    # Convert back to int16
    return (gated_audio * 32768.0).astype(np.int16)

def convert_wav_buffer_to_flac(wav_buffer):
    sound = AudioSegment.from_wav(wav_buffer)
    sound = sound.set_sample_width(2)
    flac_buffer = io.BytesIO()
    sound.export(flac_buffer, format="flac")
    raw_flac_bytes = flac_buffer.getvalue()
    return raw_flac_bytes

voice_chunks = {}
chunk_index = 0

class AudioChunk:
    def __init__(self, wav_buffer):
        global chunk_index
        self.wav_buffer = wav_buffer
        self.flac_data = convert_wav_buffer_to_flac(wav_buffer)
        self.eleven_labs_data = None
        self.processing_started = False
        self.chunk_complete_processing = False
        self.index = chunk_index
        voice_chunks[chunk_index] = self
        chunk_index += 1 
    
    def remove_chunk(self):
        del voice_chunks[self.index]

    def begin_processing(self):
        print(colored("\nStarting audio processing...", "blue"))
        self.processing_started = True
        self.apply_speech_to_speech()

    def apply_speech_to_speech(self):
        try: 
            print(colored("\nSending audio to ElevenLabs...", "blue"))
            url = f"https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}/stream"
            headers = {
                "xi-api-key" : API_KEY
            }
            data = {
                "model_id": "eleven_english_sts_v2",
                "voice_settings" : json.dumps({
                    "stability": ELEVENLABS_STABILITY,    
                    "similarity_boost": ELEVENLABS_SIMILARITY_BOOST,  
                }),
            }

            self.wav_buffer.seek(0) 
            files = {'audio': ('audio.wav', self.wav_buffer, 'audio/wav')}

            print(colored("Waiting for ElevenLabs response...", "blue"))
            response = requests.post(url, headers=headers, data=data, files=files)
            content = response.content
            
            # Add debug info about response
            print(colored(f"Response status code: {response.status_code}", "blue"))
            print(colored(f"Response content length: {len(content)} bytes", "blue"))
            
            if len(content) <= 200: 
                try:
                    load = json.loads(content)
                    details = load["detail"]
                    error_type = details["status"]
                    message = details["message"]

                    print(f'''
    {colored("ElevenLabs returned an error", "light_red")}:
        {colored(f'- Error type: "{error_type}"', "red")}
        {colored(f'- More details: "{message}"', "red")}
                    ''')
                    self.remove_chunk()
                    return 
                except Exception as e:
                    print(colored(f"Error parsing ElevenLabs response: {str(e)}", "red"))
                    self.remove_chunk()
                    return 
                
            print(colored("Received audio response from ElevenLabs", "green"))
            self.eleven_labs_data = io.BytesIO(content) 
            self.chunk_complete_processing = True 

        except Exception as e:
            print(colored(f"Error in speech-to-speech: {str(e)}", "red"))
            traceback.print_exc()
            self.remove_chunk()
            return

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(3)  # Most aggressive (1-3)

# Initialize input stream
input_stream = input_audio.open(format=pyaudio.paInt16,
                               channels=CHANNELS,
                               rate=RATE,
                               input=True,
                               frames_per_buffer=CHUNK)

print(colored("\nStarting audio recording...", "green"))
print(colored("\nSpeak into your microphone!", "yellow"))

def validate_audio(frames):
    """Validate audio input before processing"""
    if not frames:
        print(colored("\nNo audio frames to validate", "yellow"))
        return False
        
    # Check average audio level
    all_audio = np.frombuffer(b''.join(frames), dtype=np.int16)
    avg_level = np.abs(all_audio).mean()
    print(colored(f"\nAverage audio level: {avg_level}", "blue"))
    
    if avg_level < MIN_AUDIO_LEVEL:
        print(colored("Audio level too low, ignoring input", "yellow"))
        return False
        
    # Apply noise gate to frames
    gated_frames = []
    for frame in frames:
        audio_data = np.frombuffer(frame, dtype=np.int16)
        gated_audio = apply_noise_gate(audio_data)
        gated_frames.append(gated_audio.tobytes())
    
    # Check speech content with gated audio
    speech_chunks = 0
    total_chunks = len(gated_frames)
    
    for frame in gated_frames:
        try:
            # Convert stereo to mono and resample to 16kHz for VAD
            audio_array = np.frombuffer(frame, dtype=np.int16)
            stereo = audio_array.reshape(-1, 2)
            mono = stereo.mean(axis=1).astype(np.int16)
            
            # Resample to 16kHz for VAD
            mono_16k = np.interp(
                np.linspace(0, len(mono)-1, VAD_CHUNK),
                np.arange(len(mono)),
                mono
            ).astype(np.int16)
            
            mono_data = mono_16k.tobytes()
            
            if vad.is_speech(mono_data, VAD_RATE):
                speech_chunks += 1
        except Exception as e:
            print(colored(f"Error in VAD processing: {str(e)}", "red"))
            continue
    
    speech_percent = (speech_chunks / total_chunks) * 100
    print(colored(f"Speech content: {speech_percent:.1f}%", "blue"))
    
    if speech_percent < SPEECH_THRESHOLD_PERCENT:
        print(colored("Not enough speech detected, ignoring input", "yellow"))
        return False
        
    return True

@non_blocking
def record_audio():
    print(colored("Recording thread started", "green"))
    while True:
        try:
            frames = []
            num_silent_chunks = 0
            noise_chunks = 0

            one_second = RATE // CHUNK
            frame_cap = one_second * RECORD_SECONDS
            
            print(colored("\nListening for audio...", "yellow"))
            while True:
                try:
                    # Add a small delay to prevent CPU spinning
                    time.sleep(0.001)
                    
                    data = input_stream.read(CHUNK, exception_on_overflow=False)
                    if not data:  # Skip empty frames
                        continue
                        
                    # Convert bytes to int16 array to check audio levels
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_level = np.abs(audio_data).mean()
                    
                    # Print a simple VU meter
                    if audio_level > 0:
                        meter = "#" * int(audio_level / 100)
                        print(f"\rAudio Level: {meter}", end="", flush=True)
                    
                    frames.append(data)
                    if len(frames) >= frame_cap:
                        # Validate audio before processing
                        if not validate_audio(frames):
                            frames = []  # Reset frames
                            print(colored("\nListening for audio...", "yellow"))
                            continue
                            
                        print(colored("\nValid audio detected, processing...", "blue"))
                        try:
                            wav_buffer = io.BytesIO()
                            with wave.open(wav_buffer, 'wb') as wf:
                                wf.setnchannels(CHANNELS)
                                wf.setsampwidth(input_audio.get_sample_size(pyaudio.paInt16))
                                wf.setframerate(RATE)
                                wf.writeframes(b''.join(frames))

                            wav_buffer.seek(0) 

                            chunk = AudioChunk(wav_buffer)
                            chunk.begin_processing()
                            frames = []  # Reset frames after processing
                            break
                        except Exception as e:
                            print(colored(f"\nError creating WAV buffer: {str(e)}", "red"))
                            frames = []  # Reset frames on error
                            continue

                except IOError as e:
                    print(colored(f"\nError reading from audio stream: {e}", "red"))
                    time.sleep(0.1)  # Add delay before retrying
                    continue
                except Exception as e:
                    print(colored(f"\nUnexpected error in audio processing: {str(e)}", "red"))
                    time.sleep(0.1)  # Add delay before retrying
                    continue

        except Exception as e:
            print(colored(f"\nError in recording loop: {str(e)}", "red"))
            time.sleep(1)  # Wait a bit before retrying
            # Try to reset the stream if it's causing issues
            try:
                input_stream.stop_stream()
                input_stream.start_stream()
            except:
                pass

def play_audio(chunk):
    try:
        print(colored("Converting audio format...", "blue"))
        # Convert MP3 to wav format audio data
        audio_segment = AudioSegment.from_file(io.BytesIO(chunk.eleven_labs_data.getvalue()), format="mp3")
        
        # Get the original sample rate
        original_rate = audio_segment.frame_rate
        print(colored(f"Original sample rate: {original_rate}Hz", "blue"))
        
        # Convert to 48kHz stereo (standard rate for VB-Cable)
        print(colored(f"Converting to 48000Hz stereo audio...", "blue"))
        audio_segment = audio_segment.set_frame_rate(48000).set_channels(2).set_sample_width(2)
        
        # Export to WAV format
        temp_wav = "temp_output.wav"
        audio_segment.export(temp_wav, format='wav')
        
        print(colored("Creating output stream...", "blue"))
        # Create an output stream using VB-Audio Cable
        output_stream = output_audio.open(
            format=pyaudio.paInt16,
            channels=2,  # Stereo
            rate=OUTPUT_RATE,
            output=True,
            output_device_index=vb_cable_output_index,
            frames_per_buffer=CHUNK * 2  # Double chunk size for stereo
        )

        # Read and play the audio in chunks
        print(colored("Started playing audio to VB-Audio Cable...", "green"))
        with wave.open(temp_wav, 'rb') as wf:
            # Verify audio format
            print(colored(f"WAV format - Channels: {wf.getnchannels()}, Sample Rate: {wf.getframerate()}", "blue"))
            
            while True:
                data = wf.readframes(CHUNK * 2)  # Double chunk size for stereo
                if not data:
                    break
                output_stream.write(data)
        
        # Clean up
        output_stream.stop_stream()
        output_stream.close()
        os.remove(temp_wav)
        
        chunk.remove_chunk()
        print(colored("Finished playing audio chunk", "green"))
    except Exception as e:
        print(colored(f"Error playing audio: {str(e)}", "red"))
        traceback.print_exc()
        if os.path.exists("temp_output.wav"):
            os.remove("temp_output.wav")
        chunk.remove_chunk()

def play_audio_chunks(): # Race condition galore
    while True:
        if not voice_chunks:  
            time.sleep(0.1)  # Add small delay to prevent CPU spinning
            continue

        keys = list(voice_chunks.keys())
        if len(keys) == 0:
            time.sleep(0.1)  # Add small delay to prevent CPU spinning
            continue
        
        min_key = min(keys)
        chunk = voice_chunks.get(min_key)

        if not chunk:
            continue

        try:
            # Start processing if not already started
            if not chunk.processing_started:
                chunk.begin_processing()
                continue

            if chunk.chunk_complete_processing:
                print(colored(f"Found completed chunk {min_key}", "green"))
                play_audio(chunk)
                print(colored("\nListening for audio...", "yellow"))
                break
        except AttributeError as e: 
            print(colored(f"Error in play_audio_chunks: {str(e)}", "red"))
            continue

def main():
    try:
        print(colored("\nStarting voice changer...", "green"))
        print(colored("Press Ctrl+C to exit", "yellow"))
        
        # Start the recording thread
        record_thread = record_audio()
        
        # Main loop for playing audio chunks
        while True:
            play_audio_chunks()
            time.sleep(0.1)  # Small delay to prevent CPU spinning
            
    except KeyboardInterrupt:
        print(colored("\nStopping voice changer", "red"))
        input_stream.stop_stream()
        input_stream.close()
        input_audio.terminate()
        output_audio.terminate()
        exit()

if __name__ == "__main__":
    main()
