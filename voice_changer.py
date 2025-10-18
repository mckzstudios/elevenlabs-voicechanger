import asyncio
import pyaudio
import wave
import io
import json
import requests
import threading
import time
from typing import Optional, List
import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import numpy as np
from elevenlabs import ElevenLabs
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceChanger:
    def __init__(self):
        self.api_key = None
        self.voice_id = None
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.audio_stream = None
        self.output_stream = None
        self.input_device = None
        self.output_device = None
        
        # Audio parameters
        self.sample_rate = 44100  # Will be updated when devices are selected
        self.output_sample_rate = 44100  # Will be updated when output device is selected
        self.chunk_size = 4096  # Larger chunk size for better API compatibility
        self.channels = 1
        self.format = pyaudio.paInt16
        self.api_sample_rate = 22050  # Use a more compatible sample rate for API
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
    def set_api_key(self, api_key: str):
        """Set the ElevenLabs API key"""
        self.api_key = api_key
        self.client = ElevenLabs(api_key=api_key)
        
    def set_voice_id(self, voice_id: str):
        """Set the voice ID for transformation"""
        self.voice_id = voice_id
        
    def validate_voice_id(self):
        """Validate if the voice ID exists and is accessible"""
        if not self.api_key or not self.voice_id:
            return False
            
        try:
            # Try to get voice information to validate
            url = f"https://api.elevenlabs.io/v1/voices/{self.voice_id}"
            headers = {"xi-api-key": self.api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Voice ID {self.voice_id} is valid")
                return True
            elif response.status_code == 404:
                logger.error(f"Voice ID {self.voice_id} not found")
                return False
            elif response.status_code == 401:
                logger.error("Invalid API key")
                return False
            else:
                logger.warning(f"Could not validate voice ID: {response.status_code}")
                return True  # Assume it's valid if we can't check
                
        except Exception as e:
            logger.error(f"Error validating voice ID: {e}")
            return True  # Assume it's valid if we can't check
        
    def get_audio_devices(self):
        """Get list of available audio devices with their supported sample rates"""
        devices = []
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 or info['maxOutputChannels'] > 0:
                # Get supported sample rates for this device
                input_sample_rates = []
                output_sample_rates = []
                
                if info['maxInputChannels'] > 0:
                    input_sample_rates = self._get_supported_sample_rates(i, input=True)
                if info['maxOutputChannels'] > 0:
                    output_sample_rates = self._get_supported_sample_rates(i, input=False)
                
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'input_channels': info['maxInputChannels'],
                    'output_channels': info['maxOutputChannels'],
                    'input_sample_rates': input_sample_rates,
                    'output_sample_rates': output_sample_rates,
                    'default_sample_rate': info['defaultSampleRate']
                })
        return devices
        
    def _get_supported_sample_rates(self, device_index, input=True):
        """Get supported sample rates for a specific device"""
        common_rates = [8000, 11025, 16000, 22050, 44100, 48000, 88200, 96000]
        supported_rates = []
        
        for rate in common_rates:
            try:
                # Try to open a stream with this sample rate
                test_stream = self.p.open(
                    format=self.format,
                    channels=1,
                    rate=rate,
                    input=input,
                    output=not input,
                    input_device_index=device_index if input else None,
                    output_device_index=device_index if not input else None,
                    frames_per_buffer=1024
                )
                test_stream.close()
                supported_rates.append(rate)
            except:
                continue
                
        return supported_rates
        
    def set_input_device(self, device_index: int):
        """Set the input audio device"""
        self.input_device = device_index
        # Get the best sample rate for this input device
        devices = self.get_audio_devices()
        for device in devices:
            if device['index'] == device_index and device['input_sample_rates']:
                # Use the highest quality sample rate available
                self.sample_rate = max(device['input_sample_rates'])
                logger.info(f"Set input sample rate to {self.sample_rate} for device {device['name']}")
                break
        
    def set_output_device(self, device_index: int):
        """Set the output audio device"""
        self.output_device = device_index
        # Get the best sample rate for this output device
        devices = self.get_audio_devices()
        for device in devices:
            if device['index'] == device_index and device['output_sample_rates']:
                # Use the highest quality sample rate available
                self.output_sample_rate = max(device['output_sample_rates'])
                logger.info(f"Set output sample rate to {self.output_sample_rate} for device {device['name']}")
                break
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input"""
        if self.is_running:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
        
    def process_audio_chunk(self, audio_data):
        """Process audio chunk through ElevenLabs API"""
        try:
            # Convert audio data to the format expected by ElevenLabs
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Convert to bytes for API
            audio_bytes = (audio_float * 32767).astype(np.int16).tobytes()
            
            # Create a temporary WAV file in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_bytes)
            
            wav_buffer.seek(0)

            try:
                # Use ElevenLabs speech-to-speech API as documented
                url = f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}"
                headers = {
                    "Accept": "audio/mpeg",
                    "xi-api-key": self.api_key
                }
                
                # Try multiple approaches to fix the 500 error
                
                # Approach 1: Use WAV format with proper parameters
                files = {
                    'audio': ('audio.wav', wav_buffer.getvalue(), 'audio/wav')
                }
                
                data = {
                    'model_id': 'eleven_english_sts_v2',
                    'output_format': 'mp3_44100_128',
                    'voice_settings': '{"stability": 0.5, "similarity_boost": 0.5}',
                    'remove_background_noise': 'false',
                    'file_format': 'other'
                }
                
                response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
                
                # If 500 error, try with different parameters
                if response.status_code == 500:
                    logger.warning("First attempt failed with 500, trying alternative parameters...")
                    
                    # Try with minimal parameters
                    data_minimal = {
                        'model_id': 'eleven_english_sts_v2',
                        'output_format': 'mp3_44100_128'
                    }
                    
                    response = requests.post(url, headers=headers, files=files, data=data_minimal, timeout=30)
                    
                    # If still 500, try without voice_settings
                    if response.status_code == 500:
                        logger.warning("Minimal parameters failed, trying without voice_settings...")
                        
                        data_no_settings = {
                            'model_id': 'eleven_english_sts_v2',
                            'output_format': 'mp3_44100_128',
                            'file_format': 'other'
                        }
                        
                        response = requests.post(url, headers=headers, files=files, data=data_no_settings, timeout=30)
                
                if response.status_code == 200:
                    output_audio = response.content
                    self.output_queue.put(output_audio)
                elif response.status_code == 500:
                    logger.error(f"ElevenLabs API server error (500): {response.text}")
                    logger.warning("Using fallback mode - possible causes:")
                    logger.warning("1. Invalid voice_id")
                    logger.warning("2. Invalid API key")
                    logger.warning("3. Audio format not supported")
                    logger.warning("4. Server-side issues")
                    # Fallback: pass through original audio with optional voice effect
                    if hasattr(self, 'enable_voice_effect') and self.enable_voice_effect:
                        output_audio = self.apply_simple_voice_effect(wav_buffer.getvalue())
                    else:
                        output_audio = wav_buffer.getvalue()
                    self.output_queue.put(output_audio)
                else:
                    logger.error(f"Speech-to-speech API error: {response.status_code} - {response.text}")
                    # Fallback: pass through original audio
                    output_audio = wav_buffer.getvalue()
                    self.output_queue.put(output_audio)
                
            except Exception as api_error:
                logger.error(f"Voice processing error: {api_error}")
                # Fallback: just pass through the original audio
                output_audio = wav_buffer.getvalue()
                self.output_queue.put(output_audio)
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            
    def apply_simple_voice_effect(self, audio_data):
        """Apply a simple voice effect as fallback when API is not available"""
        try:
            # Convert to numpy array for processing
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Apply a simple low-pass filter to reduce static
            # Simple moving average filter
            window_size = 3
            if len(audio_float) > window_size:
                filtered_audio = np.convolve(audio_float, np.ones(window_size)/window_size, mode='same')
            else:
                filtered_audio = audio_float
            
            # Apply a gentle pitch shift effect (lower pitch)
            pitch_shift = 0.9  # Less aggressive pitch shift
            new_length = int(len(filtered_audio) / pitch_shift)
            
            if new_length > 0 and new_length < len(filtered_audio) * 2:  # Reasonable bounds
                indices = np.linspace(0, len(filtered_audio) - 1, new_length)
                audio_shifted = np.interp(indices, np.arange(len(filtered_audio)), filtered_audio)
                
                # Apply additional smoothing to reduce artifacts
                if len(audio_shifted) > 2:
                    audio_shifted[1:-1] = (audio_shifted[:-2] + audio_shifted[1:-1] + audio_shifted[2:]) / 3
            else:
                audio_shifted = filtered_audio
            
            # Convert back to int16 with proper clipping
            audio_shifted = np.clip(audio_shifted, -1.0, 1.0)
            audio_shifted_int = (audio_shifted * 32767).astype(np.int16)
            return audio_shifted_int.tobytes()
            
        except Exception as e:
            logger.error(f"Error applying voice effect: {e}")
            return audio_data
            
    def audio_processor_thread(self):
        """Thread for processing audio chunks"""
        while self.is_running:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get(timeout=0.1)
                    self.process_audio_chunk(audio_data)
                else:
                    time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio processor: {e}")
                
    def output_audio_thread(self):
        """Thread for playing output audio"""
        while self.is_running:
            try:
                if not self.output_queue.empty():
                    audio_data = self.output_queue.get(timeout=0.1)
                    
                    # Handle generator response from ElevenLabs API
                    if hasattr(audio_data, '__iter__') and not isinstance(audio_data, (bytes, bytearray)):
                        # Convert generator to bytes
                        audio_bytes = b''.join(audio_data)
                    else:
                        audio_bytes = audio_data
                    
                    # Check if this is MP3 data from ElevenLabs API
                    if audio_bytes.startswith(b'ID3') or audio_bytes.startswith(b'\xff\xfb'):
                        # This is MP3 data, we need to decode it first
                        try:
                            import io
                            import pydub
                            from pydub import AudioSegment
                            
                            # Load MP3 data
                            audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                            
                            # Convert to raw audio data
                            raw_audio = audio_segment.raw_data
                            sample_rate = audio_segment.frame_rate
                            channels = audio_segment.channels
                            
                            # Convert to the format PyAudio expects
                            if audio_segment.sample_width == 1:
                                format = pyaudio.paUInt8
                            elif audio_segment.sample_width == 2:
                                format = pyaudio.paInt16
                            elif audio_segment.sample_width == 4:
                                format = pyaudio.paInt32
                            else:
                                format = pyaudio.paInt16
                            
                            # Use device-specific sample rate if available, otherwise use MP3 sample rate
                            output_rate = sample_rate
                            if self.output_device is not None:
                                devices = self.get_audio_devices()
                                for device in devices:
                                    if device['index'] == self.output_device and device['output_sample_rates']:
                                        # Find the closest supported sample rate
                                        supported_rates = device['output_sample_rates']
                                        output_rate = min(supported_rates, key=lambda x: abs(x - sample_rate))
                                        break
                            
                            # Play the audio
                            output_stream = self.p.open(
                                format=format,
                                channels=channels,
                                rate=output_rate,
                                output=True,
                                output_device_index=self.output_device
                            )
                            
                            chunk_size = 1024
                            for i in range(0, len(raw_audio), chunk_size):
                                if not self.is_running:
                                    break
                                chunk = raw_audio[i:i+chunk_size]
                                output_stream.write(chunk)
                            
                            output_stream.stop_stream()
                            output_stream.close()
                            
                        except ImportError:
                            logger.error("pydub not installed - cannot decode MP3. Install with: pip install pydub")
                            # Fallback to treating as raw audio
                            self._play_raw_audio(audio_bytes)
                        except Exception as mp3_error:
                            logger.error(f"MP3 decode error: {mp3_error}")
                            # Fallback to treating as raw audio
                            self._play_raw_audio(audio_bytes)
                    else:
                        # This is raw audio data, play it directly
                        self._play_raw_audio(audio_bytes)
                        
                else:
                    time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in output audio: {e}")
                
    def _play_raw_audio(self, audio_bytes):
        """Play raw audio data using device-specific sample rates"""
        try:
            # Get supported sample rates for the output device
            if self.output_device is not None:
                devices = self.get_audio_devices()
                output_device_info = None
                for device in devices:
                    if device['index'] == self.output_device:
                        output_device_info = device
                        break
                
                if output_device_info and output_device_info['output_sample_rates']:
                    # Try device-specific sample rates first
                    sample_rates = output_device_info['output_sample_rates']
                else:
                    # Fallback to common rates
                    sample_rates = [44100, 22050, 16000, 8000, 48000]
            else:
                # Use common rates if no specific device
                sample_rates = [44100, 22050, 16000, 8000, 48000]
            
            audio_played = False
            
            for rate in sample_rates:
                try:
                    # Create a temporary output stream
                    output_stream = self.p.open(
                        format=self.format,
                        channels=self.channels,
                        rate=rate,
                        output=True,
                        output_device_index=self.output_device
                    )
                    
                    # Play the audio in chunks
                    chunk_size = 1024
                    for i in range(0, len(audio_bytes), chunk_size):
                        if not self.is_running:
                            break
                        chunk = audio_bytes[i:i+chunk_size]
                        output_stream.write(chunk)
                    
                    output_stream.stop_stream()
                    output_stream.close()
                    audio_played = True
                    logger.debug(f"Successfully played audio at {rate} Hz")
                    break
                    
                except Exception as pa_error:
                    logger.debug(f"Sample rate {rate} failed: {pa_error}")
                    continue
            
            if not audio_played:
                logger.error("Could not play audio with any sample rate")
                
        except Exception as e:
            logger.error(f"Error playing raw audio: {e}")
                
    def start(self):
        """Start the voice changer"""
        if not self.api_key or not self.voice_id:
            raise ValueError("API key and voice ID must be set")
            
        # Validate voice ID before starting
        if not self.validate_voice_id():
            raise ValueError("Invalid voice ID or API key")
            
        self.is_running = True
        
        # Start audio input stream
        self.audio_stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.input_device,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        
        self.audio_stream.start_stream()
        
        # Start processing threads
        self.processor_thread = threading.Thread(target=self.audio_processor_thread)
        self.output_thread = threading.Thread(target=self.output_audio_thread)
        
        self.processor_thread.start()
        self.output_thread.start()
        
        logger.info("Voice changer started")
        
    def stop(self):
        """Stop the voice changer"""
        self.is_running = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            
        if hasattr(self, 'processor_thread'):
            self.processor_thread.join()
        if hasattr(self, 'output_thread'):
            self.output_thread.join()
            
        logger.info("Voice changer stopped")
        
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        self.p.terminate()

class VoiceChangerGUI:
    def __init__(self):
        self.voice_changer = VoiceChanger()
        self.root = tk.Tk()
        self.root.title("Voice Changer by Hammer")
        self.root.geometry("600x500")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # API Key input
        tk.Label(self.root, text="ElevenLabs API Key:").pack(pady=5)
        self.api_key_var = tk.StringVar()
        api_key_entry = tk.Entry(self.root, textvariable=self.api_key_var, width=50, show="*")
        api_key_entry.pack(pady=5)
        
        # Voice ID input
        tk.Label(self.root, text="Voice ID:").pack(pady=5)
        self.voice_id_var = tk.StringVar()
        voice_id_entry = tk.Entry(self.root, textvariable=self.voice_id_var, width=50)
        voice_id_entry.pack(pady=5)
        
        # Input device selection
        tk.Label(self.root, text="Input Device:").pack(pady=5)
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(self.root, textvariable=self.input_device_var, width=50)
        self.input_device_combo.pack(pady=5)
        
        # Output device selection
        tk.Label(self.root, text="Output Device:").pack(pady=5)
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(self.root, textvariable=self.output_device_var, width=50)
        self.output_device_combo.pack(pady=5)
        
        # Voice effect options
        self.enable_voice_effect_var = tk.BooleanVar(value=True)
        self.voice_effect_checkbox = tk.Checkbutton(
            self.root, 
            text="Enable Voice Effect (when API unavailable)", 
            variable=self.enable_voice_effect_var
        )
        self.voice_effect_checkbox.pack(pady=5)
        
        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        self.start_button = tk.Button(button_frame, text="Start Voice Changer", 
                                    command=self.start_voice_changer, bg="green", fg="white")
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = tk.Button(button_frame, text="Stop Voice Changer", 
                                   command=self.stop_voice_changer, bg="red", fg="white", state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Status: Ready", fg="blue")
        self.status_label.pack(pady=10)
        
        # Load audio devices
        self.load_audio_devices()
        
    def load_audio_devices(self):
        """Load available audio devices"""
        devices = self.voice_changer.get_audio_devices()
        
        input_devices = []
        output_devices = []
        
        for device in devices:
            device_name = f"{device['name']} (Index: {device['index']})"
            if device['input_channels'] > 0:
                input_devices.append((device_name, device['index']))
            if device['output_channels'] > 0:
                output_devices.append((device_name, device['index']))
        
        self.input_device_combo['values'] = [device[0] for device in input_devices]
        self.output_device_combo['values'] = [device[0] for device in output_devices]
        
        # Store device mappings
        self.input_device_map = {device[0]: device[1] for device in input_devices}
        self.output_device_map = {device[0]: device[1] for device in output_devices}
        
    def start_voice_changer(self):
        """Start the voice changer"""
        try:
            api_key = self.api_key_var.get().strip()
            voice_id = self.voice_id_var.get().strip()
            
            if not api_key or not voice_id:
                messagebox.showerror("Error", "Please enter both API key and Voice ID")
                return
                
            # Set API key and voice ID
            self.voice_changer.set_api_key(api_key)
            self.voice_changer.set_voice_id(voice_id)
            
            # Set voice effect preference
            self.voice_changer.enable_voice_effect = self.enable_voice_effect_var.get()
            
            # Set input device
            input_device_name = self.input_device_var.get()
            if input_device_name and input_device_name in self.input_device_map:
                self.voice_changer.set_input_device(self.input_device_map[input_device_name])
            
            # Set output device
            output_device_name = self.output_device_var.get()
            if output_device_name and output_device_name in self.output_device_map:
                self.voice_changer.set_output_device(self.output_device_map[output_device_name])
            
            # Start voice changer
            self.voice_changer.start()
            
            # Update UI
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start voice changer: {str(e)}")
            
    def stop_voice_changer(self):
        """Stop the voice changer"""
        try:
            self.voice_changer.stop()
            
            # Update UI
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Stopped", fg="red")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop voice changer: {str(e)}")
            
    def run(self):
        """Run the GUI"""
        try:
            self.root.mainloop()
        finally:
            self.voice_changer.cleanup()

def main():
    """Main function"""
    app = VoiceChangerGUI()
    app.run()

if __name__ == "__main__":
    main()
