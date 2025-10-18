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
        self.sample_rate = 22050
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        
    def set_api_key(self, api_key: str):
        """Set the ElevenLabs API key"""
        self.api_key = api_key
        self.client = ElevenLabs(api_key=api_key)
        
    def set_voice_id(self, voice_id: str):
        """Set the voice ID for transformation"""
        self.voice_id = voice_id
        
    def get_audio_devices(self):
        """Get list of available audio devices"""
        devices = []
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0 or info['maxOutputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'input_channels': info['maxInputChannels'],
                    'output_channels': info['maxOutputChannels']
                })
        return devices
        
    def set_input_device(self, device_index: int):
        """Set the input audio device"""
        self.input_device = device_index
        
    def set_output_device(self, device_index: int):
        """Set the output audio device"""
        self.output_device = device_index
        
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
            
            # Use the new ElevenLabs API for voice conversion
            try:
                # Use speech_to_speech conversion
                response = self.client.speech_to_speech.convert(
                    voice_id=self.voice_id,
                    audio=wav_buffer.getvalue(),
                    model_id="eleven_multilingual_v2"
                )
                
                # Get the audio data from the response
                output_audio = response
                self.output_queue.put(output_audio)
                
            except Exception as api_error:
                logger.error(f"ElevenLabs API Error: {api_error}")
                # Fallback to direct HTTP request if the new API doesn't work
                url = f"https://api.elevenlabs.io/v1/speech-to-speech/{self.voice_id}"
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "audio/wav",
                    "xi-api-key": self.api_key
                }
                
                response = requests.post(url, headers=headers, data=wav_buffer.getvalue())
                
                if response.status_code == 200:
                    output_audio = response.content
                    self.output_queue.put(output_audio)
                else:
                    logger.error(f"HTTP API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            
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
                    # Play audio using sounddevice
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32768.0
                    
                    if self.output_device is not None:
                        sd.play(audio_float, samplerate=self.sample_rate, device=self.output_device)
                    else:
                        sd.play(audio_float, samplerate=self.sample_rate)
                    sd.wait()
                else:
                    time.sleep(0.01)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in output audio: {e}")
                
    def start(self):
        """Start the voice changer"""
        if not self.api_key or not self.voice_id:
            raise ValueError("API key and voice ID must be set")
            
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
