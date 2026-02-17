import asyncio
import logging
import numpy as np
import time
from typing import Optional, List

from mistralai import Mistral
from mistralai.models import AudioFormat, TranscriptionStreamTextDelta, TranscriptionStreamDone, RealtimeTranscriptionError
from wyoming.asr import Transcript, TranscriptStart, TranscriptStop, Transcribe
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)
EXPECTED_SAMPLE_RATE = 16000
BYTES_PER_MS = (EXPECTED_SAMPLE_RATE * 2) // 1000

class PassthroughAGC:
    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        return audio_chunk

class StreamAGC:
    def __init__(self, target_level=0.6, max_gain=30.0, min_gain=1.0):
        self.target_level = target_level
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.calib_frames = 12 
        self.calib_peak = 0.0
        self.is_calibrated = False
        self.loud_threshold = 0.031
        self.current_peak_envelope = target_level 
        self.active_max_gain = 1.0 

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        if len(audio_chunk) == 0:
            return audio_chunk
        chunk_max = np.max(np.abs(audio_chunk))
        if not self.is_calibrated:
            self.calib_peak = max(self.calib_peak, chunk_max)
            self.calib_frames -= 1
            if self.calib_frames <= 0 or self.calib_peak > self.loud_threshold:
                self._finalize_calibration()
            return audio_chunk
        if self.active_max_gain <= 1.0:
            return np.clip(audio_chunk, -1.0, 1.0)
        alpha = 0.5 if chunk_max > self.current_peak_envelope else (0.0 if chunk_max < 0.005 else 0.002)
        self.current_peak_envelope = (1 - alpha) * self.current_peak_envelope + alpha * chunk_max
        safe_envelope = max(self.current_peak_envelope, 1e-6)
        target_gain = self.target_level / safe_envelope
        final_gain = np.clip(target_gain, self.min_gain, self.active_max_gain)
        return np.tanh(audio_chunk * final_gain)

    def _finalize_calibration(self):
        self.is_calibrated = True
        self.active_max_gain = 1.0 if self.calib_peak > self.loud_threshold else self.max_gain
        _LOGGER.debug(f"AGC Calibrated. Peak: {self.calib_peak:.2f}")

class MistralEventHandler(AsyncEventHandler):
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        wyoming_info: Info,
        client: Mistral,
        model: str,
        enable_agc: bool = False,
        chunk_duration_ms: int = 480,
        *args, **kwargs,
    ) -> None:
        super().__init__(reader, writer, *args, **kwargs)
        self.wyoming_info_event = wyoming_info.event()
        self.client = client
        self.model = model
        self.enable_agc = enable_agc
        self.target_buffer_size = chunk_duration_ms * BYTES_PER_MS
        
        self.agc = None
        self.language: Optional[str] = None
        self.audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        self.mistral_task: Optional[asyncio.Task] = None
        self.accumulated_text: List[str] = []
        self.internal_buffer = bytearray()
        
        self.stop_received_at: float = 0
        self.fast_response_sent = False
        self.fast_text = ""

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            if transcribe.language:
                self.language = transcribe.language
                _LOGGER.debug("Language: '%s'", self.language)
            return True

        if AudioStart.is_type(event.type):
            self.accumulated_text = []
            self.internal_buffer = bytearray()
            self.fast_response_sent = False
            self.fast_text = ""
            self.agc = StreamAGC() if self.enable_agc else PassthroughAGC()

            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            self.mistral_task = asyncio.create_task(self._process_mistral_stream())
            await self.write_event(TranscriptStart().event())
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            audio_bytes = chunk.audio
            if self.enable_agc:
                samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                samples = self.agc.process(samples)
                audio_bytes = (samples * 32767).astype(np.int16).tobytes()

            self.internal_buffer.extend(audio_bytes)
            while len(self.internal_buffer) >= self.target_buffer_size:
                to_send = self.internal_buffer[:self.target_buffer_size]
                self.internal_buffer = self.internal_buffer[self.target_buffer_size:]
                await self.audio_queue.put(bytes(to_send))
            return True

        if AudioStop.is_type(event.type):
            self.stop_received_at = time.time()
            _LOGGER.debug("AudioStop received. Starting 200ms timer...")
            
            if self.internal_buffer:
                await self.audio_queue.put(bytes(self.internal_buffer))
                self.internal_buffer = bytearray()
            
            await self.audio_queue.put(None)
            
            # Ждем 200 мс и форсируем отправку
            await asyncio.sleep(0.2)
            await self._check_and_send_fast_response()
            return False

        return True

    async def _check_and_send_fast_response(self):
        """Отправляет текущий накопленный текст, не дожидаясь закрытия сокета."""
        if self.fast_response_sent:
            return
        
        self.fast_response_sent = True
        self.fast_text = "".join(self.accumulated_text).strip()
        
        delay = (time.time() - self.stop_received_at) * 1000
        _LOGGER.info("Fast assembly at %.1f ms: '%s'", delay, self.fast_text)
        
        if self.fast_text:
            await self.write_event(Transcript(text=self.fast_text).event())
        await self.write_event(TranscriptStop().event())

    async def _audio_generator(self):
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:
                break
            yield chunk

    async def _process_mistral_stream(self):
        start_process = time.time()
        try:
            audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=EXPECTED_SAMPLE_RATE)
            async for event in self.client.audio.realtime.transcribe_stream(
                audio_stream=self._audio_generator(),
                model=self.model,
                audio_format=audio_format,
            ):
                if isinstance(event, TranscriptionStreamTextDelta):
                    self.accumulated_text.append(event.text)
                    _LOGGER.debug("Mistral Partial: %s", event.text)
                elif isinstance(event, TranscriptionStreamDone):
                    _LOGGER.debug("Mistral stream signal: Done")

            # Финальная диагностика после полного закрытия стрима сервером
            final_server_text = "".join(self.accumulated_text).strip()
            total_delay = (time.time() - self.stop_received_at) * 1000 if self.stop_received_at else 0
            
            _LOGGER.debug("[DIAGNOSTIC] Server closed stream at %.1f ms", total_delay)
            _LOGGER.debug("[DIAGNOSTIC] Final server assembly: '%s'", final_server_text)
            
            if self.fast_text != final_server_text:
                _LOGGER.warning("[DIAGNOSTIC] TEXT MISMATCH! Fast version missed some tokens.")
                _LOGGER.warning("   Fast: '%s'", self.fast_text)
                _LOGGER.warning("   Full: '%s'", final_server_text)
            else:
                _LOGGER.debug("[DIAGNOSTIC] Perfect match. Fast hack was safe.")

            # Если хак почему-то не сработал (редко), отправляем тут
            if not self.fast_response_sent:
                await self._check_and_send_fast_response()

        except Exception:
            _LOGGER.exception("Error in Mistral stream task")
            if not self.fast_response_sent:
                await self.write_event(TranscriptStop().event())