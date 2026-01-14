#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Uplift TTS service integration."""

from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


class UpliftHttpTTSService(TTSService):
    """Uplift HTTP TTS service implementation.

    This service provides text-to-speech synthesis using the Uplift HTTP API.
    
    Supported voices:
        - v_8eelc901 (Info/Edu)
        - v_kwmp7zxt (Gen Z)
        - v_yypgzenx (Dada Jee)
        - v_30s70t3a (Nostalgic News)
    
    Supported output formats:
        - WAV_22050_16, WAV_22050_32
        - MP3_22050_32, MP3_22050_64, MP3_22050_128
        - OGG_22050_16
        - ULAW_8000_8
    """

    # Available voices
    AVAILABLE_VOICES = [
        "v_8eelc901",  # Info/Edu
        "v_kwmp7zxt",  # Gen Z
        "v_yypgzenx",  # Dada Jee
        "v_30s70t3a",  # Nostalgic News
    ]

    # Available output formats
    AVAILABLE_FORMATS = [
        "WAV_22050_16",
        "WAV_22050_32",
        "MP3_22050_32",
        "MP3_22050_64",
        "MP3_22050_128",
        "OGG_22050_16",
        "ULAW_8000_8",
    ]

    class InputParams(BaseModel):
        """Optional input parameters for Uplift TTS configuration.

        Parameters:
            voice_id: Optional default voice ID override.
            output_format: Audio output format. Defaults to WAV_22050_16.
        """

        voice_id: Optional[str] = None
        output_format: Optional[str] = "WAV_22050_16"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.upliftai.org/v1/synthesis/text-to-speech",
        voice_id: str = "v_8eelc901",
        output_format: str = "WAV_22050_16",
        sample_rate: int = 22050,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initializes the Uplift HTTP TTS service.

        Args:
            api_key: Uplift API key for authentication (format: "sk_api_...").
            base_url: Base URL for the Uplift TTS API endpoint.
            voice_id: Uplift voice identifier. Available voices:
                - v_8eelc901 (Info/Edu)
                - v_kwmp7zxt (Gen Z)
                - v_yypgzenx (Dada Jee)
                - v_30s70t3a (Nostalgic News)
            output_format: Audio output format. Defaults to WAV_22050_16.
                Available formats: WAV_22050_16, WAV_22050_32, MP3_22050_32,
                MP3_22050_64, MP3_22050_128, OGG_22050_16, ULAW_8000_8
            sample_rate: Audio sample rate in Hz. Defaults to 22050.
            aiohttp_session: Optional aiohttp session for making requests.
            params: Voice customization parameters.
            **kwargs: Additional arguments passed to parent TTSService.

        Raises:
            ValueError: If api_key is not provided or invalid settings.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        if not api_key:
            raise ValueError("Missing Uplift API key")

        self._api_key = api_key
        self._base_url = base_url
        self._params = params or UpliftHttpTTSService.InputParams()

        # Session management: create session if not provided
        self._session = aiohttp_session
        self._created_session = False
        if self._session is None:
            self._session = aiohttp.ClientSession()
            self._created_session = True
            logger.debug("Created internal aiohttp session")

        # Validate and set output format
        final_output_format = self._params.output_format or output_format
        if final_output_format not in self.AVAILABLE_FORMATS:
            logger.warning(
                f"Output format '{final_output_format}' not in known formats list. "
                f"Available formats: {', '.join(self.AVAILABLE_FORMATS)}"
            )

        # Initialize settings dictionary for dynamic updates
        self._settings = {
            "voice_id": self._params.voice_id or voice_id,
            "output_format": final_output_format,
        }

        # Validate voice_id
        if self._settings["voice_id"] not in self.AVAILABLE_VOICES:
            logger.warning(
                f"Voice '{self._settings['voice_id']}' not in known voices list. "
                f"Available voices: {', '.join(self.AVAILABLE_VOICES)}"
            )

        # Voice_id configuration
        self.set_voice(self._settings["voice_id"])

        logger.debug(
            f"Uplift HTTP TTS initialized with voice_id={self._settings['voice_id']}, "
            f"output_format={self._settings['output_format']}"
        )

    async def start(self, frame: StartFrame):
        """Start the service and initialize sample rate from pipeline.
        
        This method is called when the pipeline starts and receives the
        sample rate configuration from the StartFrame as per Pipecat guidelines.
        
        Args:
            frame: StartFrame containing pipeline configuration.
        """
        await super().start(frame)
        # Sample rate is now available via self.sample_rate property from parent class
        logger.debug(f"Uplift TTS started with sample_rate={self.sample_rate} Hz")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Uplift HTTP TTS service supports metrics generation.
        """
        return True

    async def set_voice_id(self, voice_id: str):
        """Set the voice ID for TTS generation.

        Args:
            voice_id: The Uplift voice identifier to use.
        """
        if voice_id not in self.AVAILABLE_VOICES:
            logger.warning(
                f"Voice '{voice_id}' not in known voices list. "
                f"Available voices: {', '.join(self.AVAILABLE_VOICES)}"
            )
        
        logger.info(f"Switching Uplift TTS voice to: [{voice_id}]")
        self._settings["voice_id"] = voice_id
        self.set_voice(voice_id)

    async def set_output_format(self, output_format: str):
        """Set the output audio format.

        Args:
            output_format: The audio format to use.
                Available: WAV_22050_16, WAV_22050_32, MP3_22050_32, 
                MP3_22050_64, MP3_22050_128, OGG_22050_16, ULAW_8000_8
        """
        if output_format not in self.AVAILABLE_FORMATS:
            logger.warning(
                f"Output format '{output_format}' not in known formats list. "
                f"Available formats: {', '.join(self.AVAILABLE_FORMATS)}"
            )
        
        logger.info(f"Switching Uplift TTS output format to: [{output_format}]")
        self._settings["output_format"] = output_format

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.cleanup()

    async def cleanup(self):
        """Cleanup resources, closing session if we created it."""
        if self._created_session and self._session:
            await self._session.close()
            self._session = None
            logger.debug("Closed internal aiohttp session")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Uplift's TTS endpoint.

        Args:
            text: The text to synthesize into speech. Maximum length: 2500 characters.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Validate text length
        if len(text) > 2500:
            logger.warning(
                f"Text length ({len(text)}) exceeds Uplift's maximum of 2500 characters. "
                f"Truncating..."
            )
            text = text[:2500]

        try:
            await self.start_ttfb_metrics()

            # Setting the header and payload for the api call
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "text": text,
                "voiceId": self._settings["voice_id"],
                "outputFormat": self._settings["output_format"],
            }

            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_message = f"Uplift HTTP TTS error: {response.status} - {error_text}"
                    logger.error(error_message)
                    
                    # Push error to pipeline
                    await self.push_error(ErrorFrame(error=error_message))
                    
                    # Also yield error frame for backward compatibility
                    yield ErrorFrame(error=error_message)
                    return

                audio_bytes = await response.read()

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()

                # WAV formats include a 44-byte header that must be skipped
                # MP3 and OGG formats contain raw audio data without headers
                if self._settings["output_format"].startswith("WAV_"):
                    audio_content = audio_bytes[44:]
                    logger.debug("Skipping WAV header (44 bytes)")
                else:
                    audio_content = audio_bytes
                    logger.debug("Using raw audio bytes (no header to skip)")

                CHUNK_SIZE = self.chunk_size

                for i in range(0, len(audio_content), CHUNK_SIZE):
                    chunk = audio_content[i : i + CHUNK_SIZE]
                    if not chunk:
                        break
                    await self.stop_ttfb_metrics()
                    # Wrapping audio bytes in a Pipecat frame
                    frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                    yield frame

                yield TTSStoppedFrame()

        except Exception as e:
            error_message = f"TTS generation error: {str(e)}"
            logger.error(error_message)
            
            # error to pipeline
            await self.push_error(ErrorFrame(error=error_message))
            
            # yield error frame
            yield ErrorFrame(error=error_message)