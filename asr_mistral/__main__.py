import argparse
import asyncio
import logging
from functools import partial

from mistralai import Mistral
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from .handler import MistralEventHandler

_LOGGER = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = [
    "en", "zh", "hi", "es", "ar", "fr", "pt", "ru", "de", "ja", "ko", "it", "nl"
]

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="voxtral-mini-transcribe-realtime-2602")
    parser.add_argument("--uri", default="tcp://0.0.0.0:10304")
    parser.add_argument("--enable-agc", action="store_true")
    parser.add_argument("--ms", type=int, default=240, help="Chunk duration in ms (160, 240, 480, 960)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    client = Mistral(api_key=args.api_key)

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="voxtral-rt",
                description="Mistral AI Realtime Transcription",
                attribution=Attribution(name="Mistral AI", url="https://mistral.ai"),
                installed=True,
                supports_transcript_streaming=True,
                version="0.0.1",
                models=[
                    AsrModel(
                        name=args.model,
                        description="Mistral Voxtral Model",
                        attribution=Attribution(name="Mistral AI", url="https://mistral.ai"),
                        installed=True,
                        languages=SUPPORTED_LANGUAGES,
                        version="26.02",
                    )
                ],
            )
        ],
    )

    _LOGGER.info("Server starting [Chunk size: %s ms]", args.ms)

    handler_factory = partial(
        MistralEventHandler,
        wyoming_info=wyoming_info,
        client=client,
        model=args.model,
        enable_agc=args.enable_agc,
        chunk_duration_ms=args.ms,
    )

    server = AsyncServer.from_uri(args.uri)
    await server.run(handler_factory)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:

        pass
