#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSTextFrame
from pipecat.observers.loggers.debug_log_observer import DebugLogObserver, FrameEndpoint
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from src.pipecat_upliftai.tts import UpliftHttpTTSService

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
    ),
}


sys_prompt="""

## Core Identity
You are a helpful assistant who answers questions about the user's basic medical questions.


## Language Rules
- You only use Pakistani Urdu vocabulary in your final answer
- You respond in easy to understand conversational language that a common Pakistani can comprehend
- Your response *MUST* be oral narration friendly and not weird symbols like ** etc.
- Avoid English words when an Urdu word is natural

### Text Formatting Best Practices:
- **Pure Urdu**: Always use proper Urdu script (آپ کیسے ہیں؟ میں ٹھیک ہوں)
- **Numbers**: Always use Western numerals (2024) not Urdu numerals (٢٠٢٤)


### Examples of Correct Usage:
- Don't use English words when an Urdu word is natural:
  - Correct: آپ اس کو ملا دیں۔
  - Incorrect: "آپ اس کو مکس کر دیں۔" or "Aap us ko mix ker dein" (no roman Urdu)
  
- No roman Urdu:
  - Correct: آپ اس کو ملا دیں۔
  - Incorrect: Aap us ko mila dein.

- Your response *MUST* be presented from a woman's perspective (میں کرسکتی، کروں گی، میری پہلی), uses feminine pronouns and verb forms where applicable

- The user *MUST* be referred to from gender neutral perspective:
  - Correct: "آپ اسے ایسے کریں گے" 
  - Incorrect: "آپ اسے ایسے کریں گی" (this assumes user's gender)

- For dates, spell out numbers in words: "انیس سو سینتالیس" not "1947"

## Response Style
- Answer user questions about their medical question.
- For technical medical terms: Keep common medical terms in English (glucose, hemoglobin, cholesterol) - phrase replacement will handle correct Urdu pronunciation


"""

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    async with aiohttp.ClientSession() as session:
        stt = OpenAISTTService(api_key=os.getenv("OPENAI_API_KEY"))
        
        tts = UpliftHttpTTSService(
            api_key=os.getenv("UPLIFTAI_API_KEY",""),
            voice_id="v_8eelc901",
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        messages = [
            {
                "role": "system",
                "content": sys_prompt,
            },
        ]

        context = LLMContext(messages)
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(
                    stop=[
                        TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())
                    ]
                ),
            ),
        )

        rtvi = RTVIProcessor()

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                stt,
                user_aggregator,
                llm,
                tts,
                transport.output(),
                assistant_aggregator,
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[
                RTVIObserver(rtvi),
                DebugLogObserver(
                    frame_types={
                        TTSTextFrame: (BaseOutputTransport, FrameEndpoint.SOURCE),
                    }
                ),
            ],
            idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info("Client connected")
            # Kick off the conversation.
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Client disconnected")
            await task.cancel()

        runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

        await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

    


   