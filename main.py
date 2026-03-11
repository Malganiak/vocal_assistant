import asyncio

from assistant.pipeline import VoiceAssistantPipeline


def main():
    pipeline = VoiceAssistantPipeline(
        asr_model="nvidia/parakeet-tdt-0.6b-v3",
        llm_model="qwen3.5:9b",
        tts_lang="a",  # 'a' = American English
        tts_voice="af_heart",
    )
    pipeline.load_models()
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
