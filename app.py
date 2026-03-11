"""Gradio interface for testing the vocal assistant pipeline.

Flow: Record audio → ASR → LLM (streaming) → TTS → Playback
Each stage is timed and displayed in a live timing table.
"""

import asyncio
import time

import gradio as gr
import numpy as np

from assistant.asr import ASRModel
from assistant.llm import LLMClient
from assistant.tts import TTSEngine

# ── Model singletons (lazy-loaded on first request) ──────────────────────────
_asr = ASRModel()
_llm = LLMClient(model="qwen3.5:0.8b")
_tts = TTSEngine(lang_code="a", voice="af_heart")
_models_loaded = False
_models_lock = asyncio.Lock()


async def _ensure_models():
    """Load models in a thread executor so the event loop stays responsive."""
    global _models_loaded
    if _models_loaded:
        return
    async with _models_lock:
        if not _models_loaded:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _asr.load)
            await loop.run_in_executor(None, _tts.load)
            _models_loaded = True


# ── Timing display ────────────────────────────────────────────────────────────
def _timing_html(
    asr: float | None = None,
    llm_first: float | None = None,
    llm_total: float | None = None,
    tts: float | None = None,
    total: float | None = None,
    status: str = "",
) -> str:
    def fmt(v):
        return f"{v:.2f}s" if v is not None else "<span style='color:#aaa'>—</span>"

    def row(label, value, bold=False):
        style = "font-weight:bold;background:#f0f7ff" if bold else ""
        return (
            f"<tr style='{style}'>"
            f"<td style='padding:5px 10px'>{label}</td>"
            f"<td style='padding:5px 10px;text-align:right'>{fmt(value)}</td>"
            f"</tr>"
        )

    status_html = (
        f"<p style='color:#555;font-style:italic;margin:6px 0'>{status}</p>"
        if status
        else ""
    )

    return (
        status_html
        + "<table style='border-collapse:collapse;width:100%;font-size:0.88em;"
        "border:1px solid #ddd;border-radius:6px;overflow:hidden'>"
        "<thead><tr style='background:#e8e8e8'>"
        "<th style='text-align:left;padding:6px 10px'>Stage</th>"
        "<th style='text-align:right;padding:6px 10px'>Latency</th>"
        "</tr></thead><tbody>"
        + row("ASR &nbsp;(Parakeet)", asr)
        + row("LLM &nbsp;– first token", llm_first)
        + row("LLM &nbsp;– full response", llm_total)
        + row("TTS &nbsp;(Kokoro)", tts)
        + row("Total &nbsp;end-to-end", total, bold=True)
        + "</tbody></table>"
    )


# ── Preprocessing helper ──────────────────────────────────────────────────────
def _prepare_audio(sample_rate: int, audio_array: np.ndarray) -> np.ndarray:
    """Normalize and resample to 16 kHz mono float32."""
    audio = audio_array.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.abs(audio).max() > 1.0:
        audio /= 32_768.0  # int16 → float32

    if sample_rate != 16_000:
        n_target = int(round(len(audio) * 16_000 / sample_rate))
        indices = np.linspace(0, len(audio) - 1, n_target)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    return audio


# ── Main streaming handler ────────────────────────────────────────────────────
async def process_audio(audio, history):
    """Async generator: streams chat and timing updates through the pipeline."""
    if audio is None:
        yield history, _timing_html(status="Record audio then click Send."), None
        return

    if not _models_loaded:
        yield history, _timing_html(status="⏳ Loading models (first run, may take a while)…"), None
    await _ensure_models()

    sample_rate, raw = audio
    audio_16k = _prepare_audio(sample_rate, raw)
    t_total_start = time.perf_counter()

    # ── 1. ASR ────────────────────────────────────────────────────────────────
    yield history, _timing_html(status="🎙️ Transcribing speech…"), None
    t0 = time.perf_counter()
    transcript = await _asr.transcribe_async(audio_16k)
    transcript = transcript.strip()
    asr_time = time.perf_counter() - t0

    if not transcript:
        yield history, _timing_html(asr=asr_time, status="⚠️ No speech detected."), None
        return

    history = history + [{"role": "user", "content": transcript}]
    yield history, _timing_html(asr=asr_time, status="💬 Generating response…"), None

    # ── 2. LLM streaming ─────────────────────────────────────────────────────
    assistant_text = ""
    history = history + [{"role": "assistant", "content": "▌"}]

    t_llm = time.perf_counter()
    llm_first_token_time: float | None = None

    async for token in _llm.stream(transcript):
        if llm_first_token_time is None:
            llm_first_token_time = time.perf_counter() - t_llm
        assistant_text += token
        history[-1] = {"role": "assistant", "content": assistant_text + "▌"}
        yield (
            history,
            _timing_html(
                asr=asr_time,
                llm_first=llm_first_token_time,
                status="💬 Generating…",
            ),
            None,
        )

    llm_total_time = time.perf_counter() - t_llm
    history[-1] = {"role": "assistant", "content": assistant_text}
    yield (
        history,
        _timing_html(
            asr=asr_time,
            llm_first=llm_first_token_time,
            llm_total=llm_total_time,
            status="🔊 Synthesizing speech…",
        ),
        None,
    )

    # ── 3. TTS ────────────────────────────────────────────────────────────────
    t_tts = time.perf_counter()
    tts_audio = await _tts.synthesize_async(assistant_text)
    tts_time = time.perf_counter() - t_tts
    total_time = time.perf_counter() - t_total_start

    audio_output = (24_000, tts_audio) if tts_audio.size else None

    yield (
        history,
        _timing_html(
            asr=asr_time,
            llm_first=llm_first_token_time,
            llm_total=llm_total_time,
            tts=tts_time,
            total=total_time,
        ),
        audio_output,
    )


# ── Reset helper ──────────────────────────────────────────────────────────────
def reset_conversation():
    _llm.reset_history()
    return [], _timing_html(status="Conversation reset."), None


# ── Gradio UI ─────────────────────────────────────────────────────────────────
_CSS = """
#chatbot { height: 480px; }
"""

with gr.Blocks(title="Vocal Assistant") as demo:
    gr.Markdown(
        "# 🎙️ Vocal Assistant\n"
        "**Parakeet ASR → Ollama LLM → Kokoro TTS** &nbsp;|&nbsp; "
        "Record your voice, click **Send**, and watch the pipeline run step by step."
    )

    with gr.Row():
        # ── Left column: chat ──────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Conversation",
                layout="bubble",
                avatar_images=(
                    None,
                    "https://api.dicebear.com/7.x/bottts/svg?seed=assistant",
                ),
                show_label=True,
                height=480,
            )

            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Your voice",
                )

            with gr.Row():
                send_btn = gr.Button("Send ▶", variant="primary", scale=3)
                clear_btn = gr.Button("Reset conversation", variant="secondary", scale=1)

        # ── Right column: timing + audio output ───────────────────────────
        with gr.Column(scale=2):
            timing_display = gr.HTML(
                value=_timing_html(status="Record audio then click Send."),
                label="Pipeline timings",
            )
            audio_output = gr.Audio(
                label="Assistant voice",
                type="numpy",
                autoplay=True,
                buttons=["download"],
            )

    # ── Wire up events ────────────────────────────────────────────────────────
    send_btn.click(
        fn=process_audio,
        inputs=[audio_input, chatbot],
        outputs=[chatbot, timing_display, audio_output],
    )

    clear_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[chatbot, timing_display, audio_output],
    )


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft(), css=_CSS)
