# viz (harness/viz/)
> Purpose: record and visualize a run - animation frames + LLM trace, synced.
> Read when: visualizing runs, or changing the viewer.
> Key files: recorder.py, html.py, live.py

## Public API

- TraceRecorder(capture_frames, metadata).record(...) / finish(...) / steps
- TraceStep (step, observation_text, prompt_messages, llm_response, action, reward, success, info, frame)
- render_html(trace, title, fps) -> str; save_html(trace, path, ...) -> Path
- ConsoleTracer.on_step(ts)  (live terminal trace)
- MatplotlibViewer(fps, title).on_step(ts) / close()  (live window)

## How it works

LLMController records a TraceStep per step (observation, prompt, response,
action, reward, frame) and passes it to recorder + on_step. html.py embeds the
frames as PNG data URIs and the steps as JSON into a self-contained viewer
(play/pause/speed/scrub + click-to-jump, animation and trace kept in sync).

## Config

viz.enabled / backend (html|console|live|none) / output / fps / capture_frames.

## Related
- modules/agent.md (recording), modules/eval.md (logging), concepts.md.
