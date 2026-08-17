"""Render a recorded trace to a self-contained HTML viewer.

Produces a single file that shows the animation (replayed frames) and the LLM
trace (observation -> response -> action -> reward) side by side, kept in sync.
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import Optional

import numpy as np

from harness.perception.vision import encode_image
from harness.viz.recorder import TraceRecorder

_BLANK: Optional[str] = None


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _blank_frame() -> str:
    global _BLANK
    if _BLANK is None:
        _BLANK = encode_image(np.zeros((1, 1, 3), dtype=np.uint8), "image/png")
    return _BLANK


def _frame_uri(frame: Optional[np.ndarray]) -> str:
    b64 = _blank_frame() if frame is None else encode_image(frame, "image/png")
    return "data:image/png;base64," + b64


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f1115; color: #e6e6e6; display: flex; height: 100vh; }
.left { width: 46%; padding: 16px; display: flex; flex-direction: column; min-width: 320px; }
.right { flex: 1; padding: 16px; overflow-y: auto; border-left: 1px solid #2a2f3a; }
h2 { margin: 0 0 8px; font-size: 18px; }
canvas { width: 100%; aspect-ratio: 1 / 1; background: #000; border-radius: 10px; border: 1px solid #2a2f3a; image-rendering: pixelated; }
.controls { display: flex; align-items: center; gap: 10px; margin: 12px 0; flex-wrap: wrap; }
button, select { background: #1c212b; color: #e6e6e6; border: 1px solid #333a46; padding: 6px 14px; border-radius: 8px; cursor: pointer; font-size: 13px; }
button:hover { background: #262d39; }
input[type=range] { flex: 1; }
.muted { color: #9aa3b2; font-size: 12px; }
.step { border: 1px solid #262c37; border-radius: 10px; margin-bottom: 12px; padding: 12px; background: #161a21; cursor: pointer; }
.step:hover { border-color: #3b4657; }
.step.active { border-color: #4a9eff; background: #1a2332; }
.badge { font-weight: 700; padding: 2px 8px; border-radius: 6px; font-size: 12px; }
.ok { background: #0f3d20; color: #7ee2a0; }
.no { background: #3d1a1a; color: #f0a0a0; }
pre { white-space: pre-wrap; word-break: break-word; background: #0b0e13; padding: 10px; border-radius: 8px; font-size: 12px; margin: 4px 0 10px; max-height: 180px; overflow: auto; }
h3 { margin: 0; font-size: 14px; display: flex; align-items: center; gap: 8px; }
.kv { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #8b95a7; margin-top: 8px; }
.empty { color: #6b7381; font-style: italic; padding: 20px; text-align: center; }
</style>
</head>
<body>
<div class="left">
  <h2>__TITLE__</h2>
  <canvas id="frame" width="512" height="512"></canvas>
  <div class="controls">
    <button id="play">Pause</button>
    <select id="speed">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
      <option value="8">8x</option>
    </select>
    <input id="scrub" type="range" min="0" max="0" value="0">
    <span id="counter" class="muted">0 / 0</span>
  </div>
  <div id="meta" class="muted"></div>
</div>
<div class="right" id="trace"></div>
<script>
var FRAMES = __FRAMES__;
var STEPS = __STEPS__;
var FPS = __FPS__;
var META = __META__;

var N = Math.max(FRAMES.length, STEPS.length);
var idx = 0;
var playing = false;
var speed = 1;
var imgs = [];
var canvas = document.getElementById('frame');
var ctx = canvas.getContext('2d');

function esc(s) {
  if (s === null || s === undefined) { return ''; }
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function preload() {
  for (var i = 0; i < FRAMES.length; i++) {
    var img = new Image();
    img.src = FRAMES[i];
    imgs.push(img);
  }
}

function draw() {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, 512, 512);
  if (imgs.length > 0) {
    var img = imgs[Math.min(idx, imgs.length - 1)];
    if (img && img.complete && img.naturalWidth > 0) {
      var s = Math.min(512 / img.naturalWidth, 512 / img.naturalHeight);
      var w = img.naturalWidth * s;
      var h = img.naturalHeight * s;
      ctx.drawImage(img, (512 - w) / 2, (512 - h) / 2, w, h);
    }
  }
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + N;
  document.getElementById('scrub').value = idx;
  var items = document.querySelectorAll('.step');
  for (var i = 0; i < items.length; i++) { items[i].classList.remove('active'); }
  if (items[idx]) { items[idx].classList.add('active'); }
}

function actionText(a) {
  if (a && a.kind === 'code') { return a.code || ''; }
  return JSON.stringify(a);
}

function buildTrace() {
  if (STEPS.length === 0) {
    document.getElementById('trace').innerHTML = '<div class="empty">No steps recorded.</div>';
    return;
  }
  var out = '';
  for (var i = 0; i < STEPS.length; i++) {
    var s = STEPS[i];
    var badge = s.success ? '<span class="badge ok">success</span>' : '<span class="badge no">step</span>';
    var rw = (s.reward === null || s.reward === undefined) ? '' : '<span class="muted">reward=' + Number(s.reward).toFixed(3) + '</span>';
    out += '<div class="step" data-i="' + i + '">';
    out += '<h3>Step ' + (i + 1) + ' ' + badge + ' ' + rw + '</h3>';
    out += '<div class="kv">Observation</div><pre>' + esc(s.observation_text) + '</pre>';
    out += '<div class="kv">LLM response</div><pre>' + esc(s.llm_response) + '</pre>';
    out += '<div class="kv">Action</div><pre>' + esc(actionText(s.action)) + '</pre>';
    out += '</div>';
  }
  document.getElementById('trace').innerHTML = out;
  var items = document.querySelectorAll('.step');
  for (var i = 0; i < items.length; i++) {
    (function (n) {
      items[n].addEventListener('click', function () { idx = n; setPlaying(false); draw(); });
    })(i);
  }
}

function buildMeta() {
  var parts = [];
  if (META && META.env) { parts.push('env=' + META.env); }
  if (META && META.mode) { parts.push('mode=' + META.mode); }
  if (META && META.llm) { parts.push('llm=' + META.llm); }
  document.getElementById('meta').textContent = parts.join('  |  ');
}

var timer = null;
function setPlaying(p) {
  playing = p;
  document.getElementById('play').textContent = playing ? 'Pause' : 'Play';
  if (playing) { startTimer(); } else { stopTimer(); }
}
function startTimer() {
  stopTimer();
  timer = setInterval(function () {
    idx = (idx + 1) % N;
    draw();
  }, 1000 / FPS / speed);
}
function stopTimer() {
  if (timer) { clearInterval(timer); timer = null; }
}

document.getElementById('play').addEventListener('click', function () { setPlaying(!playing); });
document.getElementById('speed').addEventListener('change', function (e) { speed = parseFloat(e.target.value); if (playing) { startTimer(); } });
document.getElementById('scrub').addEventListener('input', function (e) { idx = parseInt(e.target.value, 10); draw(); });
document.getElementById('scrub').max = Math.max(0, N - 1);

preload();
buildTrace();
buildMeta();
draw();
if (N > 1) { setPlaying(true); }
</script>
</body>
</html>
"""


def render_html(trace: TraceRecorder, *, title: str = "", fps: int = 8) -> str:
    frames = [_frame_uri(s.frame) for s in trace.steps]
    steps_data = [TraceRecorder.step_to_dict(s) for s in trace.steps]

    out = _TEMPLATE
    out = out.replace("__TITLE__", _html.escape(title or "Robotics Harness"))
    out = out.replace("__FRAMES__", json.dumps(frames))
    out = out.replace("__STEPS__", json.dumps(steps_data, default=_default))
    out = out.replace("__FPS__", str(int(fps) if fps else 8))
    out = out.replace("__META__", json.dumps(trace.metadata, default=_default) if trace.metadata else "{}")
    return out


def save_html(trace: TraceRecorder, path, *, title: str = "", fps: int = 8) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(trace, title=title, fps=fps), encoding="utf-8")
    return p
