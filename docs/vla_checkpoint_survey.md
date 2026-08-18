# Open-weights VLA checkpoint survey (August 2026)

**Purpose.** Pick a real, downloadable vision-language-action policy to sit behind
`harness/tools/policy_tool.py`'s `run_policy`, served over the HTTP contract in
`harness/policies/remote.py`.

**Epistemic labelling.** Every claim below is tagged:

- **[V]** — verified on this machine: I ran the command, fetched the bytes, or parsed the file.
- **[R]** — read from a page/paper/README. Plausible, but not reproduced here.
- **[U]** — unverified / my estimate. Treat as a guess.

---

## 1. Machine constraints (all measured)

| Constraint | Measured value | How |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, **16380 MiB** total, **1338 MiB already used** by Xorg/GNOME/Slack/Electron → **~14.7 GB actually free** | `nvidia-smi` **[V]** |
| Driver / CUDA | 570.172.08 / CUDA 12.8 | `nvidia-smi` **[V]** |
| GPU arch | Ada Lovelace (`sm_89`) | model name **[V]** |
| Disk | **113 GB free** on `/` (913 G total, 87% used) | `df -h /` **[V]** |
| Harness python | `/home/zimo/miniconda3/bin/python` → **3.12.9** | `--version` **[V]** |
| RoboLab python | `/home/zimo/Documents/RoboLab/.venv/bin/python` → **3.11.13** | `--version` **[V]** |

### The constraint nobody mentioned: the network

This turned out to dominate every decision.

- **All egress is forced through a local proxy** at `127.0.0.1:7897`
  (`http_proxy`/`https_proxy`/`ALL_PROXY=socks://…` are set machine-wide). **[V]**
- **Direct (`--noproxy '*'`) connections to `storage.googleapis.com` and
  `hf-mirror.com` time out** — the proxy is not optional, it is the only route. **[V]**
- Measured throughput to GCS through the proxy: **0.26–0.74 MB/s**. **[V]**
- **Parallelism does not rescue it.** A controlled benchmark on one blob
  (`ThreadPoolExecutor`, 4 MB ranged GETs):

  | streams | aggregate |
  |---|---|
  | 1 | 0.62 MB/s |
  | 8 | 0.74 MB/s |
  | 24 | 0.35 MB/s (worse — proxy contention) |

  **[V]** The pipe, not the connection count, is the limit. `~8` streams is the
  practical ceiling.
- Consequence: **~1 GB/hour**. A 12 GB checkpoint is an overnight job, and a
  32 GB one is effectively out of reach. This should be the first thing you
  check before picking any model.

---

## 2. What RoboLab itself expects

`/home/zimo/Documents/RoboLab/policies/README.md` names five shipped backends **[V]**:

| Backend | Model | Serving protocol |
|---|---|---|
| `pi0_family/` | π₀, π₀-FAST, **π₀.₅**, PaliGemma(-FAST) | OpenPI **WebSocket** |
| `cosmos3/` | Cosmos3-Nano-Policy-DROID | OpenPI WebSocket |
| `gr00t/` | GR00T N1.7 / N1.6 DROID | **ZMQ** |
| `dreamzero/` | DreamZero-DROID | HTTP-ish, port 5000 |
| `volo/` | VoLoAgent (a *proxy* wrapping cosmos3 / pi0-family) | proxy over the above |

Two things matter here:

1. **RoboLab's pi0 client is a `*_jointpos` client.** `policies/pi0_family/client.py`
   is literally named `Pi0DroidJointposClient`, and the README's launch command is
   `--policy.config=pi05_droid_jointpos --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos`. **[V]**
   That variant emits **joint positions**, which is exactly
   `harness/envs/robolab.py`'s `action_mode="joint_position"` (7 joint angles + gripper).
2. **`policies/volo/` is a wrapper, not a model.** It composes
   `OrchestratorMetadataMixin` over cosmos3 / pi0-family and adds depth, camera
   calibration, and `__episode_id`. There is no separate "VoLo checkpoint" to
   download. **[V]**

The observation contract RoboLab sends π₀.₅ (`policies/pi0_family/client.py`) **[V]**:

```python
{
  "observation/exterior_image_1_left": resize_with_pad(over_shoulder_left_camera, 224, 224),
  "observation/wrist_image_left":      resize_with_pad(wrist_cam, 224, 224),
  "observation/joint_position":        arm_joint_pos,     # 7
  "observation/gripper_position":      gripper_pos,       # 1
  "prompt": instruction,
}
# response["actions"] -> chunk; chunk[..., -1] = (chunk[..., -1] > 0.5)  # gripper binarised
# open_loop_horizon: pi05 -> 15, pi0/pi0_fast/paligemma -> 10
```

---

## 3. Survey

Sizes marked **[V]** are exact byte sums I fetched from the HuggingFace
`/api/models/<id>/tree/main?recursive=true` endpoint or the GCS JSON API — measured,
not quoted from a README.

| Model | Weights on disk | Params | Licence | Gated? | Inference VRAM | Action space | Cameras / proprio | Franka+DROID in-dist? | Fits 14.7 GB? |
|---|---|---|---|---|---|---|---|---|---|
| **π₀.₅-DROID-jointpos** `gs://openpi-assets-simeval/pi05_droid_jointpos` | **12.44 GB** (26 objects, fp32 orbax) **[V]** | **3,353,433,872** **[V]** | code Apache-2.0 **[V]**; **weights licence unstated** in openpi, and the HF mirror of the same weights declares `license: gemma` **[V]** → treat as **Gemma Terms of Use** | **No** — anonymous HTTPS GET works, no token **[V]** | **>8 GB** (documented, "RTX 4090") **[R]** | **8-dim = 7 joint positions + gripper**, chunk 15 **[V]** via RoboLab client / **[R]** openpi | exterior 224² **+ wrist 224²**; **proprio required** (7+1) **[V]** | **Yes — DROID *is* a Franka Panda** **[R]** | **Yes** (~6.7 GB bf16) |
| π₀-FAST-DROID-jointpos `gs://…/pi0_fast_droid_jointpos` | in same bucket **[V]** (size not measured) | ~2.92 B **[R]** | as above | No **[V]** | >8 GB **[R]** | 8-dim, autoregressive FAST tokens, chunk 10 **[R]** | same **[V]** | Yes **[R]** | Yes |
| **GR00T-N1.7-DROID** `nvidia/GR00T-N1.7-DROID` | **6.91 GB** bf16 **[V]** | ~3 B **[R]** | repo `README` says NVIDIA Open Model License, but the actual in-repo `LICENSE` file is the **non-commercial** NVIDIA License **[R]** — resolve before commercial use | checkpoint itself `gated: false` **[V]**, **but** RoboLab's own README says N1.7 "requires access to the gated `nvidia/Cosmos-Reason2-2B` backbone" **[V]**, and that repo is `gated: auto` **[V]** → click-through + `HF_TOKEN` | **16 GB+ documented floor**; Lovelace/Ada listed as supported **[R]** | 17-dim DROID head (eef_9d + gripper + 7 joint), `action_horizon: 40` **[V]** from `config.json` | exterior + wrist, 256² target / 230 crop **[V]**; RoboLab uses 180×320 **[V]** | Yes (`oxe_droid` tag, id 17) **[V]** | **Weights yes, but 16 GB is the *documented minimum* — no headroom to co-host Isaac Sim on the same card** |
| GR00T-N1.5-3B | **5.45 GB** **[V]** | ~3 B **[R]** | same non-commercial LICENSE **[R]** | `gated: false` **[V]** | ~16 GB **[R]** | 29-dim state/action, horizon 16 **[R]** | 2 cams **[R]** | needs post-training **[R]** | Yes |
| **Cosmos3-Nano-Policy-DROID** `nvidia/Cosmos3-Nano-Policy-DROID` | **32.94 GB** **[V]** | 16 B **[R]** | `license:other` = OpenMDW-1.1, permissive **[R]** | `gated: false` **[V]** | ≫16 GB **[U]** | 8-dim DROID, chunk 16/32 **[R]** | **1 image** **[R]** | Yes **[R]** | **No** — and a 33 GB download is ~36 h on this link |
| Cosmos3-**Edge**-Policy-DROID | 9.14 GB **[R]** | 4 B **[R]** | OpenMDW-1.1 (permissive, commercial OK) **[R]** | `gated: false` **[R]** | fits on memory **[R]** | **8-dim DROID**, chunk 16/32 **[R]** | **1 image** (256p/480p) **[R]** | Yes **[R]** | Memory yes; **1.25 s/chunk on an H100** and **Ada not in the tested-arch list** **[R]** → likely several s/chunk here |
| Cosmos-Policy-LIBERO-Predict2-2B | 3.91 GB **[R]** | 2 B **[R]** | **NSCLv1, non-commercial**; collection **archived** **[R]** | `gated: false` **[R]** | **6.8 GB documented** **[R]** | 7-dim EEF + gripper, horizon 16 **[R]** | agentview + wrist 224²; proprio 9-dim **[R]** | Franka Panda, but **LIBERO** not DROID **[R]** | Yes, comfortably |
| DreamZero-DROID `GEAR-Dreams/DreamZero-DROID` | **64.79 GB** **[V]** | Wan2.1-I2V-14B based **[V]** (tags) | **CC-BY-NC-4.0** **[V]** | `gated: false` **[V]** | ≫16 GB **[U]** | DROID **[R]** | 2 exterior + wrist, 180×320 **[V]** (RoboLab flags) | Yes **[R]** | **No** — 65 GB is ~70 h on this link |
| OpenVLA-7B `openvla/openvla-7b` | **15.09 GB** **[V]** | 7.54 B **[R]** | **MIT** **[V]** — cleanest licence here | `gated: false` **[V]** | **16.8 GB bf16 / 10.2 GB int8 / 7.0 GB int4** (paper Table 2) **[R]** | 7-dim EEF delta (x,y,z,r,p,y,gripper), 256 bins, **no chunking** **[R]** | **1 third-person cam @224², no proprio at all** **[R]** | **No — DROID is not in its pretraining mix**; card says it does not zero-shot to unseen embodiments **[R]** | bf16 **no**; int4 yes. Dormant since 2025-03 **[R]** |
| SmolVLA `lerobot/smolvla_base` | **0.91 GB** **[V]** | 450 M **[R]** | **none declared** on the repo (`cardData` has no `license`; code is Apache-2.0) **[R]** | `gated: false` **[V]** | ~2–4 GB **[U]** | **6-dim** joint pos, chunk 50 **[R]** | 3 cams @256²; proprio 6-dim **[R]** | **No** — SO-100/SO-101 community corpus; 6-dim head ≠ 7-DoF Panda + gripper **[R]** | Yes, trivially — **and it is the only model here you could also fine-tune locally** |
| Octo-base `rail-berkeley/octo-base` | 0.81 GB **[R]** | 93 M **[R]** | MIT **[R]** | no **[R]** | <2 GB **[U]** | 7-dim EEF delta, chunk 4 **[R]** | 1 cam @256² (+opt wrist); no proprio **[R]** | No (no DROID in mix) **[R]** | Yes — but **dead since 2024-07**, pinned to `jax==0.4.20` + **CUDA 11** + `tensorflow 2.15`; will fight CUDA 12.8 **[R]** |
| RDT-1B | 2.46 GB + encoders **[R]** | 1.2 B **[R]** | MIT **[R]** | no **[R]** | ~5–7 GB *only with precomputed text embeddings* (T5-v1.1-XXL is 44.5 GB) **[R]** | unified **128-dim** vector, chunk 64 **[R]** | up to 3 views **[R]** | single-arm supported **[R]** | marginal, awkward |
| RDT2-VQ | 16.58 GB **[R]** | 8 B **[R]** | Apache-2.0 **[R]** | no **[R]** | ~16 GB measured on a **24 GB** 4090 **[R]** | 20-dim, chunk 24 **[R]** | **2 wrist cams, no third-person camera at all**; no proprio **[R]** | **No — bimanual only** **[R]** | **No** |
| lerobot PyTorch mirrors (`lerobot/pi05_base`, `pi05_droid`, `xvla-base`, `eo1-base`, …) | `pi05_droid` = **16.57 GB** single fp32 safetensors **[V]**; `pi05_base` 14.47 GB **[R]** | 3.6 B **[R]** | `pi05_base` → **`license: gemma`** **[V]** | `gated: false` **[V]** | ~7 GB bf16 **[U]** | as π₀.₅ | as π₀.₅ | Yes | Yes if forced to bf16 — but a **larger download** than the GCS orbax checkpoint, and no `*_jointpos` mirror exists **[V]** |

### Not obtainable

- **π\*0.6 / π0.7** — announced (2025-11 / 2026-04), **weights not released**; no
  `pi06*` prefix exists in `gs://openpi-assets/checkpoints/`. Six open GitHub issues
  ask for them. Third-party reimplementations (`exla-ai/openpie-0.6`) are *not* PI
  weights. **[R]**
- **`nvidia/Cosmos-Reason2-2B`** — `gated: auto` **[V]**. Requires accepting terms and
  an `HF_TOKEN`. **I did not attempt to circumvent this.** It is also a *text-only*
  reasoner, not a policy — its role is the System-2 backbone inside GR00T N1.7. **[R]**
- **Gemini Robotics** — no open weights. **[R]**

---

## 4. Recommendation: `pi05_droid_jointpos`

```
gs://openpi-assets-simeval/pi05_droid_jointpos      (12.44 GB, 26 objects)
```

### Why

1. **Its action vector already *is* the harness's action vector.** This is the
   decisive argument. `harness/envs/robolab.py:733` documents
   `joint_position -> 7 absolute joint angles + gripper`. The `*_jointpos` π₀.₅
   variant emits exactly that, and RoboLab's `Pi0DroidJointposClient` feeds it
   straight to the jointpos env with no conversion beyond binarising the gripper at
   0.5. Every other candidate needs a coordinate-frame or dimensionality transform
   (see §6), which is exactly the kind of silent-failure seam
   `robolab.py`'s own comment warns about ("three different contracts, and using the
   wrong one fails silently").
2. **In-distribution.** DROID is a Franka Panda with one over-shoulder camera and one
   wrist camera — the same rig RoboLab's DROID registrations simulate. π₀.₅-DROID is
   PI's strongest DROID policy. **[R]**
3. **It fits.** 3.353 B params **[V]** → ~6.7 GB in bf16, against a documented
   ">8 GB" inference floor **[R]** and ~14.7 GB free **[V]**. That leaves real
   headroom, which matters because Isaac Sim wants several GB of the *same* card.
   RoboLab's own README anticipates this and sets
   `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` **[V]**.
4. **No gate, no token.** Anonymous HTTPS works **[V]**. Contrast GR00T N1.7, whose
   backbone is gated **[V]**.
5. **RoboLab ships the client.** `policies/pi0_family/client.py` is a working
   reference for the wire format; I do not have to reverse-engineer it. **[V]**

### Why not the runners-up

- **GR00T-N1.7-DROID** (6.91 GB, a third of the download) is tempting, and its
  RoboLab client is the most thoroughly documented. Two things kill it as *first*
  choice: NVIDIA's **documented inference floor is 16 GB**, i.e. the entire card with
  nothing left for Isaac Sim **[R]**; and its backbone repo is **gated** **[V]**, so
  it cannot be acquired unattended. Its in-repo LICENSE is also non-commercial **[R]**.
  Keep it as plan B if you serve the policy on a *different* machine.
- **Cosmos3-Edge-Policy-DROID** has the best licence (OpenMDW-1.1, commercial-OK) and
  the best observation contract for this harness (**a single image** — no wrist camera
  needed). But 1.25 s/chunk on an H100 **[R]** and Ada not in its tested-arch list
  **[R]** make it a research curiosity here, not a controller.
- **Cosmos3-Nano-Policy-DROID / DreamZero-DROID** — 33 GB and 65 GB **[V]**. At
  ~1 GB/h these are 36 h and 70 h downloads, and neither fits 16 GB.
- **OpenVLA / Octo / SmolVLA** all need fine-tuning for a Franka DROID rig, and
  fine-tuning needs 22–72 GB **[R]** — not available here.

### Fallback if you need something running *today*

`lerobot/smolvla_base` — 0.91 GB **[V]**, ~1 h on this link, runs in ~2–4 GB **[U]**,
and is the only candidate you could also fine-tune on this GPU. Be explicit that it is
**out of distribution** for a Franka (6-dim SO-100 action head) and that its weights
carry **no declared licence** **[R]** — it is a scaffolding/plumbing target for
validating the `/act` seam end-to-end, not a policy you should report success numbers
from.

---

## 5. Download status and exact commands

### What is on disk right now

**A download is in progress, not finished.** Be precise about this.

```
/home/zimo/Documents/vla_checkpoints/
├── download_pi05_droid_jointpos.py   # the downloader (resumable)
├── download.log
└── pi05_droid_jointpos/
    ├── assets/droid/norm_stats.json  # complete
    └── params/                       # _METADATA, _sharding, manifest.ocdbt,
                                      # array_metadatas/ complete; the 13 large
                                      # ocdbt data blobs still streaming
```

At the time of writing: **413 MB of 12,435 MB (3.3%)**, sustaining **0.2–0.7 MB/s**,
so **ETA is on the order of 8–15 h**. **[V]** Zero errors or retries so far **[V]**.
The process is detached (`setsid`), so it survives the shell that started it — but it
will not survive a reboot or a proxy outage, hence the resume path below.

### Resume / restart

The script is idempotent: completed files are skipped by size, partial files resume
with an HTTP `Range` request, and network errors retry with backoff. Just run it again.

```bash
cd /home/zimo/Documents/vla_checkpoints
nohup /home/zimo/miniconda3/bin/python download_pi05_droid_jointpos.py >> download.log 2>&1 &
tail -f download.log
```

Check progress / completion:

```bash
du -sh /home/zimo/Documents/vla_checkpoints/pi05_droid_jointpos   # target: 12.44 GB
grep -c COMPLETE /home/zimo/Documents/vla_checkpoints/download.log
```

`gsutil` is on `PATH` **[V]** and would also work, but plain HTTPS was used because
the bucket is anonymously readable and `gsutil` adds a credentials path for no gain:

```bash
# equivalent, if you prefer
gsutil -m cp -r gs://openpi-assets-simeval/pi05_droid_jointpos /home/zimo/Documents/vla_checkpoints/
```

### Verification performed (and its limits)

**Verified [V]:**

- All 26 object names and sizes match the GCS listing; total is exactly
  **12,435,136,033 bytes = 12.435 GB**.
- `params/_METADATA` parses as JSON and describes **51 tensors** totalling
  **3,353,433,872 parameters (3.353 B)**, decomposing as:

  | subtree | params |
  |---|---|
  | `PaliGemma` | 3,351,268,080 |
  | `time_mlp_in` | 1,049,600 |
  | `time_mlp_out` | 1,049,600 |
  | `action_in_proj` | 33,792 |
  | `action_out_proj` | 32,800 |

  This matches the published π₀ architecture (PaliGemma 3B + ~300 M action expert) **[R]**.
- **The action head width is 32, not 8**: `action_in_proj.kernel = [32, 1024]`,
  `action_out_proj.kernel = [1024, 32]`. Consistent with
  `assets/droid/norm_stats.json`, whose `actions`/`state` `mean`/`std`/`q01`/`q99` are
  each **length 32**. openpi pads to `action_dim=32` and the DROID output transform
  slices `[..., :8]` **[R]**. **Do not wire 32 dims into the env.**
- 12.44 GB / 3.353 B ≈ **3.71 bytes/param → the checkpoint is fp32**. It must be cast
  to bf16 at load, or it will not fit. (This is also why the LeRobot mirrors are
  14–16 GB.)

**NOT verified — do not claim otherwise:**

- **I have not loaded the weights.** The checkpoint is incomplete, and loading it
  requires an openpi/JAX environment that is not installed here. The parameter count
  above was computed by parsing orbax's own `_METADATA` sidecar, *not* by
  instantiating the model.
- **I have not run a forward pass**, so the ">8 GB inference" figure is openpi's
  documentation **[R]**, not a measurement on this 4060 Ti.
- **I have not measured latency** on this GPU.

---

## 6. Remaining work to serve it behind `RemotePolicy`

`RemotePolicy` (`harness/policies/remote.py`) posts exactly this to `/act`:

```json
{"instruction": str, "observation_text": str, "env_id": int, "image_b64": str?}
```

and expects `{"action": [float, ...]}` — **one** action vector. π₀.₅-DROID wants two
images, numeric proprio, and returns a **chunk of 15**. Four concrete gaps.

### 6.1 Action space — the good news

**No conversion is needed if you run the env in `joint_position` mode.**

| Env `action_mode` | Vector RoboLab expects | π₀.₅-jointpos gives | Work |
|---|---|---|---|
| `joint_position` (`…jointpos` registrations) | **7 joint angles + gripper = 8** | **7 joint positions + gripper = 8** | **none** — slice `[:8]`, binarise the gripper at 0.5 |
| `ee_pose` (**AbsIK**) | `(x, y, z, qw, qx, qy, qz, gripper)` = **8** | joint targets | needs **forward kinematics** on the predicted joint vector, plus the `_EEF_OFFSET_ROT` / TCP-offset corrections `robolab.py` applies. Lossy and easy to get silently wrong. |
| `ee_delta` (**RelIK**) | `(dx, dy, dz, droll, dpitch, dyaw, gripper)` = **7** | joint targets | FK **and** differencing against the current pose, then per-step clipping. Worse. |

**So: construct `RoboLabEnv(action_mode="joint_position")`.** Two traps:

- `harness/tools/policy_tool.py:115` reads `kind = getattr(env.action_space, "kind", "")
  or "joint_position"`, and `robolab.py:325` sets
  `kind = "ee_delta" if action_mode == "ee_delta" else "joint_position"` — so
  `ee_pose` mode reports itself as `"joint_position"`. If you pick the wrong
  `action_mode` you will send joint angles into an AbsIK env and it will *not* error;
  it will just fly somewhere wrong. Assert the registration name contains `JointPos`.
- π₀.₅ emits **absolute** joint positions. `robolab.py:794` passes `joint_position`
  values straight through as targets, which is correct. Do not add the current pose.

### 6.2 The wire protocol carries neither the wrist camera nor numeric proprio

This is the real integration cost.

| π₀.₅-DROID needs | `/act` payload has | Gap |
|---|---|---|
| `observation/exterior_image_1_left` 224²  | `image_b64` (single frame) | OK — `robolab.py:_extract_image` already *ranks* cameras and prefers the third-person/over-shoulder view **[V]** |
| `observation/wrist_image_left` 224² | — | **missing** |
| `observation/joint_position` (7 floats) | only as text: `"Arm joint positions (rad): (…)"` | **missing as numbers** |
| `observation/gripper_position` (1 float) | only as text: `"Gripper: (…) (higher = more closed)"` | **missing as numbers** |

Do **not** try to recover proprio by parsing `observation_text`. `_fmt_vec` formats
with `f"{x:.3f}"` **[V]** — 3 decimal places, i.e. ~1 mrad of quantisation on every
joint, fed into a policy that was normalised against `q01`/`q99` statistics. It is
also a text contract that exists for the *LLM*, and re-purposing it as a numeric
channel couples two things that should stay independent.

Two honest options:

- **(preferred) Extend the payload.** Add optional `wrist_image_b64` and
  `state: [j1..j7, gripper]` to `/act`, populated by the env-side caller. Everything
  in `remote.py` is additive: the server ignores fields it does not know, and
  `RemotePolicy` already tolerates a `/begin` that 404s. This is a small, honest
  change to `harness/policies/remote.py` + `harness/serving.py` + the `run_policy`
  call site that supplies the observation.
- **(fallback, and say so in results) Send a black wrist frame and zeroed proprio.**
  π₀/π₀.₅ mask `right_wrist_0_rgb` to False during training, but
  `left_wrist_0_rgb`'s mask is **True** — the wrist view is *expected*, not optional
  **[R]**. A black wrist image is out of distribution. Any success rate measured this
  way is not comparable to a published DROID number and must not be reported as one.

### 6.3 Action chunking vs. one-action-per-call

π₀.₅ returns a chunk of **15**; RoboLab executes **8** of them before re-querying
(`open_loop_horizon`) **[V]**. `RemotePolicy.act()` asks for one action per env step.

The chunk must therefore be cached **server-side**, keyed by `env_id`
(`PolicySessionManager` in `harness/serving.py` is already per-`env_id` **[V]**):

- `/act` pops the next action from the cached chunk; re-queries the model when the
  chunk is exhausted (or after 8 of 15, to match RoboLab's validated horizon).
- **`/begin` must flush the cache.** `RemotePolicy.begin()` posts `/begin` on every
  `run_policy` call **[V]**, so this is the natural boundary — a new sub-instruction
  must not replay actions predicted for the previous one.
- Binarise the gripper: `chunk[..., -1] = chunk[..., -1] > 0.5` **[V]**.

**The `monitor_every` interaction is a real bug waiting to happen.**
`policy_tool.py` can now pause a rollout and `abort_policy` it. `RemotePolicy.reset()`
only clears the local `_instruction` — **it does not notify the server** **[V]**. So
after an abort, a stale chunk from the abandoned sub-instruction can still be sitting
in the server cache and will be replayed by whatever runs next. Either have
`abort_policy` trigger a `/begin` (or a new `/reset`), or key the cache on an
instruction id. `continue_policy`, by contrast, is safe — resuming the *same* rollout
should keep the chunk.

### 6.4 Serving stack

RoboLab's π₀ path uses an **OpenPI WebSocket** server; the harness speaks **HTTP
JSON**. You need a small adapter — not a new model server:

```
harness RemotePolicy  --HTTP /act-->  adapter  --in-process openpi policy-->  π₀.₅
```

Concretely:

1. Install openpi in **its own venv** (RoboLab's README is explicit: "**Do not**
   install OpenPI in the same virtual environment as RoboLab" **[V]**).
   Requirements to expect **[R]**: Ubuntu 22.04, Python ≥3.11, `uv`,
   `--recurse-submodules`, and `GIT_LFS_SKIP_SMUDGE=1 uv sync`.
   For `pi05_droid_jointpos` you likely want the RoboLab-referenced fork
   `github.com/xuningy/openpi` (Apache-2.0, last push 2026-04-17 **[V]**), because the
   `pi05_droid_jointpos` *config* is not in the upstream README's checkpoint table
   **[V]** even though the checkpoint is in the bucket.
2. Load once: `create_trained_policy(get_config("pi05_droid_jointpos"), <local dir>)`,
   pointing at `/home/zimo/Documents/vla_checkpoints/pi05_droid_jointpos`.
3. Wrap it in a `BaseHTTPRequestHandler` mirroring `harness/serving.py`'s `/begin`
   + `/act`, doing the chunk caching from §6.3.
4. Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` (or lower) **[V]** — JAX preallocates 75%
   of VRAM by default, which will starve Isaac Sim on the same 16 GB card.
5. Point the harness at it: `RemotePolicy(base_url="http://localhost:8000", action_dim=8)`.

If VRAM contention turns out to be fatal, the protocol is already HTTP — serve π₀.₅
on a second machine and change `base_url`. That is the cheapest escape hatch and is
worth verifying early.

### 6.5 Open questions worth resolving before trusting any number

- **Does `pi05_droid_jointpos` output joint *positions* or *velocities*?* The
  non-jointpos `pi05_droid` is documented as "7 joint **velocity** + gripper position"
  **[R]**; the `*_jointpos` variants are the position variants **[R]**, and RoboLab's
  client name and jointpos registration imply positions **[V]**. **Confirm against
  `openpi`'s `DroidActionSpace` enum before running an eval** — this is exactly the
  silent-failure class §6.1 warns about.
- **`norm_stats` provenance.** The checkpoint ships `assets/droid/norm_stats.json`
  **[V]**; make sure the serving config loads *that* file and not a default.
- **Which camera does `_extract_image` actually pick** in the specific RoboLab task
  you evaluate? It logs `"rendering from camera %s (rank %d)"` **[V]** — read the log
  and confirm it is the over-shoulder view, not a wrist or viewport cam.
- Measure real VRAM and latency on the 4060 Ti. Everything about performance in this
  document is **[R]** or **[U]**.
