# README screenshots

Captured from the Three.js replay viewer on September 6, 2026 with headless
Chromium through Playwright, at 1600 × 900. These are application screenshots;
ship states, ordnance, damage, and decision text come from the archived battles.
The two cinematic frames hide the HUD. Images are unretouched PNG captures.

| File | Recording | Battle time | View |
| --- | --- | ---: | --- |
| `torpedo-cruiser.png` | GLM–Luna 200-point draft | 100s | `alpha_1`; distance 1.4 hull lengths, azimuth −20°, elevation 25°; HUD hidden |
| `fleet-battle.png` | GLM–Luna 200-point draft | 138s | Fleet framing, contact labels, standard HUD |
| `reactor-breach.png` | GLM–Luna 200-point draft | 137.5s | `beta_3`; distance 6 hull lengths, azimuth 35°, elevation 18°; HUD hidden |
| `intent-consequence.png` | Qwen–GPT-OSS | 184.8s | `alpha_2` decision `e152`; camera on `beta_1`, distance 2 hull lengths, azimuth 35°, elevation 18°; Decisions panel scrolled to ship execution and observed consequences |

Recordings:

- [GLM–Luna](../../../data/recordings/glm_luna_200pt_live_20260906/battle_glm_5_3_flash_vs_gpt_5_6_luna_20260906_140449.json)
- [Qwen–GPT-OSS](../../../data/recordings/intent_qwen_gptoss.json)

To capture similar cinematic frames, stage the recordings as described in the
[root README](../../../README.md#watch-a-replay), then use the snapshot helper:

```bash
uv run playwright install chromium
uv run python scripts/viewer_snapshot.py \
  --recording glm_luna_200pt_live_20260906.json \
  --time 100 --focus alpha_1 --dist 1.4 --az -20 --el 25 \
  --hide-ui --width 1600 --height 900 --out /tmp/torpedo-cruiser.png

uv run python scripts/viewer_snapshot.py \
  --recording glm_luna_200pt_live_20260906.json \
  --time 137.5 --focus beta_3 --dist 6 --az 35 --el 18 \
  --hide-ui --width 1600 --height 900 --out /tmp/reactor-breach.png
```

The interface captures use the same Playwright browser setup and
`window.visualizer`. The decision view opens
`?recording=/recordings/intent_qwen_gptoss.json&decisions=1&ship=alpha_2&decision=e152&t=184.8`.
Camera placement and panel scrolling change the view, not the recorded events.
