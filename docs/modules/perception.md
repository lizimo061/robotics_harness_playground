# perception (harness/perception/)
> Purpose: image -> base64 for vision-capable LLMs.
> Read when: enabling vision (agent.use_vision) or encoding camera frames.
> Key files: vision.py

## Public API

- encode_image(image, media_type='image/png') -> base64 str (no data-URI prefix)
- image_to_data_uri(image, media_type) -> 'data:image/png;base64,...'

## Notes

- Requires Pillow. Accepts HxW, HxWx3, HxWx4 uint8 arrays.
- The agent builds a multimodal user message with ChatMessage.user_vision.
- env.render() provides frames; set agent.use_vision=true and ensure the env
  returns an RGB frame.

## Related
- modules/agent.md, modules/viz.md, modules/envs.md (render).
