# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Help texts of EVOKE worlds (UI, Deepy) and the prompt enhancer instructions."""

EVOKE_INFOS = """### What it makes
An explorable world that starts from your **Start Image**: the camera walks, strafes and turns through it 1.5 seconds at a time, and the
world remembers what it has shown, so places you come back to keep their layout. Output: 640x384 at 24 fps. The world is saved in the
**Worlds** gallery, where **Explore** continues it full screen with the keyboard.

### Inputs
- **Start Image** (required for a new world): the first view of the world. Landscapes, streets, interiors and stylized scenes work best
  when they show some depth. The image is center-cropped to 5:3 (640x384), the format the model was trained on: keep what matters
  near the middle.
- **Prompt**: a description of the scene and its look. Lines after it written as exploration instructions walk the camera
  (`[Forward]`, `[Look_Left 3s]`, `[Event] it starts to snow`); without them the camera stays still for the selected length.
- **Continue a World**: a world of the Worlds gallery to extend with the prompt's instructions; choose **Text Only** (the world's
  memory replaces the Start Image), the description can then be left out.

### Memory
The world stores depth views of what it generates and renders them into each new view, for the last 90 seconds of video by default
(the Memory menu of the explorer changes it). It keeps the coarse layout of places, not every detail: a place seen again is recognizable
rather than identical.

### Limits
- Trained at 640x384 only; the camera stays at ground level (no vertical moves).
- Events add things where the camera has not looked yet; they rarely change what the world already remembers.
- Each 1.5 s chunk takes several seconds on a 24 GB GPU.
- The depth model the memory needs (ViGeo) is for non-commercial use only."""

EVOKE_PROMPT_INFOS = """### Writing the prompt
Describe the place in one or two plain sentences, then its look: *"A narrow cobblestone street in an old town on a sunny afternoon,
cafe terraces and flower boxes on both sides. Warm cinematic light, natural colors."* The description should match the Start Image.

### Exploration instructions
Add one line per move after the description:
- `[Forward]`, `[Backward]`, `[Left]`, `[Right]` walk and step sideways; `[Look_Left]`, `[Look_Right]`, `[Look_Up]`, `[Look_Down]` turn;
  `[Wait]` holds the camera still. Commands joined by `+` act together: `[Forward+Look_Right]`.
- A move lasts 1.5 s (one chunk) unless it gives its length: `[Forward 4s]` (rounded to whole chunks).
- `[Event] text` makes something happen during the move: *`[Forward] [Event] the street lamps light up`*. The event is added to the
  description for that move, which lasts at least 3 s so that the event has time to happen.

Example:
```
A frozen tundra plain with dark rocks and distant reindeer under a wide sky. Epic cinematic landscape, clear polar daylight.
[Forward 3s]
[Look_Left 3s]
[Wait] [Event] the daylight fades and an aurora ripples across the sky
```"""

EVOKE_DEEPY_INFOS = """EVOKE Turbo World (model_type evoke_turbo): explorable world from a start image (image_start, image_prompt_type "S"; center-cropped to 5:3,
640x384, the trained format), 24 fps,
1.5 s chunks (36 frames). Prompt = scene description, then optional instruction lines ([Forward], [Look_Left 3s], [Event] text) that walk
the camera; without instructions the camera stays still for video_length frames. Output world_output: "world" (Worlds gallery
.wgpworld), "world_video" or "video". custom_guide = a .wgpworld to continue (image_prompt_type "" then). Memory: geometric (depth views
rendered into the next view), coarse layout of the last 90 s, recognizable not pixel-exact. No vertical camera moves; events rarely
change remembered places. 3 fixed denoising steps, no guidance. Depth model weights are non-commercial.
Camera rates: Look_* turn 15 degrees per second (90 degrees = [Look_Left 6s], an about-turn 12 s).
A picture seen from another spot (a step aside, a turn, closer): image_start = the picture, the instructions of that move only,
world_output "video", then extract_image of its last frame (640x384; upscale it if needed). For an about-turn or an unseen room, an image
editing model inventing the reverse view is faster and sharper."""

EVOKE_DEEPY_PROMPT_INFOS = """Prompt: one or two plain sentences describing the place shown by the start image and its look (lighting, palette, style). Then one
instruction per line: [Forward] [Backward] [Left] [Right] [Look_Left] [Look_Right] [Look_Up] [Look_Down] [Wait], joined with +, optional
length in seconds ([Forward 4s], whole 1.5 s chunks), [Event] followed by what happens (appended to the description for that move, 3 s at
least)."""

EVOKE_ENHANCER_INSTRUCTIONS = """You write the prompt of an EVOKE world, an explorable world that starts from a given image. Write one or two plain sentences
describing the place the image shows (layout, objects, scale, time of day), then its look (lighting, palette, style), as a film still
caption. Keep the user's words, language and explicit wishes. If the user's text contains exploration instruction lines in square
brackets ([Forward], [Look_Left 3s], [Event] ...), keep them unchanged after the description, one per line. Return only the prompt."""
