DEFAULT_PROMPT = "Hello! The morning light is coming through the window, and it is a wonderful time to begin something new."
INFOS = """### MOSS-TTS v1.5

Turn written text into speech, with an optional reference recording to clone a voice.

#### Choose a Voice

- **New voice:** enter your text without a reference recording.
- **Clone a voice:** add a clean recording of one speaker. You do not need its transcript.
- **Choose the language:** enter the spoken language when known. The model supports 31 languages, including English, French, Chinese and Japanese.

#### Natural Completion or Target Duration?

- **Natural Completion:** lets the speech finish at its own pace. The selected duration is a maximum, so the recording can be shorter. Choose this when reading the complete script naturally matters most.
- **Target Duration:** asks the speech to fit roughly within the selected duration by adjusting its pacing. Choose this when you have a preferred recording length. The timing is approximate; it does not guarantee an exact fit.
- **For both modes:** the selected duration caps the recording. If it is too short for your script, the speech can be cut off. Increase it or shorten the text.

#### Dialogue with Two or Three Voices

- Add one reference recording per speaker, with only that speaker in each recording.
- Start each turn on a new line with `Speaker 1:`, `Speaker 2:` or `Speaker 3:`. The numbers match the reference order; use every selected speaker.
- Turns play one after another in a single recording. Voices do not overlap.
- **Pause between Multi Speakers sentences** adds 0.2 seconds of silence when the speaker changes by default. Set it to 0 for no added silence. Pauses count toward the whole dialogue's duration.
- The duration applies to the **whole dialogue**. Target Duration shares the available time across its turns.

#### Optional Writing Help

Use **Write a Speech** or **Write a Dialogue** to draft a script from your brief. These options are off by default; leave them off to speak your supplied text directly.
"""
PROMPT_INFOS = """### Speech Text
Write the words to speak. Preserve punctuation for phrasing. For an explicit pause, write `The doors are opening.[pause 1.2s]Please stand clear.`
The released model also accepts Pinyin and IPA pronunciation input and mixed-language text. Avoid image prompts, scene directions or invented emotion tags.
Set Spoken Language to a language name such as `French`; leave it empty for unspecified language. A reference is optional and does not need a transcript.
Temperature 1.7, Top-p 0.8 and Top-k 25 are the released starting values. The optional Write action edits the speech script; it is disabled by default.
Dialogue example (choose two voice references):
```text
Speaker 1: Are you ready to leave?
Speaker 2: Almost.[pause 0.5s]Where did I put my keys?
Speaker 1: They are beside the door.
```
Start every turn on a new line. Unlabelled following lines continue that turn. The labels select references and are not spoken. Write a Speech and Write a Dialogue are separate opt-in enhancer choices; the dialogue writer defaults to two speakers.
"""
DEEPY_INFOS = """MOSS-TTS v1.5 8B: text-to-speech, 24000 Hz mono, optional reference voice cloning.
Use audio_prompt_type='' for one new voice, 'A' with audio_guide, 'AB' with audio_guide and audio_guide2, or 'ABD' adding audio_guide3. No reference transcript is needed.
AB/ABD require all corresponding Speaker 1: through Speaker 2:/3: labels at line starts. Each turn is generated with its own reference, then joined in order; no overlap. duration_seconds caps the complete recording. Target timing is apportioned by turn text length.
pause_seconds inserts silence between speaker changes (default 0.2; 0 disables it); pauses count toward duration_seconds and are deducted from the speech target. No leading/trailing pause is inserted.
alt_prompt is a spoken-language name (e.g. English, French, Chinese), not voice styling. model_mode=0 makes duration_seconds a maximum;
model_mode=1 also requests that target duration through upstream token-duration conditioning. Natural termination can precede the maximum.
No VoiceGenerator voice-design or native TTSD joint dialogue generation.
The Deepy speech template uses gen_speech_from_sample. Plain text-to-speech is available through normal generation with audio_prompt_type='';
gen_speech_from_description is not supported because its voice_description input is not a spoken-language name.
"""
DEEPY_PROMPT_INFOS = """prompt contains exact words to speak; punctuation controls phrasing, [pause 1.2s] requests a pause.
Supports upstream Pinyin/IPA and code-switching. Set alt_prompt to the known language name. Do not insert unsupported emotional tags or scene descriptions.
Recommended temperature=1.7, top_p=0.8, top_k=25; prompt_enhancer='' preserves the supplied script.
Dialogue: Speaker 1: Ready? followed on a new line by Speaker 2: Almost. Labels map directly to reference order; use exactly 2 or 3 speaker IDs for AB or ABD. No stage directions. prompt_enhancer='T' writes speech; 'T1' writes dialogue, both opt-in.
"""
ENHANCER = """Write only the speech script requested by the user. Preserve their language, exact supplied wording when requested, meaning and exclusions.
Do not add voice descriptions, speaker tags, production notes or Markdown. Use natural punctuation. Preserve existing Pinyin, IPA and [pause X.Ys] markers.
Do not rewrite an already supplied script unless requested. Return only the words to speak."""

DIALOGUE_ENHANCER = """Write only a dialogue script for MOSS-TTS from the user's brief, with two speakers unless three are requested. At most three speakers are supported.
Preserve the user's language, characters, meaning, explicit exclusions and exact supplied lines. If the input is already a script, preserve its wording and speaker assignments unless the user requests a rewrite. Never invent a fourth speaker.
Every turn starts on a new line with Speaker 1:, Speaker 2: or Speaker 3:. Keep each character's number consistent, use all requested speakers, and write 6 to 12 turns unless another length is requested.
Only words to speak follow each label. Use natural punctuation. Preserve existing Pinyin, IPA and [pause X.Ys] markers. Do not add unsupported emotion tags, stage directions, voice descriptions, reference transcripts, Markdown or commentary. Example of the format:
Speaker 1: Did you bring the map?
Speaker 2: Yes.[pause 0.5s]The trail starts just past the bridge.
Speaker 1: Then let's go before the rain arrives.
Return only the dialogue lines."""
