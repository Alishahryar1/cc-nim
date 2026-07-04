# Seedance Recipe — 16-Section Prompt Template

Canonical source of truth for the 16-section prompt used by `hailuo-film`. The architect (`scripts/architect.py`) fills one copy of this template per shot, and the Hailuo driver (`scripts/hailuo_driver.md`) pastes it into the video-gen prompt field.

Keep all 16 headings present even when a section is brief. Hailuo/Seedance-style models use the structure to disambiguate scene context from action, performance, physics, and output settings.

## Template

```
SCENE CONTEXT
{scenes_before} → this scene → {scenes_after}. Emotional arc position: {arc_position}. Pacing: {pacing}.

ACTIVE REFERENCES
{ref_list}
- Use {asset_id_1} as the recurring hero character.
- Use {asset_id_2} as the primary location/establishing view.
- Use {asset_id_N} for any recurring props, logos, or motifs.

LOCATION MAP
{location_name}: {description}, time of day {time}, weather {weather}, camera-relevant geography {geography}.
Entry path: {entry}. Exit path: {exit}. Fixed landmarks: {landmarks}.

FIRST FRAME / BLOCKING
Composition: {composition}. Subject placement: {subject_blocking}. Background layers: {bg_layers}. Foreground layers: {fg_layers}. Negative space: {neg_space}. Eyeline / look direction: {eyeline}.

FORMAT MODE
{format_mode} (e.g. cinematic film, product motion-graphics, social vertical, documentary, anime, etc.)

OPTICS
Lens: {lens}. Aperture / depth of field: {dof}. Focal behavior: {focus_pull}. Optical effects: {lens_flare, bloom, vignette, chromatic_aberration}. Anamorphic / spherical: {anamorphic}. Filter / glass: {filter}.

CAMERA
Shot size: {shot_size}. Angle: {angle}. Height: {height}. Movement: {movement}. Speed / easing: {speed}. Motion control: {motion_control}. Framing constraint: {framing}. Cut intent: {cut_intent}.

ACTION
{action_description}. Start pose: {start_pose}. End pose: {end_pose}. Key beats: {beats}. Prop interaction: {props}. Screen direction: {screen_direction}. Continuity hook: {continuity}.

PERFORMANCE
Expression: {expression}. Body language: {body_language}. Gaze: {gaze}. Micro-gestures: {micro_gestures}. Energy level: {energy}. Emotion target: {emotion}.

PHYSICS
Cloth / hair: {cloth_hair}. Liquid / particles: {liquid_particles}. Smoke / atmosphere: {smoke_atmosphere}. Rigid-body motion: {rigid}. Scale cues: {scale}. Contact / impact: {contact}. Gravity read: {gravity}.

LIGHTING
Key light: {key_light}. Fill / rim: {fill_rim}. Source motivation: {motivation}. Time-of-day feel: {time_feel}. Shadows: {shadows}. Reflections: {reflections}. Volumetrics: {volumetrics}.

COLOR GRADE
Palette: {palette}. Saturation: {saturation}. Contrast: {contrast}. Highlights / shadows tint: {tint}. Skin tone treatment: {skin_tones}. Color story note: {color_story}.

AUDIO
SFX layer: {sfx_layer}. Music mood / tempo: {music}. Voiceover intent: {voiceover_intent}. Diegetic sound: {diegetic}. Audio-design note: {audio_note}.

STYLE
Aesthetic: {aesthetic}. Mood: {mood}. Reference tone: {reference_tone}. Production value: {production_value}. VFX style: {vfx_style}. Logo / typography rule: {typography}.

OUTPUT SETTINGS
Aspect ratio: {aspect_ratio}. Resolution: {resolution}. Duration: {duration_seconds}s. Frame rate: {fps}. Format: {format}. Loop / one-shot: {loop_or_oneshot}. Safety margin: {safety_margin}.

POSITIVE LOCKS
Always include: {must_include}. Never include: {must_avoid}. Style locks: {style_locks}. Brand locks: {brand_locks}. Continuity locks: {continuity_locks}.
```

## Asset rules

1. **Recurring entities** (hero character, hero locations, signature props) are generated as standalone images first and referenced by `asset_id` in shot prompts. This keeps faces, costumes, and sets consistent across shots.
2. **One asset per concept** — do not ask for multiple characters in one image unless it is a group shot in the shot list.
3. **Asset prompts use the same 16-section template** but with `FORMAT MODE` set to "standalone reference asset" and `OUTPUT SETTINGS` constrained to a single square/vertical/horizontal still image.
4. **First-frame uploads for I2V** must match the shot's `FIRST FRAME / BLOCKING` as closely as possible; include the asset references explicitly.

## Values commonly used

- `aspect_ratio`: `16:9`, `9:16`, `1:1`, `4:3`, `2.39:1`
- `resolution`: `720p`, `1080p`, `4K`
- `format_mode`: `cinematic film`, `documentary`, `social vertical`, `product motion-graphics`, `anime`, `music video`, `corporate`, `real estate`, `event recap`
- `duration_seconds`: 5–10 for typical social clips; up to 6 for Hailuo free-tier quick generations

---
*Adapted from the Seedance recipe for Hailuo image-to-video generation.*
