# Avatar animation sheets

LocalPilot's illustrated companion uses one independent hand-drawn sprite sheet per primary visual state. This keeps the artwork expressive and avoids the whole-character squeeze/warp effect that occurred when all states were packed into one synthesized atlas.

Primary sheets live in `localpilot/webview/avatar/anim/`:

- `idle.png`
- `listening.png`
- `thinking.png`
- `researching.png`
- `working.png`
- `speaking.png`
- `success.png`
- `error.png` (currently shared by `uncertain` and `error`)
- `sleeping.png`
- `offline.png`

The runtime manifest records each sheet's grid geometry, frame count, expected byte size, playback timing, and runtime-state mapping. `learning` temporarily reuses the Thinking artwork and `restarting` temporarily reuses Working until dedicated sheets are drawn.

Generated sheets are not assumed to have perfectly square or evenly divisible source cells. The native renderer and WebView both split the declared grid using proportional boundaries, alpha-trim each frame, preserve its aspect ratio, and fit it into the avatar viewport. This prevents stretching the character merely to normalize source-sheet geometry.

Dedicated transition sheets are intentionally deferred. Native mode switches directly between completed state loops; the WebView uses a short opacity crossfade. When transition sheets are later supplied, they can be added without redesigning the state assets.

The legacy pixel avatar remains the fail-safe whenever the manifest, an expected sheet, or native frame materialization fails validation.
