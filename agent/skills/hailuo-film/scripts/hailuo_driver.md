# Hailuo Browser Driver Playbook

Read this at runtime when driving Hailuo image-gen and I2V video-gen for `hailuo-film`.

## Sign-in gate

1. Open `https://hailuoai.video/` via Chrome DevTools MCP `new_page`.
2. Check the top-right for an avatar / credit counter. If absent, the user is not signed in.
3. Ask the user: "Please sign in to Hailuo in the browser, then reply 'done'." Do not proceed until they confirm.
4. If the credit counter shows `0`, warn the user and ask whether to continue (generations will fail).

## Common selectors (see `selectors.json`)

- **Prompt input**: `div[contenteditable="true"][role="textbox"]` (fallback: `div[contenteditable="true"]`). It is a React contenteditable. Use Playwright `fill` on it, or set `innerText` and dispatch `input` + `change` events.
- **Create button**: `button` with accessible name "Create" (verify via snapshot — there is one per page).
- **File upload input**: hidden `input[type="file"][accept*=".png"]`; use `upload_file` against this element.
- **Tabs**: "Create Video" / "Create Image" in the tablist.

## Image generation flow (for each asset)

1. Navigate to `https://hailuoai.video/create/image-generation`.
2. Click the "Create Image" tab if not active.
3. Click the prompt input.
4. Type the asset prompt (16-section Seedance recipe, `FORMAT MODE` = standalone reference asset).
5. If the desired aspect ratio differs from the default, click the aspect-ratio toggle and pick the value from `selectors.json.image_gen.aspect_ratio_selector`. Defaults are usually fine for reference assets.
6. Click **Create**.
7. Wait for generation to finish:
   - Look for a result card / image to appear.
   - No hard timeout; poll every 3s. Typical wait: 10–30s.
8. Download the image:
   - Right-click the result image → "Save image as" is unreliable.
   - Prefer: locate the `<img>` src, fetch it with the same cookies, save to `assets/<asset_id>.png`.
   - If a download button exists, click it.
9. Mark `progress.json` asset as `done` with the saved path.

## Video generation flow (for each shot)

1. Navigate to `https://hailuoai.video/create/image-to-video`.
2. Click the "Create Video" tab if not active.
3. Click **Start Frame** upload and upload the shot's first-frame asset (`assets/<first_frame_asset_id>.png`).
4. Click the prompt input and paste the shot's 16-section prompt.
5. Verify settings:
   - Model: default "Hailuo 2.3" is fine unless the shot specifies otherwise.
   - Duration: defaults to 6s; adjust if the shot requires a different duration (free tier may only offer 6s).
   - Resolution: default 768p is fine.
6. Click **Create**.
7. Wait for generation:
   - Poll every 5s. Free-tier generations can take 30–120s.
   - Look for status text like "Generating", progress bars, or a completed video card.
8. Download the result:
   - Locate the `<video>` element src or a download button.
   - Fetch the video with the same cookies and save to `clips/<shot_id>.mp4`.
9. Mark `progress.json` shot as `done` with the saved path.

## Error handling

- **Selector not found** → halt and say: "Hailuo UI may have changed. Please update `scripts/selectors.json` for: <element>."
- **NSFW filter / policy error** → log shot/asset as `failed` with the error text, continue to the next item.
- **Rate limit / insufficient credits** → log as `failed`, report at end; do not retry automatically.
- **Generation timeout** (>10 min) → mark as `failed` with `error: timeout`.
- **Browser session lost** → re-open `https://hailuoai.video/`, re-check sign-in, resume from `progress.json`.

## Download helper strategy

Because Hailuo does not expose a stable public URL for every result, prefer extracting the media src from the DOM:

- Images: `document.querySelector('img[alt*="AI Image"], img[draggable="true"]').src`
- Videos: `document.querySelector('video source, video').src`

If the src is a blob URL, use the network panel (DevTools MCP `list_network_requests` / `get_network_request`) to capture the underlying MP4/PNG request and download that instead.

## Resume behavior

Always load `progress.json` before starting. Skip items with `status: done`. Include `status: failed` only when the user passes `--retry-failed`.
