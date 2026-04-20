# PylaAI — notes for Claude

External Brawl Stars bot. Reads the game via ADB screenshots from an Android emulator, runs ONNX vision models (aim, walls, tiles, UI states), and drives input back through ADB. The bot is fully external — it does not modify the game client.

## Entry point and flow

[main.py](main.py) bootstraps logging, builds `App` (Tkinter GUI in [gui/](gui/)), and on user start calls `pyla_main(data)`. That function defines and runs `Main`, whose `main()` is the hot loop:

1. `WindowController.screenshot()` — pulls a frame over ADB ([window_controller.py](window_controller.py))
2. `get_state(frame)` — classifies the screen (lobby, match, result, ...) via template matching + models ([state_finder.py](state_finder.py))
3. `StageManager.do_state(state, ...)` — dispatches per-state behavior ([stage_manager.py](stage_manager.py))
4. `Play.main(frame, brawler, self)` — in-match movement/aim/supers ([play.py](play.py))
5. `TrophyObserver` tracks wins/losses/trophies, including the showdown trio place-based trophies added recently ([trophy_observer.py](trophy_observer.py))
6. `LobbyAutomation` handles menu navigation and brawler selection ([lobby_automation.py](lobby_automation.py))

Support modules: [time_management.py](time_management.py) (periodic checks / idle / stale frames), [detect.py](detect.py) (ONNX inference helpers), [utils.py](utils.py) (config loading, API client, notifications).

## Config

All runtime config lives in [cfg/](cfg/):

- `general_config.toml` — `pyla_version`, `max_ips`, `run_for_minutes`, `visual_debug`, `super_debug`
- `bot_config.toml` — per-brawler behavior
- `time_tresholds.toml` — timing thresholds (**note the spelling: `tresholds`, not `thresholds`** — it's baked into multiple readers; do not rename casually)
- `match_history.toml` — mutated by the bot at runtime
- `brawlers_info.json` — brawler metadata, synced from the API when online
- `lobby_config.toml` — template-matching regions for lobby screens
- `login.toml` — local credentials for the online mode

Several of these files are modified during normal runs (see `git status`). Don't bundle those churn edits into unrelated commits.

## Models

[models/](models/) holds the `.onnx` files actually used: `mainInGameModel.onnx`, `tileDetector.onnx`, `brawlersInGame.onnx`, `startingScreenModel.onnx`. The `.pt` sources live in a separate repo (see README). Wall model is auto-updated from the API in online mode via `get_latest_wall_model_file()` in [utils.py](utils.py).

## Run / test

- Python 3.11.9, Windows. ADB ships in the repo (`adb.exe`, `AdbWinApi.dll`, `AdbWinUsbApi.dll`).
- Install: `python setup.py install`
- Run: `python main.py`
- Tests: `python -m unittest discover` (see [tests/](tests/))
- `api_base_url` in [utils.py](utils.py) switches between `localhost` (this repo, offline) and the online backend. In localhost mode, `update_missing_brawlers_info`, `check_version`, and wall-model auto-update are skipped.

## Non-obvious gotchas

- **Import order in [main.py](main.py) matters.** `setup_logging()` is called before any other project imports so that module-level logger calls are captured. Don't reorder.
- **`Main` is defined inside `pyla_main`** — it closes over `data` (the user's picks from the GUI). Keep it nested unless you also refactor the data passing.
- **Template matching scales from a 1920×1080 reference** (`orig_screen_width/height` in [state_finder.py](state_finder.py)). Any new region coordinates must be given in 1920×1080 space.
- **`time_tresholds.toml`** — keep the misspelling.
- **Stale-frame recovery** — if `get_latest_frame()` hasn't updated for `FRAME_STALE_TIMEOUT` seconds, the bot restarts the game. When debugging loops that seem to restart randomly, check ADB/emulator frame delivery before the bot logic.
- **`run_for_minutes` + cooldown** — after the timer expires, the bot stays alive for a 3-minute cooldown to finish the current match. `Stage_manager.states['lobby']` is replaced with a no-op during cooldown so it won't queue a new game.

## What not to do

- Don't add online-API features here — the site backend is not open source (README).
- Don't commit churn to `cfg/*.toml`, `logs/`, `latest_brawler_data.json`, or `images/` unless that's the actual task.
- Don't propose changes that would modify the game client — this project is strictly external.
- Respect the "no selling" license.
