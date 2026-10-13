# Playing the real game

CrownZero's network plays the real Clash Royale through
[Build-A-Bot](https://github.com/Pbatch/ClashRoyaleBuildABot) (MIT): Build-A-Bot connects
to an Android emulator over ADB, reads each screenshot (units, hand, elixir, tower health)
and taps the screen. CrownZero replaces only the decision-making.

> **Fair play.** Supercell's terms ban automating the game, and accounts can be banned.
> Use **Training Camp** or friendly battles on a **throwaway account** only. Never ladder.

The screen reader expects a **720 x 1280 portrait** screen at **240 dpi**, in **English**.

## 1. Emulator

### macOS (Apple Silicon): BlueStacks Air

Settings > Display: portrait, 720 x 1280 if offered, 240 DPI. Turn on **Android Debug
Bridge** (Settings > Advanced or Preferences, depending on the version) and note the
address it shows, usually `127.0.0.1:5555`; pass it to the bot with
`--device-serial 127.0.0.1:5555` (the bot runs `adb connect` for you). Some BlueStacks Air
versions have no ADB setting; without it the bot cannot see the screen, so use the
Android Studio emulator below instead.

### macOS (Apple Silicon): Android Studio's emulator

1. Install [Android Studio](https://developer.android.com/studio) (Apple Silicon download).
2. More Actions > **Virtual Device Manager** > **+** > **New Hardware Profile**:
   resolution **720 x 1280**, screen size **6.1"** (that works out to 240 dpi), portrait.
3. System image: one whose Target says **Google Play** (arm64; Android 13 or 14 is fine).
4. Start it, sign in to the Play Store with a **throwaway** Google account, install Clash
   Royale, finish the tutorial, and set the game language to English.

Its ADB name is `emulator-5554`, which is what Build-A-Bot looks for by default.

### Windows: BlueStacks 5

Settings > Display: Portrait, 720 x 1280, 240 DPI. Settings > Advanced: turn on Android
Debug Bridge. Settings > Preferences: no ads during gameplay. Then install Clash Royale as
above. Pass the serial BlueStacks shows (e.g. `--device-serial 127.0.0.1:5555`).

## 2. Install

Unzip `crownzero.zip` into the same folder you clone Build-A-Bot into, so the two folders
`crownzero` and `ClashRoyaleBuildABot` sit side by side. Then, on macOS (Terminal;
[Homebrew](https://brew.sh) needed for the first line):

```bash
brew install git uv
git clone https://github.com/Pbatch/ClashRoyaleBuildABot.git
cd ClashRoyaleBuildABot
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra cpu
uv pip install -e "../crownzero[search]"   # wherever you put the CrownZero folder
```

Windows (PowerShell): the same, with `.venv\Scripts\Activate.ps1` to activate and
`..\crownzero` as the path.

Start the bot with plain `python` as below, not `uv run` or `uv sync`: those re-sync the
environment and uninstall CrownZero (re-run the last line if that happens). To swap in a
newer model later, just replace `crownzero/models/student.pt`.

## 3. Set your deck in the game

The network knows the 18 cards the simulator models accurately. Make one of these your
active battle deck in the game:

| `--deck` | Cards |
|---|---|
| `hog` | Hog Rider, Valkyrie, Musketeer, Skeleton Army, Tesla, The Log, Fireball, Baby Dragon |
| `giant` | Giant, Musketeer, Minions, Knight, Archers, Fireball, Zap, Cannon |
| `prince` | Prince, Giant, Wizard, Minions, Archers, Zap, Arrows, Knight |
| `bait` | Goblin Barrel, Knight, Archers, Minions, Skeleton Army, Tesla, The Log, Fireball |

Or pass any 8 of the 18 cards: `--deck HogRider,Valkyrie,...`.

## 4. Pick a brain

- `--brain search` (default, stronger): every move, the bot rebuilds the board it sees
  inside the simulator and tries its options there, like a chess engine. About 0.1-0.2 s
  per move. Needs the `[search]` install above.
- `--brain network`: the trained network alone, no simulator. Weaker, lighter.

## 5. Play

With the emulator running, from **inside the ClashRoyaleBuildABot folder** (so Python
finds Build-A-Bot), with its virtual environment active:

```bash
python -m crownzero.real.play --deck hog
```

Then open Training Camp in the game and start a battle. The bot waits on menus and plays
once the battle screen is up. Each play is logged, for example:

```
play 7: HogRider at tile (14, 14) | searched in 140 ms
```

Add `--record ~/Downloads/cr_recordings` to save every frame with what the bot saw and did.

`Ctrl+C` stops it.

## Troubleshooting

- **"No connected devices found"**: the emulator must be running before you start the bot.
  Build-A-Bot downloads ADB into `clashroyalebuildabot/emulator/platform-tools`; run
  `./clashroyalebuildabot/emulator/platform-tools/adb devices` to see the serial and pass
  it with `--device-serial`.
- **It never plays**: the screen reader must recognise the battle screen and your cards,
  which depends on the 720 x 1280 / 240 dpi / English setup. To see what it detects, set
  `show_images: True` in `clashroyalebuildabot/config.yaml` and run Build-A-Bot's own
  `python main.py` once.
- **Plays look odd**: the screen reader sometimes misses or misreads units. The network was
  trained with that kind of noise added, but not every case. Card levels in your account
  also differ from the simulator's.

## What to expect

The network was trained entirely in a simulator calibrated against real battles, with
noise added to mimic the screen reader. Training Camp is the fairest first test: its
opponent is the game's built-in AI.
