# Recording top-player replays (v3 training data)

v3 learns from top players instead of from the simulator. The game has no download of
move-by-move data, so we record replays and extract the moves from the video with the same
screen reader the bot plays with.

## Record one replay

1. In BlueStacks, open Clash Royale and find a replay of a top player:
   **TV Royale** (top games), or tap a top ladder player's name > battle log > a replay.
   Pick one where the player whose view you see uses mostly cards from our 18
   (Knight, Archers, Musketeer, Giant, Hog Rider, Minions, Baby Dragon, Valkyrie,
   Skeleton Army, Cannon, Tesla, Fireball, Zap, Arrows, The Log, Prince, Wizard,
   Goblin Barrel) if you can find one; any top replay is still useful for the first test.
2. Pause it at the very start if the game lets you, or just be quick.
3. In Terminal, inside the ClashRoyaleBuildABot folder with its environment active:

   ```bash
   python -m crownzero.real.capture --device-serial 127.0.0.1:5555 --name tv_royale_1
   ```

4. Let the replay play to the end, then press **Ctrl+C** once and wait for "done".

The video lands in `~/Downloads/cr_replays/tv_royale_1_<date-time>/` as `part_000.mp4`,
`part_001.mp4`, ... Recording happens inside the emulator, so moving or covering the
BlueStacks window does not matter.

## Send it

Zip that folder and drop it in the chat, or tell Claude to fetch it from your Mac.
