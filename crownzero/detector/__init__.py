"""A unit detector for the real game, trained on generated images.

Build-A-Bot's detector (units_M_480x352.onnx) misreads too much: the enemy's tower troops
as Bombers on every frame, an Elixir Golem as five different cards in one push, units
flipping sides. This package trains a replacement:

  space.py     the arena "part2" frame (568 x 896, KataCR's) <-> screenshot <-> tiles
  generate.py  training images: sprites cut from real battles (the KataCR detection
               dataset, MIT) pasted onto empty arenas, with exact labels
  train.py     YOLOv8 on them (Apple Silicon: device "mps")
  detector.py  the trained model behind Build-A-Bot's interface: (allies, enemies)

Sprite data: https://github.com/wty-yy/Clash-Royale-Detection-Dataset (MIT, wty-yy).
"""
