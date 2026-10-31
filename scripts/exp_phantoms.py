"""The eighth recording's misread, reproduced: Hog 2.6 (the real-game brain, the client's
delay, a view one step old) vs the scripted defender, with or without the two phantom
Bombers the screen reader saw on the enemy's princess towers.

    python scripts/exp_phantoms.py phantom|clean [games]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.arena import make_env, make_bot, TEAM, AGENTS
from crownzero.live_search import LiveSearchBot
from crownzero.real.play import REAL_GAME_SEARCH
HOG=("HogRider","Skeletons","IceSpirits","Fireball","Cannon","Log","IceGolemite","Musketeer")
OPP=("Knight","Archer","Giant","Minions","Fireball","Tesla","Zap","Musketeer")
phantom = sys.argv[1]=="phantom"
games=int(sys.argv[2]) if len(sys.argv) > 2 else 8
res=[]; plays={}
for seed in range(games):
    env=make_env([list(HOG), list(OPP)], delay=21)
    kw=dict(REAL_GAME_SEARCH, stale_steps=1)
    if phantom: kw["phantoms"]=[("Bomber",2,24),("Bomber",13,24)]
    me=LiveSearchBot(kw); opp=make_bot("defend")
    obs,_=env.reset(seed=seed); rng=np.random.default_rng(seed)
    while env.agents:
        acts={"blue":int(me.act(env,0,obs["blue"],obs["blue"]["action_mask"],rng)),
              "red":int(opp.act(env,1,obs["red"],obs["red"]["action_mask"],rng))}
        acts={a:acts[a] for a in env.agents}
        obs,*_=env.step(acts)
    s=env.battle_state
    for k,v in me.plays.items(): plays[k]=plays.get(k,0)+v
    res.append((int(s.winner), [p.crowns for p in s.players]))
print(sys.argv[1], "wins", sum(w==0 for w,_ in res), "/", games, "crowns", res, "plays", plays, flush=True)
