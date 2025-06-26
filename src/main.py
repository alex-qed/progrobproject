#!/usr/bin/env python3
"""
Запуск: жесты + MuJoCo.
CLI:
  --xml world_spot.xml  --rmin 0.01 --rmax 0.08
  --slew 60            # °/s
  --headless
"""
import argparse, threading, time
from pathlib import Path
import mujoco as mj
from mujoco.viewer import launch_passive
from src.sim import Spot
from src.gesture import run_gesture_loop

def mujoco_worker(cfg, shared):
    model = mj.MjModel.from_xml_path(str(cfg.xml))
    data  = mj.MjData(model)
    viewer = None if cfg.headless else launch_passive(model, data)

    spot = Spot(model, data)
    beam_gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "spot_beam")
    cur_pan = cur_tilt = 0.0
    max_step = cfg.slew * model.opt.timestep   # ° на шаг

    def slew(cur, tgt):
        delta = max(-max_step, min(max_step, tgt-cur))
        return cur + delta

    while True:
        with shared["lock"]:
            tgt_pan, tgt_tilt, dim = shared["pan"], shared["tilt"], shared["dim"]

        cur_pan  = slew(cur_pan,  tgt_pan)
        cur_tilt = slew(cur_tilt, tgt_tilt)
        spot.set_direct(cur_pan, cur_tilt)

        radius = cfg.rmin + (cfg.rmax-cfg.rmin)*(dim/100)
        model.geom_size[beam_gid, 0] = radius

        mj.mj_step(model, data)
        if viewer: viewer.sync()

def main():
    pa = argparse.ArgumentParser("gesture-spot")
    pa.add_argument("--xml", type=Path, default=Path(__file__).parent / "sim" / "world_spot.xml")
    pa.add_argument("--rmin", type=float, default=0.01)
    pa.add_argument("--rmax", type=float, default=0.08)
    pa.add_argument("--slew", type=float, default=90.0, help="макс. скорость °/с")
    pa.add_argument("--headless", action="store_true")
    cfg = pa.parse_args()

    shared = {"pan":0., "tilt":0., "dim":0., "lock":threading.Lock()}
    th = threading.Thread(target=mujoco_worker, args=(cfg, shared), daemon=True)
    th.start()

    for pan, tilt, dim in run_gesture_loop():
        with shared["lock"]:
            shared["pan"], shared["tilt"], shared["dim"] = pan, tilt, dim

if __name__ == "__main__":
    main()
