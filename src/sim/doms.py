#!/usr/bin/env python3
"""
doms.py — Simplified unified script combining PID, Spot, Core, and WorldProc with trajectory support.
"""

import math
import argparse
import time
import csv
import numpy as np
import mujoco as mj
from mujoco.viewer import launch_passive
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
#   Geometry / Angle Math
# =============================================================================

def wrap_deg(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle to (-180, 180] degrees."""
    return (angle + 180.0) % 360.0 - 180.0


def wrap_rad(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle to (-pi, pi] radians."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def deg_ang_diff(a: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """Difference between two angles in degrees."""
    return wrap_deg(a - b)


def geo_to_angles(target_xyz: np.ndarray, origin_xyz: np.ndarray) -> tuple[float, float]:
    """Return required (pan, tilt) in degrees for vector origin->target."""
    vec = np.asarray(target_xyz, float) - np.asarray(origin_xyz, float)
    x, y, z = vec
    pan = wrap_deg(math.degrees(math.atan2(x, y)) + 90.0)
    horiz = math.hypot(x, y)
    #tilt = math.degrees(math.atan2(z, horiz))
    tilt = wrap_deg(math.degrees(math.atan2(z, math.hypot(x, y))) + 90.0)
    return pan, tilt

def trajectory_circle(t: float) -> np.ndarray:
    """
    Круг радиусом 4:
      x = cos(t)
      y = sin(t)
      z = 1
    """
    return np.array([4.0 * math.cos(t),
                     4.0 * math.sin(t),
                     1.0])

def trajectory_ellipse(t: float) -> np.ndarray:
    """
    Эллипс с полуосями 2 и 1:
      x = 2*cos(t)
      y = 1*sin(t)
      z = 1
    """
    return np.array([4.0 * math.cos(t),
                     2.0 * math.sin(t),
                     1.0])

def trajectory_square(t: float) -> np.ndarray:
    """
    Параметрическая «квадратная» траектория
    через нормировку (cos, sin):
      u = cos(t), v = sin(t)
      max(|u|,|v|) == 1 → (u/max, v/max)
      z = 1
    """
    u = math.cos(t)
    v = math.sin(t)
    m = max(abs(u), abs(v)) or 1.0
    return np.array([4.0 * u/m,
                     4.0 * v/m,
                     1.0])

# Собираем всё в один словарь
TRAJECTORIES: dict[str, callable] = {
    "circle":  trajectory_circle,
    "ellipse": trajectory_ellipse,
    "square":  trajectory_square,
}

# =============================================================================
#   PID Controller
# =============================================================================

class PID:
    """Simple PID controller for 2 channels (pan, tilt)."""
    def __init__(
        self,
        Kp: tuple[float, float] = (5, 5),
        Ki: tuple[float, float] = (1, 1),
        Kd: tuple[float, float] = (0.1, 0.1),
    ):
        self.Kp = np.array(Kp, float)
        self.Ki = np.array(Ki, float)
        self.Kd = np.array(Kd, float)
        self.prev_error = np.zeros(2)
        self.integral = np.zeros(2)

    def reset(self) -> None:
        self.prev_error.fill(0.0)
        self.integral.fill(0.0)

    def update(self, ref: np.ndarray, act: np.ndarray, dt: float) -> np.ndarray:
        error = np.array([wrap_rad(ref[i] - act[i]) for i in range(2)])
        self.integral += error * dt

        diff       = np.array([wrap_rad(error[i]-self.prev_error[i])
                             for i in range(2)])  
        derivative = diff / dt

        self.prev_error = error.copy()

        u = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        return np.clip(u, -1.5, 1.5)

# =============================================================================
#   Spotlight Interface
# =============================================================================

class Spot:
    """Handles reading and writing spotlight pan/tilt in MuJoCo."""
    def __init__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        pan_joint: str = "pan",
        tilt_joint: str = "tilt",
        tilt_offset_deg: float = 0.0,
        origin_xyz: tuple[float, float, float] | None = None,
    ):
        self.model = model
        self.data = data
        self.TILT_OFFSET_DEG = tilt_offset_deg

        # Joint and actuator IDs
        self.jid_pan = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, pan_joint)
        self.jid_tilt = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, tilt_joint)
        self.act_pan  = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, pan_joint)
        self.act_tilt = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, tilt_joint)

        # Origin: either provided or from 'head_site'
        if origin_xyz is not None:
            self.origin = np.asarray(origin_xyz, float)
        else:
            sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "head_site")
            self.origin = self.data.site_xpos[sid].copy()

    def get_angles(self, degrees: bool = False) -> np.ndarray:
        pan_rad = self.data.qpos[self.jid_pan]
        tilt_rad = self.data.qpos[self.jid_tilt] - math.radians(self.TILT_OFFSET_DEG)
        if degrees:
            return np.array([math.degrees(pan_rad), math.degrees(tilt_rad)])
        return np.array([pan_rad, tilt_rad])

    def set_direct(self, pan_deg: float, tilt_deg: float) -> None:
        pan_rad  = math.radians((pan_deg + 180.0) % 360.0 - 180.0)
        tilt_rad = math.radians(tilt_deg + self.TILT_OFFSET_DEG)
        self.data.qpos[self.jid_pan]  = pan_rad
        self.data.qpos[self.jid_tilt] = tilt_rad

    def apply_control(self, u: np.ndarray) -> None:
        # Pan: direct control
        self.data.ctrl[self.act_pan]  = float(u[0])
        # Tilt: invert so positive u[1] lowers the light
        self.data.ctrl[self.act_tilt] = float(u[1])

# =============================================================================
#   Core Computations and Logging
# =============================================================================

class Core:
    """Computations: reference, error, constraints; logging to CSV."""
    TILT_MIN_DEG: float = -135.0
    TILT_MAX_DEG: float =  135.0
    PAN_MIN_DEG: float = -270.0
    PAN_MAX_DEG: float =  270.0
    INIT_TIME:    float =   1.0  # seconds

    def __init__(self,
        log_path: str = "debug.csv",
        log_interval: float = 0.01,
    ):
        self.log_path = log_path
        self.log_interval = log_interval
        self._next_log_time = 0.0
        # Заголовок с двумя ошибками
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time", "pan_ref_deg", "tilt_ref_deg",
                "pan_act_deg", "tilt_act_deg",
                "pan_err_deg", "tilt_err_deg",
                "ctrl_pan", "ctrl_tilt",
            ])

    def compute_reference_angles(
        self,
        target_xyz: np.ndarray,
        origin_xyz: np.ndarray,
    ) -> np.ndarray:
        pan_deg, tilt_deg = geo_to_angles(target_xyz, origin_xyz)
        tilt_deg = np.clip(tilt_deg, self.TILT_MIN_DEG, self.TILT_MAX_DEG)
        # возвращаем в радианах
        return np.deg2rad([pan_deg, tilt_deg])

    def log_step(
        self,
        t: float,
        ref: np.ndarray,
        act: np.ndarray,
        ctrl: np.ndarray,
    ) -> None:
        if t < self._next_log_time:
            return
        self._next_log_time = t + self.log_interval
        # Переводим в градусы для логирования
        ref_deg = np.degrees(ref)
        act_deg = np.degrees(act)
        # Ошибки в градусах по компонентам
        err_deg = wrap_deg(ref_deg - act_deg)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round(t,    4),
                round(ref_deg[0], 2), round(ref_deg[1], 2),
                round(act_deg[0], 2), round(act_deg[1], 2),
                round(err_deg[0], 2), round(err_deg[1], 2),
                round(ctrl[0], 4), round(ctrl[1], 4),
            ])

    def compute_error_angle(
        self,
        ref: np.ndarray,
        act: np.ndarray,
    ) -> float:
        v_ref = self._angles_to_vector(ref)
        v_act = self._angles_to_vector(act)
        dot   = np.clip(float(np.dot(v_ref, v_act)), -1.0, 1.0)
        return math.acos(dot)

    @staticmethod
    def _angles_to_vector(angles: np.ndarray) -> np.ndarray:
        pan, tilt = angles
        cos_t = math.cos(tilt)
        v = np.array([
            math.sin(pan) * cos_t,
            math.cos(pan) * cos_t,
            math.sin(tilt),
        ])
        return v / np.linalg.norm(v)
    
    @staticmethod
    def plot_logs(csv_path: str = "debug.csv") -> None:
        """
        Читает лог из `csv_path` и выводит 4 графика:
        1) ошибка pan/tilt; 2) ref vs act; 3) управление; 4) траектория.
        """
        df = pd.read_csv(csv_path)

        # --- 1. ошибка --------------------------------------------------
        plt.figure("error")
        plt.title("Позиционная ошибка (°)")
        plt.plot(df["time"], df["pan_err_deg"],  label="pan_err")
        plt.plot(df["time"], df["tilt_err_deg"], label="tilt_err")
        plt.ylabel("угол, °");  plt.xlabel("t, c");  plt.legend();  plt.grid()

        # --- 2. ref / act ----------------------------------------------
        plt.figure("ref_vs_act")
        plt.title("Reference vs Actual (°)")
        plt.plot(df["time"], df["pan_ref_deg"],   "--", label="pan_ref")
        plt.plot(df["time"], df["pan_act_deg"],         label="pan_act")
        plt.plot(df["time"], df["tilt_ref_deg"],  "--", label="tilt_ref")
        plt.plot(df["time"], df["tilt_act_deg"],        label="tilt_act")
        plt.ylabel("угол, °");  plt.xlabel("t, c");  plt.legend();  plt.grid()

        # --- 3. управление ---------------------------------------------
        plt.figure("control")
        plt.title("Control effort")
        plt.plot(df["time"], df["ctrl_pan"],  label="u_pan")
        plt.plot(df["time"], df["ctrl_tilt"], label="u_tilt")
        plt.ylabel("момент, Н·м");  plt.xlabel("t, c");  plt.legend();  plt.grid()

        # --- 4. траектория актёра --------------------------------------
        # if {"target_x", "target_y"}.issubset(df.columns):
        #     plt.figure("trajectory")
        #     plt.title("Траектория актёра")
        #     plt.plot(df["target_x"], df["target_y"], "-o", markersize=2)
        #     plt.axis("equal");  plt.xlabel("X");  plt.ylabel("Y");  plt.grid()
        # else:
        #     print("⚠ В логе нет колонок target_x / target_y – траекторию пропускаю")

        plt.show()


# =============================================================================
#   World Processing with Trajectory
# =============================================================================

class WorldProc:
    """Loads MuJoCo world, steps simulation, and updates target trajectory."""
    def __init__(
        self,
        xml_path: str = "world2.xml",
        headless: bool = False,
    ):
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data  = mj.MjData(self.model)
        self.torso_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "torso")
        mj.mj_forward(self.model, self.data)
        # Find the mocap id for the torso, if it exists
        self.torso_bid      = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "torso")
        self.torso_mocap_i  = self.model.body_mocapid[self.torso_bid]
        if self.torso_mocap_i < 0:
            raise RuntimeError("Body 'torso' is not mocap-enabled in the model. Please check your XML model.")
        

        self.viewer = None
        if not headless:
            self.viewer = launch_passive(self.model, self.data)
        # Site ID for target
        self.target_sid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SITE, "target_site"
        )
        # Default static trajectory
        self.trajectory = lambda t: self.data.site_xpos[self.target_sid].copy()

    def set_trajectories(self, name):
        if name == "circle": 
            self.trajectory = trajectory_circle
        elif name == 'ellipse':
            self.trajectory = trajectory_ellipse
        elif name == 'square':
            self.trajectory = trajectory_square

    def update_trajectory(self):
        t = self.data.time
        new_pos = np.asarray(self.trajectory(t), float)
        # обновляем позицию torso-мокапа
        self.data.mocap_pos[self.torso_mocap_i] = new_pos
        # и сразу пересчитываем киноматику, чтобы viewer видел новое место
        mj.mj_forward(self.model, self.data)

    def get_target_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.target_sid].copy()

    def step(self) -> None:
        mj.mj_step(self.model, self.data)
        if self.viewer:
            self.viewer.sync()

# =============================================================================
#   Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MuJoCo Spotlight PID Demo")
    parser.add_argument(
        "--mode", choices=("direct", "pid"),
        default="direct", help="Control mode"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without visualization"
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="Simulation duration in seconds"
    )
    args = parser.parse_args()

    world = WorldProc(headless=args.headless)
    # Example: circular trajectory radius 1m at 0.5Hz
    # world.set_trajectory(lambda t: np.array([
    #     math.cos(2*math.pi*0.1*t),
    #     math.sin(2*math.pi*0.1*t),
    #     1.0
    # ]))
    world.set_trajectories("square") #, "circle", "ellipse", "square"

    spot  = Spot(world.model, world.data)
    pid   = PID()
    core  = Core()

    sim_start = world.data.time
    last_time = sim_start

    try:
        while True:
            t = world.data.time - sim_start
            if args.duration and t >= args.duration:
                break

            # Update target trajectory
            world.update_trajectory()

            # Compute reference from updated target
            actor_pos = world.get_target_pos()
            sid = mj.mj_name2id(world.model, mj.mjtObj.mjOBJ_SITE, "head_site")
            origin = spot.data.site_xpos[sid].copy()
            ref = core.compute_reference_angles(actor_pos, origin)
            

            # Actual angles
            act = spot.get_angles()

            # Control logic
            if t <= Core.INIT_TIME:
                spot.set_direct(0.0, 0.0)
                ctrl = np.array([0.0, 0.0])
                world.data.qvel[spot.jid_pan]  = 0.0
                world.data.qvel[spot.jid_tilt] = 0.0
                pid.reset()
            else:
                if args.mode == "direct":
                    pan_deg  = math.degrees(ref[0])
                    tilt_deg = math.degrees(ref[1])
                    spot.set_direct(pan_deg, tilt_deg)
                    ctrl = np.array([0.0, 0.0])
                else:
                    dt   = world.model.opt.timestep
                    ctrl = pid.update(ref, act, dt)
                    spot.apply_control(ctrl)

            # Step simulation and visualize
            world.step()

            # Logging
            act = spot.get_angles()
            core.log_step(t, ref, act, ctrl)

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        Core.plot_logs("debug.csv")

if __name__ == "__main__":
    main()
