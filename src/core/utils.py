"""Разные геометрические функции и траектории."""
import math, numpy as np

wrap_deg = lambda a: (a+180.)%360.-180.
wrap_rad = lambda a: (a+math.pi)%(2*math.pi)-math.pi
deg_diff = lambda a,b: wrap_deg(a-b)

def geo_to_angles(target, origin):
    """Возвращает (pan, tilt) в ° для вектора origin→target."""
    x,y,z = np.asarray(target) - np.asarray(origin)
    pan  = wrap_deg(math.degrees(math.atan2(x, y)) + 90)
    tilt = wrap_deg(math.degrees(math.atan2(z, math.hypot(x,y))) + 90)
    return pan, tilt

# --- простые траектории -----------------------------------
def circle(t):  return np.array([4*math.cos(t), 4*math.sin(t), 1])
def ellipse(t): return np.array([4*math.cos(t), 2*math.sin(t), 1])
def square(t):
    u,v=math.cos(t), math.sin(t); m=max(abs(u),abs(v)) or 1
    return np.array([4*u/m, 4*v/m, 1])
TRAJ = {"circle":circle, "ellipse":ellipse, "square":square}
