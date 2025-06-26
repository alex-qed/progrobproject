#!/usr/bin/env python3
"""
PTD v9  —  распознаёт жесты и выдаёт поток (pan, tilt, dim).
Используется в main.py: for pan, tilt, dim in run_gesture_loop(): …
Нажмите «q» в окне OpenCV, чтобы выйти.
"""
from __future__ import annotations
import cv2, mediapipe as mp, numpy as np, collections
from statistics import median
import math

# ─────────────────── константы фильтров ───────────────────
EMA_ALPHA = 0.25        # сглаживание
DEAD_ZONE = 2.0         # ° — мёртвая зона
MED_WIN   = 5           # окно медианы

PAN_RANGE  = (-270.0,  270.0)
TILT_RANGE = (-120.0,  120.0)

# шкала диммера (доли от высоты кадра)
TOP_FRAC, BOTTOM_FRAC = 0.35, 0.65
LINE_X1_OFF = 130      # отступ шкалы от правого края

CONTACT_THR, SPREAD_THR = 40, 60   # px

tips = [8, 12, 16, 20]             # fingertip ids
pips = [6, 10, 14, 18]             # pip-joint ids

# ─────────────────── helpers ───────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(False, 2, 1, 0.5, 0.5)

def to_np(lm, w, h):
    a = np.empty((21, 3))
    for i, p in enumerate(lm.landmark):
        a[i] = [p.x * w, p.y * h, p.z * w]
    return a

dist = lambda h: float(np.linalg.norm(h[4][:2] - h[8][:2]))
clipmap = lambda x, a, b, c, d: float(np.clip(np.interp(x, (a, b), (c, d)), c, d))

class Med:
    """скользящее окно медианы"""
    def __init__(self, n): self.buf = collections.deque(maxlen=n)
    def __call__(self, v): self.buf.append(v); return median(self.buf)

def dead(prev, new, dz): return new if abs(new - prev) >= dz else prev

# ───── 4-шаговая калибровка ─────
class Step: PAN_MIN, PAN_MAX, TILT_MIN, TILT_MAX, DONE = range(5)
PROMPT = {
    Step.PAN_MIN : "Touch LEFT thumb+index  (-270)",
    Step.PAN_MAX : "Spread LEFT thumb+index (+270) hold 2 s",
    Step.TILT_MIN: "Touch RIGHT thumb+index (-120)",
    Step.TILT_MAX: "Spread RIGHT thumb+index (+120) hold 2 s",
}
class Calib:
    HOLD = 2.0; ABS = 8; REL = 0.15
    def __init__(self, fps: float):
        self.N = int(self.HOLD * fps)
        self.buf = collections.deque(maxlen=self.N)
        self.stage = Step.PAN_MIN
        self.val = {}          # pan_min, pan_max, …
    def stable(self):
        if len(self.buf) < self.N: return False
        a = np.array(self.buf)
        return a.ptp() < self.ABS or a.ptp() < self.REL * a.mean()
    def record(self, v, cond):
        self.buf.append(v) if cond else self.buf.clear()
    def update(self, left, right):
        st = self.stage
        if st == Step.PAN_MIN and left is not None:
            d = dist(left); self.record(d, d < CONTACT_THR)
            if self.stable(): self.val['pn'] = np.mean(self.buf); self.buf.clear(); self.stage = Step.PAN_MAX
        elif st == Step.PAN_MAX and left is not None:
            d = dist(left); self.record(d, d > SPREAD_THR)
            if self.stable(): self.val['px'] = np.mean(self.buf); self.buf.clear(); self.stage = Step.TILT_MIN
        elif st == Step.TILT_MIN and right is not None:
            d = dist(right); self.record(d, d < CONTACT_THR)
            if self.stable(): self.val['tn'] = np.mean(self.buf); self.buf.clear(); self.stage = Step.TILT_MAX
        elif st == Step.TILT_MAX and right is not None:
            d = dist(right); self.record(d, d > SPREAD_THR)
            if self.stable(): self.val['tx'] = np.mean(self.buf); self.buf.clear(); self.stage = Step.DONE
        return self.stage == Step.DONE, len(self.buf) / self.N
    def pan(self, d):  return clipmap(d, self.val['pn'], self.val['px'], *PAN_RANGE)
    def tilt(self, d): return clipmap(d, self.val['tn'], self.val['tx'], *TILT_RANGE)

# ─────────── основной генератор ────────────
def run_gesture_loop():
    cap = cv2.VideoCapture(0);  assert cap.isOpened()
    W, H = int(cap.get(3)), int(cap.get(4))
    fps  = cap.get(5) or 30
    TOP, BOT = int(TOP_FRAC * H), int(BOTTOM_FRAC * H)
    MID = (TOP + BOT)//2
    X1, X2 = W - LINE_X1_OFF, W - 30

    def dashed(y):
        for x in range(X1, X2, 20):
            cv2.line(frame, (x, y), (x + 10, y), (0,165,255), 1)

    calib = Calib(fps)
    m_pan, m_tilt = Med(MED_WIN), Med(MED_WIN)
    pan = tilt = dim = 0.0

    while True:
        ok, frame = cap.read();  assert ok
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        left = right = None
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                a = to_np(lm, W, H)
                if hd.classification[0].label == 'Left':  left = a
                else:                                     right = a
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        msg = ""
        col = (0, 255, 255)
        # --- CONFIG ---
        if calib.stage != Step.DONE:
            done, pr = calib.update(left, right)

            # строка подсказки + прогресс
            #cv2.putText(frame, f"{msg} {int(pr*100):3d}%", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

            msg = PROMPT.get(calib.stage, "Calibration done")
            col = (0,255,0) if pr else (0,255,255)
            cv2.putText(frame, f"{msg} {int(pr*100):3d}%", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        else:
            # ------------- MOVE -----------------------
            if left is not None:
                pan = dead(pan, m_pan(calib.pan(dist(left))), DEAD_ZONE)
                cv2.line(frame, tuple(left[4][:2].astype(int)),
                                tuple(left[8][:2].astype(int)), (0,255,255),2)
            if right is not None:
                tilt = dead(tilt, m_tilt(calib.tilt(dist(right))), DEAD_ZONE)
                cv2.line(frame, tuple(right[4][:2].astype(int)),
                                tuple(right[8][:2].astype(int)), (255,0,255),2)
                y = right[4][1]
                dim = 0 if y>=BOT else 100 if y<=TOP else (BOT-y)/(BOT-TOP)*100

        # --- рисуем шкалу и HUD (общая часть) --------
        for y in (TOP, MID, BOT): dashed(y)
        cv2.putText(frame,"dim",(X1-35,TOP+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

        cv2.putText(frame,f"Pan  {pan:+07.1f}", (10, H-60),
            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame,f"Tilt {tilt:+07.1f}", (10, H-35),
            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame,f"Dim  {dim:07.1f}%", (10, H-10),
            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("Gesture→PTD", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
        yield pan, tilt, dim

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    for _ in run_gesture_loop(): pass
