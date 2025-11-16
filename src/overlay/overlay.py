#https://medium.com/@skash03/getting-started-with-desktop-overlays-with-python-and-tkinter-bfa92a23cf0 
#getting started with python and tkinter

import tkinter as tk
import math
import os
import json
import random
import time

JSON_PATH = "latest_direction.json"  # same location capture.py writes to

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        if not raw or raw.strip() == "":
            return None

        raw = raw.strip()
        if not raw.startswith("{") or not raw.endswith("}"):
            return None

        return json.loads(raw)

    except:
        return None
    
class Overlay(tk.Tk):
    MAX_PARTICLES = 40   # adjust
    PARTICLE_SIZE = (2, 6)
    FADE_RATE = 0.2

    def __init__(self, *a, **kw):
        tk.Tk.__init__(self, *a, **kw)
        super().__init__(*a, **kw)
        self._setup_window()
        self._create_ui_elements()

        self.active_particles = []
        self.last_update_time = time.time()
        self.update_overlay()

    def _setup_window(self):
        self.geometry("400x400+100+100")
        self.wm_attributes("-topmost", True)
        self.overrideredirect(True)
        self.wm_attributes("-transparentcolor", "black")

        self.bind("<Button-1>", self.on_mouse_click)
        self.bind("<B1-Motion>", self.on_mouse_drag)

    def _create_ui_elements(self):
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_oval(0, 0, 400, 400, outline="#00ff00", width=2)

    #indicates where mouse is in relation to window
    def on_mouse_click(self, event): 
        self.x_offset = self.winfo_pointerx() - self.winfo_rootx()
        self.y_offset = self.winfo_pointery() - self.winfo_rooty()

    def on_mouse_drag(self, event):
        if None not in (self.x_offset, self.y_offset):
            x_window = self.winfo_pointerx() - self.x_offset
            y_window = self.winfo_pointery() - self.y_offset
            self.geometry("+%d+%d" % (x_window, y_window))


    def on_mouse_rel(self, event):
        self.x_offset = None
        self.y_offset = None
    

    def spawn_particles(self, angle_deg, intensity):
        if intensity <= 0:
            return

        count = max(5, int(intensity * self.MAX_PARTICLES))
        r = 200
        base_ang = math.radians(angle_deg)

        # smoothed angular spread
        max_spread_deg = max(10, 60 * (1 - intensity)**0.6)
        spread = math.radians(max_spread_deg)

        for _ in range(count):
            a = base_ang + random.uniform(-spread, spread)

            x = 200 + r * math.cos(a)
            y = 200 - r * math.sin(a)

            size = random.uniform(*self.PARTICLE_SIZE)

            p = {
                "x": x, "y": y,
                "size": size,
                "alpha": 1.0,
                "color": int(150 + 105 * intensity)   # brightness
            }
            self.active_particles.append(p)

    # Particle fading + draw
    def render_particles(self):
        self.canvas.delete("particle")

        new_particles = []
        for p in self.active_particles:
            # fade particle
            p["alpha"] -= self.FADE_RATE
            if p["alpha"] <= 0:
                continue  # skip dead particles

            size = p["size"]
            col = f"#{p['color']:02x}55ff"

            self.canvas.create_oval(
                p["x"] - size, p["y"] - size,
                p["x"] + size, p["y"] + size,
                fill=col, outline="", tags="particle"
            )

            new_particles.append(p)

        self.active_particles = new_particles

    def update_overlay(self):
            data = read_json(JSON_PATH)
            if data:
                angle = data.get("angle", 0)
                intensity = data.get("intensity", 0)
                self.spawn_particles(angle, intensity)

            self.render_particles()
            self.after(33, self.update_overlay)


    def run(self):
        self.mainloop()

#driver code
if __name__ == "__main__":
    app = Overlay()
    app.run()