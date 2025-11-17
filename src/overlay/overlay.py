#https://medium.com/@skash03/getting-started-with-desktop-overlays-with-python-and-tkinter-bfa92a23cf0 
#getting started with python and tkinter

import tkinter as tk
import math
import os
import json
import random
import time

JSON_PATH = "latest_direction.json"  # same location capture.py writes to

ICON_MAP = {
    "footsteps": "👣",
    "gunshot": "💥",
    "gun_handling": "🔧",
    "explosion": "💣",
    "knife": "🔪",
    "interface": "🎯",
    "background": "🌫️"
}

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
    MAX_PARTICLES = 120   # adjust
    PARTICLE_SIZE = (2, 6)
    FADE_RATE = 0.12
    ICON_FADE_RATE = 0.03
    CIRCLE_RADIUS = 250

    def __init__(self, *a, **kw):
        tk.Tk.__init__(self, *a, **kw)
        super().__init__(*a, **kw)

        self.update_idletasks()
        self.WINDOW_W = self.winfo_screenwidth()
        self.WINDOW_H = self.winfo_screenheight()
        self.CENTER = (self.WINDOW_W // 2, self.WINDOW_H // 2)

        self._setup_window()
        self._create_ui_elements()

        self.active_particles = []
        self.active_icons = []

        self.update_overlay()
        self.animate()

    def _setup_window(self):
        # full screen transparent overlay
        self.geometry(f"{self.WINDOW_W}x{self.WINDOW_H}+0+0")
        self.wm_attributes("-topmost", True)
        self.overrideredirect(True)
        self.wm_attributes("-transparentcolor", "black")

        self.bind("<Button-1>", self._click)
        self.bind("<B1-Motion>", self._drag)

    def _create_ui_elements(self):
        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        cx, cy = self.CENTER
        r = self.CIRCLE_RADIUS
        self.canvas.create_oval(
            cx - r, cy - r,
            cx + r, cy + r,
            outline="#b5b5b5", width=2
)
    #indicates where mouse is in relation to window
    def _click(self, event): 
        self.x_offset = self.winfo_pointerx() - self.winfo_rootx()
        self.y_offset = self.winfo_pointery() - self.winfo_rooty()

    def _drag(self, event):
        x = self.winfo_pointerx() - self.x_offset
        y = self.winfo_pointery() - self.y_offset
        self.geometry(f"+{x}+{y}")

    def spawn_particles(self, angle_deg, intensity):
        if intensity <= 0:
            return

        count = max(10, int(intensity * self.MAX_PARTICLES))
        base_ang = math.radians(angle_deg)
        cx, cy = self.CENTER
        R = self.CIRCLE_RADIUS

        count = max(10, int(intensity * self.MAX_PARTICLES))
        
        angle = (angle_deg + 360) % 360
        if angle >= 330 or angle <= 30:
            spread_mult = 2.2
            count_mult = 1.55
        elif 90 <= angle <= 150:
            spread_mult = 1.9
            count_mult = 1.35
        elif 210 <= angle <= 270:
            spread_mult = 1.9
            count_mult = 1.35
        else:
            spread_mult = 1.0
            count_mult = 1.0

        count = int(count * count_mult)

        max_spread_deg = max(10, 60 * (1 - intensity)**0.6)
        boosted_spread = math.radians(max_spread_deg * spread_mult)

        for _ in range(count):
            ang_offset = random.uniform(-boosted_spread, boosted_spread)
            a = base_ang + ang_offset
            dist_factor = abs(ang_offset) / boosted_spread 

            if random.random() < dist_factor:
                continue  

            radial_offset = random.uniform(-6 * spread_mult, 6 * spread_mult)
            radius = R + radial_offset

            x = cx + radius * math.cos(a)
            y = cy - radius * math.sin(a)

            # size taper
            size = random.uniform(*self.PARTICLE_SIZE) * (1 - 0.45 * dist_factor)

            # color & opacity taper 
            alpha = 1.0 * (1 - 0.35 * dist_factor)

            p = {
                "x": x,
                "y": y,
                "size": size,
                "alpha": alpha,
                "intensity": intensity,
                "dist_factor": dist_factor
            }

            self.active_particles.append(p)

    def emit_icon(self, angle_deg, label):
        if isinstance(label, list):
            label = label[0] if label else "background"

        emoji = ICON_MAP.get(label, "❓")

        angle_rad = math.radians(angle_deg)
        x = self.CENTER[0] + (self.CIRCLE_RADIUS + 18) * math.cos(angle_rad)
        y = self.CENTER[1] - (self.CIRCLE_RADIUS + 18) * math.sin(angle_rad)

        self.active_icons.append({
            "x": x, "y": y,
            "emoji": emoji,
            "alpha": 1.0
        })

    @staticmethod
    def fade_color(alpha):
        val = int(alpha * 255)
        return f"#{val:02x}{val:02x}{val:02x}"

    def animate(self):
        self.canvas.delete("particle")
        self.canvas.delete("icon")

        # animate particles
        new_particles = []
        for p in self.active_particles:
            p["alpha"] -= self.FADE_RATE
            if p["alpha"] > 0:
                col = self.intensity_to_color(p["intensity"], p["alpha"])
                self.canvas.create_oval(
                    p["x"] - p["size"], p["y"] - p["size"],
                    p["x"] + p["size"], p["y"] + p["size"],
                    fill=col, outline="", tags="particle"
                )
                new_particles.append(p)
        self.active_particles = new_particles

        # animate icons
        new_icons = []
        for icon in self.active_icons:
            icon["alpha"] -= self.ICON_FADE_RATE
            if icon["alpha"] > 0:
                fill = self.fade_color(icon["alpha"])
                self.canvas.create_text(
                    icon["x"], icon["y"],
                    text=icon["emoji"],
                    fill=fill,
                    font=("Segoe UI Emoji", 22),
                    tags="icon"
                )
                new_icons.append(icon)
        self.active_icons = new_icons
        self.after(33, self.animate)

    def update_overlay(self):
        data = read_json(JSON_PATH)
        if data:
            angle = data.get("angle", 0)
            intensity = data.get("intensity", 0)
            label = data.get("label", "background")

            if intensity > 0.05:  # adjust threshold if needed
                self.spawn_particles(angle, intensity)
                self.emit_icon(angle, label)

        self.after(70, self.update_overlay)

    def intensity_to_color(self, intensity, alpha=1.0):
        #Convert intensity (0-1) into a gradient
        intensity = max(0.0, min(1.0, intensity))

        if intensity < 0.25:  # blue to purple
            t = intensity / 0.25
            r = int(0 + 80*t)
            g = int(0)
            b = int(255 - 55*t)

        elif intensity < 0.5:  # purple to magenta
            t = (intensity - 0.25) / 0.25
            r = int(80 + 175*t)
            g = int(0)
            b = int(200 - 200*t)

        elif intensity < 0.75:  # magentato orange
            t = (intensity - 0.5) / 0.25
            r = int(255)
            g = int(0 + 140*t)
            b = int(0)

        else:  # orange to yellow/white
            t = (intensity - 0.75) / 0.25
            r = int(255)
            g = int(140 + 115*t)
            b = int(0 + 255*t)

        r = int(r * alpha)
        g = int(g * alpha)
        b = int(b * alpha)
        return f"#{r:02x}{g:02x}{b:02x}"


    def run(self):
        self.mainloop()

#driver code
if __name__ == "__main__":
    app = Overlay()
    app.run()