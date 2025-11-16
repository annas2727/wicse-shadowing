#https://medium.com/@skash03/getting-started-with-desktop-overlays-with-python-and-tkinter-bfa92a23cf0 
#getting started with python and tkinter

import tkinter as tk
import math
import os
import json

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
    def __init__(self, *a, **kw):
        tk.Tk.__init__(self, *a, **kw)
        super().__init__(*a, **kw)
        self._set_window_attributes()
        self.set_window_transparency()
        self._create_ui_elements()

        self.update_overlay()

    def _set_window_attributes(self):
        self.title("Overlay")
        self.geometry("400x400+100+100")
        self.focus_force() # focus on window
        self.wm_attributes("-topmost", True) #force window to be on top at all times
        self.overrideredirect(True) #remove borders to prevent resizing
   
        #window variables
        self.offset_x, self.offset_y = None, None

        #bind the functions
        self.bind("<Button-1>", self.on_mouse_click)
        self.bind("<B1-Motion>", self.on_mouse_drag)
        self.bind("<ButtonRelease-1>", self.on_mouse_rel)

    def set_window_transparency(self):
        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(side = "top", fill = "both", expand = True)
        self.canvas.config(highlightthickness = 0)
        self.wm_attributes("-transparentcolor", "black")

    def _create_ui_elements(self):
        self.canvas.create_oval(0, 0, 400, 400, fill="green", outline="green") # test oval 
        self.dot = self.canvas.create_oval(195, 195, 205, 205, fill="red", outline="red")
    
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
        
    def draw_dot(self, angle_deg, intensity):
        #angle_deg: 0-360 degrees
        #intensity: 0-1 (distance from center)

        radius = 200  # circle radius
        inner_radius = radius * intensity
        angle_rad = math.radians(angle_deg)

        # Circle center and dot position
        cx, cy = 200, 200
        x = cx + inner_radius * math.cos(angle_rad)
        y = cy - inner_radius * math.sin(angle_rad)

        # Update dot on canvas
        self.canvas.coords(self.dot, x - 5, y - 5, x + 5, y + 5)

    def update_overlay(self):
        data = read_json(JSON_PATH)

        if data is not None: 
            angle = data.get("angle", 0)
            intensity = data.get("intensity", 0)
            self.draw_dot(angle, intensity)
            
        self.after(50, self.update_overlay)


    def run(self):
        self.mainloop()

#driver code
if __name__ == "__main__":
    app = Overlay()
    app.run()