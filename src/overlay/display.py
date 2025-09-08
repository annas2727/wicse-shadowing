#https://medium.com/@skash03/getting-started-with-desktop-overlays-with-python-and-tkinter-bfa92a23cf0 
#getting started with python and tkinter

import tkinter as tk

class Overlay(tk.Tk):
    def __init__(self, *a, **kw):
        tk.Tk.__init__(self, *a, **kw)
        self._set_window_attributes()
        self.set_window_transparency()

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
            self.canvas.create_oval(0, 0, 400, 400, fill="green", outline="green")
            self.wm_attributes("-transparentcolor", "black")


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
        

    def run(self):
        self.mainloop()

#driver code
if __name__ == "__main__":
    app = Overlay()
    app.run()