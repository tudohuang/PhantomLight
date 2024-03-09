from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = 0
        self.y = 0
        self.count = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Retrieving...", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Retrieve", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Dropdown menu for selecting spells
        self.spell_var = tk.StringVar(self)
        self.spell_var.set("Incendio")
        self.spell_menu = tk.OptionMenu(self, self.spell_var, "Incendio", "Aqua", "Arresto", "Alohomora", "Lumos")

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=0, pady=2, padx=2)
        self.button_clear.grid(row=1, column=1, pady=2)
        self.spell_menu.grid(row=2, column=0, pady=2, padx=2)

        # Event bindings
        self.canvas.bind("<Button-1>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")
        self.count = 0

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWND)
        self.count +=  1
        offset = 0
        rect1 = (rect[0]+offset, rect[1]+offset, rect[2]+offset, rect[3]+offset)
        im = ImageGrab.grab(rect1)

        current_spell = self.spell_var.get()
        image_path = f"data/{current_spell}{self.count:03d}.png"
        im.save(image_path)
        print(image_path)

        self.label.configure(text=f'{current_spell}{str(self.count)}')
        self.canvas.delete("all")

    def start_pos(self, event):
        self.x = event.x
        self.y = event.y

    def draw_lines(self, event):
        x1, y1 = self.x, self.y
        x2, y2 = event.x, event.y
        self.canvas.create_line(x1, y1, x2, y2, fill='black', width=8)
        self.x, self.y = x2, y2

app = App()
mainloop()
