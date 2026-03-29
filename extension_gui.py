# Sangeeth Deleep Menon
# CS5330 Project 5 - Extension: Interactive Digit Recognition GUI
# Spring 2026
#
# A tkinter GUI with a 280x280 drawing canvas. Draw a digit with the mouse
# and the model predicts it in real time on mouse release. A probability bar
# chart shows confidence across all 10 classes. You can also load any image
# from disk.
#
# Run:  python extension_gui.py [model_path]
#       model_path defaults to mnist_model.pth

import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import torch
from torchvision import transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from task1 import MyNetwork

CANVAS_SIZE  = 280   # display canvas in pixels
IMG_SIZE     = 28    # model input size
BRUSH_RADIUS = 12    # half-width of the drawing brush


# loads the trained MNIST model from disk
def load_model(model_path='mnist_model.pth'):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model


# converts a grayscale PIL image (white digit on black) to the normalised
# 1x1x28x28 tensor that MyNetwork expects
def image_to_tensor(pil_img):
    resized = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(resized, dtype=np.uint8)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(Image.fromarray(arr)).unsqueeze(0), arr


class DigitRecogniserApp:

    # ── construction ──────────────────────────────────────────────────────

    def __init__(self, root, model):
        self.root  = root
        self.model = model
        self.root.title('MNIST Digit Recogniser  –  Sangeeth Deleep Menon')
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(False, False)

        # off-screen PIL buffer for the drawn digit
        self._new_buffer()
        self._build_ui()
        self._reset()

    def _new_buffer(self):
        self.pil_buf  = Image.new('L', (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw = ImageDraw.Draw(self.pil_buf)

    # ── UI layout ─────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── left panel: drawing area ──────────────────────────────────────
        left = tk.Frame(self.root, bg='#1e1e1e')
        left.pack(side=tk.LEFT, padx=14, pady=14)

        tk.Label(left, text='Draw a digit below',
                 bg='#1e1e1e', fg='#cccccc',
                 font=('Helvetica', 12)).pack(pady=(0, 4))

        self.canvas = tk.Canvas(
            left, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg='black', cursor='crosshair',
            highlightthickness=2, highlightbackground='#444')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>',      self._on_draw)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

        btn_row = tk.Frame(left, bg='#1e1e1e')
        btn_row.pack(pady=8)

        _btn = dict(font=('Helvetica', 11), width=12,
                    relief=tk.FLAT, cursor='hand2', pady=5)
        tk.Button(btn_row, text='Clear',
                  bg='#555555', fg='white', activebackground='#777',
                  command=self._reset, **_btn).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text='Load Image',
                  bg='#2979ff', fg='white', activebackground='#5393ff',
                  command=self._load_image, **_btn).pack(side=tk.LEFT, padx=4)

        # status bar
        self.status_var = tk.StringVar(value='Draw a digit or load an image.')
        tk.Label(left, textvariable=self.status_var,
                 bg='#1e1e1e', fg='#888888',
                 font=('Helvetica', 9)).pack()

        # ── right panel: results ──────────────────────────────────────────
        right = tk.Frame(self.root, bg='#1e1e1e')
        right.pack(side=tk.LEFT, padx=14, pady=14, fill=tk.BOTH)

        tk.Label(right, text='Prediction',
                 bg='#1e1e1e', fg='#cccccc',
                 font=('Helvetica', 12)).pack(pady=(0, 0))

        # large digit label
        self.pred_label = tk.Label(
            right, text='—', bg='#1e1e1e', fg='#00e676',
            font=('Helvetica', 80, 'bold'), width=3)
        self.pred_label.pack()

        self.conf_label = tk.Label(
            right, text='', bg='#1e1e1e', fg='#aaaaaa',
            font=('Helvetica', 11))
        self.conf_label.pack(pady=(0, 6))

        # probability bar chart
        self.fig, self.ax = plt.subplots(figsize=(4.2, 2.6))
        self.fig.patch.set_facecolor('#1e1e1e')
        self.ax.set_facecolor('#2a2a2a')
        self.bars = self.ax.bar(range(10), [0.0] * 10,
                                color='#2979ff', edgecolor='none')
        self.ax.set_xticks(range(10))
        self.ax.set_xticklabels([str(i) for i in range(10)], color='white', fontsize=10)
        self.ax.set_ylim(0, 1)
        self.ax.set_ylabel('Probability', color='#aaa', fontsize=9)
        self.ax.tick_params(axis='y', colors='#aaa', labelsize=8)
        self.ax.tick_params(axis='x', length=0)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.fig.tight_layout(pad=1.0)

        self.mpl_widget = FigureCanvasTkAgg(self.fig, master=right)
        self.mpl_widget.get_tk_widget().pack()

        # 28x28 preview
        tk.Label(right, text='28 × 28 model input',
                 bg='#1e1e1e', fg='#666666',
                 font=('Helvetica', 9)).pack(pady=(6, 2))
        self.preview = tk.Canvas(
            right, width=84, height=84, bg='black',
            highlightthickness=1, highlightbackground='#444')
        self.preview.pack()

    # ── drawing callbacks ─────────────────────────────────────────────────

    def _on_draw(self, event):
        r = BRUSH_RADIUS
        x, y = event.x, event.y
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill='white', outline='white')
        self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _on_release(self, _event):
        self._predict()

    # ── prediction ────────────────────────────────────────────────────────

    def _predict(self):
        # light blur smooths jagged strokes before downsampling
        smoothed = self.pil_buf.filter(ImageFilter.GaussianBlur(radius=1))
        tensor, arr28 = image_to_tensor(smoothed)

        with torch.no_grad():
            output = self.model(tensor)
        probs = torch.exp(output).squeeze().tolist()
        pred  = int(np.argmax(probs))
        conf  = probs[pred]

        self.pred_label.config(text=str(pred))
        self.conf_label.config(text='{:.1f}% confident'.format(conf * 100))
        self._update_bars(probs, highlight=pred)
        self._update_preview(arr28)
        self.status_var.set('Top guess: {}  |  {:.1f}% confidence'.format(pred, conf * 100))

    def _update_bars(self, probs, highlight=None):
        for i, (bar, p) in enumerate(zip(self.bars, probs)):
            bar.set_height(p)
            bar.set_color('#00e676' if i == highlight else '#2979ff')
        self.mpl_widget.draw()

    def _update_preview(self, arr28):
        big = Image.fromarray(arr28).resize((84, 84), Image.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(big)
        self.preview.delete('all')
        self.preview.create_image(0, 0, anchor=tk.NW, image=self._preview_photo)

    # ── helpers ───────────────────────────────────────────────────────────

    def _reset(self):
        self._new_buffer()
        self.canvas.delete('all')
        self.pred_label.config(text='—')
        self.conf_label.config(text='')
        self._update_bars([0.0] * 10)
        self.preview.delete('all')
        self.status_var.set('Draw a digit or load an image.')

    def _load_image(self):
        path = filedialog.askopenfilename(
            title='Open digit image',
            filetypes=[('Image files', '*.png *.jpg *.jpeg *.bmp')])
        if not path:
            return

        img = Image.open(path).convert('L').resize(
            (CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)
        arr = np.array(img)

        # invert if the digit appears dark on a light background
        if arr.mean() > 128:
            arr = 255 - arr
            img = Image.fromarray(arr)

        self.pil_buf  = img
        self.pil_draw = ImageDraw.Draw(self.pil_buf)

        # show on tkinter canvas
        self.canvas.delete('all')
        self._canvas_photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._canvas_photo)

        self.status_var.set('Loaded: {}'.format(path.split('/')[-1]))
        self._predict()


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv):
    model_path = argv[1] if len(argv) > 1 else 'mnist_model.pth'

    try:
        model = load_model(model_path)
        print('Model loaded from', model_path)
    except FileNotFoundError:
        print('Error: model file "{}" not found.'.format(model_path))
        print('Run task1.py first to train and save the model.')
        sys.exit(1)

    root = tk.Tk()
    DigitRecogniserApp(root, model)
    root.mainloop()


if __name__ == '__main__':
    main(sys.argv)
