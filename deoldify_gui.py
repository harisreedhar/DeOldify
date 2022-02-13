import os
import cv2
import platform
import threading
import subprocess
import webbrowser
import numpy as np
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw

from deoldify import device
from deoldify.visualize import *
from deoldify.device_id import DeviceId
device.set(device=DeviceId.GPU0)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

class DeoldifyColorize(ModelImageVisualizer):
    def colorize_single(self, frame, render_factor=21, watermark=True, post_process=True):
        frame = Image.fromarray(frame).convert('RGB')
        filtered_image = self.filter.filter(frame, frame, render_factor=render_factor, post_process=post_process)
        if watermark:
            filtered_image = get_watermarked(filtered_image)
        restored = cv2.cvtColor(np.asarray(filtered_image), cv2.COLOR_BGR2RGB)
        return restored

def getDeoldifyModel(model="artistic", factor=21):
    print(f"Loading model {model}")
    root_folder = Path("./")
    if model == "stable":
        learn = gen_inference_wide(root_folder=root_folder, weights_name='ColorizeStable_gen')
    if model == "artistic":
        learn = gen_inference_deep(root_folder=root_folder, weights_name='ColorizeArtistic_gen')
    if model == "video":
        learn = gen_inference_wide(root_folder=root_folder, weights_name='ColorizeVideo_gen')
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=factor)
    vis = DeoldifyColorize(filtr, results_dir="result")
    print("Model loaded")
    return vis

class PreProcess:
    def unsharp_mask(image, kernel_size=(5, 5), sigma=3.0, amount=1.0, threshold=0):
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / max(gamma, 0.00001)
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def apply_brightness_contrast(input_img, brightness=0, contrast=0):
            if brightness != 0:
                if brightness > 0:
                    shadow = brightness
                    highlight = 255
                else:
                    shadow = 0
                    highlight = 255 + brightness
                alpha_b = (highlight - shadow) / 255
                gamma_b = shadow
                buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
            else:
                buf = input_img.copy()
            if contrast != 0:
                f = 131 * (contrast + 127) / (127 * (131 - contrast))
                alpha_c = f
                gamma_c = 127 * (1 - f)
                buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
            return buf

    def process(image, values):
        brightness, contrast = values["brightness"], values["contrast"]
        gamma, sharpness, denoise = values["gamma"], values["sharpness"], values["denoise"]
        detail_kernel, detail_range = values["detail_kernel"], values["detail_range"]
        if gamma != 1:
            image = PreProcess.adjust_gamma(image, gamma=gamma)
        if detail_range > 0:
            image = cv2.detailEnhance(image, sigma_s=detail_kernel, sigma_r=detail_range)
        image = PreProcess.apply_brightness_contrast(image, brightness=brightness, contrast=contrast)
        if sharpness > 0:
            image = PreProcess.unsharp_mask(image, amount=sharpness)
        if denoise > 0:
            image = cv2.fastNlMeansDenoisingColored(image, None, denoise, 6, 7, 21)
        return image

class Utils:
    def create_image(size, text, tk_image=True):
        img = Image.new('RGB', size)
        img_draw = ImageDraw.Draw(img)
        w, h = img_draw.textsize(text)
        img_draw.text(((size[0]-w)/2,(size[1]-h)/2), text, fill="white")
        if tk:
            return ImageTk.PhotoImage(img)
        return img

    def make_preview_image(img, expected_size=(500, 600), scale_to_fit=False, tk_image=True):
        if scale_to_fit:
            base_width = expected_size[0]
            wpercent = (base_width/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((base_width,hsize), Image.ANTIALIAS)

        img.thumbnail((expected_size[0], expected_size[1]))
        delta_width = expected_size[0] - img.size[0]
        delta_height = expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
        new_img = ImageOps.expand(img, padding)
        if tk_image:
            return ImageTk.PhotoImage(new_img)
        return new_img

    def get_error_image(size):
        img = np.ones((size[1], size[0], 3), dtype=np.uint8)
        img[:] = (125, 0, 0)
        return img

class Monitors:
    def __init__(self, frame, size=(500, 400)):
        self.dual_monitor_frame = ttk.Frame(frame)
        self.dual_monitor_frame.pack(side="top")

        # monitor 1
        img_1 = Utils.create_image(size, "Monitor 1")
        self.monitor_1_img = ttk.Label(self.dual_monitor_frame, image=img_1)
        self.monitor_1_img.pack(side="left", fill="both", expand="yes", padx=(2,0), pady=2)
        self.monitor_1_img.photo = img_1

        # monitor 2
        img_2 = Utils.create_image(size, "Monitor 2")
        self.monitor_2_img = ttk.Label(self.dual_monitor_frame, image=img_2)
        self.monitor_2_img.pack(side="left", fill="both", expand="yes", padx=(0,2), pady=2)
        self.monitor_2_img.photo = img_2

        self.controller_frame = ttk.Frame(frame)
        self.controller_frame.pack(side="top")

        # slider
        self.slider = ttk.Scale(self.controller_frame, length=size[0]*2)
        self.slider.pack(side="top", pady=5)

        # buttons and controls
        self.sub_controller_frame = ttk.Frame(self.controller_frame)
        self.sub_controller_frame.pack(side="top", anchor=tk.CENTER, pady=5)

        self.start_frame_spin = ttk.Spinbox(self.sub_controller_frame, width=10, from_=1)
        self.stop_frame_spin = ttk.Spinbox(self.sub_controller_frame, width=10, from_=1)
        self.forward_btn = ttk.Button(self.sub_controller_frame, text=">>", style='btn.TButton')
        self.bacward_btn = ttk.Button(self.sub_controller_frame, text="<<", style='btn.TButton')
        self.trim_start_btn = ttk.Button(self.sub_controller_frame, text=u"[--", style='btn.TButton')
        self.trim_stop_btn = ttk.Button(self.sub_controller_frame, text="--]", style='btn.TButton')
        self.current_frame_label = ttk.Label(self.sub_controller_frame, text="0001", font='Helvetica 16 bold')

        self.start_frame_spin.grid(column=0, row=0, padx=(0,20))
        self.trim_start_btn.grid(column=1, row=0, padx=2)
        self.bacward_btn.grid(column=2, row=0, padx=2)
        self.current_frame_label.grid(column=3, row=0)
        self.forward_btn.grid(column=4, row=0, padx=2)
        self.trim_stop_btn.grid(column=5, row=0, padx=2)
        self.stop_frame_spin.grid(column=6, row=0, padx=(20,0))

    def set_max_frame(self, value):
        self.start_frame_spin.config(to=value)
        self.stop_frame_spin.config(to=value)

    def disable_widgets(self):
        self.slider.config(state="disabled")
        self.bacward_btn.config(state="disabled")
        self.forward_btn.config(state="disabled")
        self.start_frame_spin.config(state="disabled")
        self.stop_frame_spin.config(state="disabled")
        self.trim_start_btn.config(state="disabled")
        self.trim_stop_btn.config(state="disabled")

    def enable_widgets(self):
        self.slider.config(state="enabled")
        self.bacward_btn.config(state="enabled")
        self.forward_btn.config(state="enabled")
        self.start_frame_spin.config(state="enabled")
        self.stop_frame_spin.config(state="enabled")
        self.trim_start_btn.config(state="enabled")
        self.trim_stop_btn.config(state="enabled")

class MenuBar:
    def __init__(self, root):
        self.menu_bar = tk.Menu(root)
        self.filemenu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.filemenu)

        self.view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='Tools', menu=self.view_menu)

        self.properties_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='Properties', menu=self.properties_menu)

        helpmenu = tk.Menu(self.menu_bar, tearoff=0)
        url = "https://github.com/harisreedhar/DeOldify/issues/new"
        helpmenu.add_command(label="Report", command=lambda: webbrowser.open(url, new=1))
        info = "Unofficial gui implementaion of deoldify.\nAuthor: Harisreedhar\n"
        helpmenu.add_command(label="About", command=lambda: messagebox.showinfo("About", info))
        self.menu_bar.add_cascade(label="Help", menu=helpmenu)
        root.config(menu=self.menu_bar)

class StatusBar:
    def __init__(self, root):
        self.status = tk.Label(root, text="...", bd=1, relief=tk.RIDGE, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

class PreProcessWindow:
    def __init__(self, root, size):
        pre_process_frame = ttk.Frame(root)
        pre_process_frame.pack(side="top", fill="both", padx=5, pady=5)

        settings = {"length":int(size[0]/4), "length":170, "orient":tk.HORIZONTAL}

        self.brightness = tk.DoubleVar(value=0)
        brightness_label = ttk.Label(pre_process_frame, text="Brightness")
        self.brightness_box = ttk.Scale(pre_process_frame, variable=self.brightness, from_=-200, to=200, **settings)
        self.contrast = tk.DoubleVar(value=0)
        contrast_label = ttk.Label(pre_process_frame, text="Contrast")
        self.contrast_box = ttk.Scale(pre_process_frame, variable=self.contrast, from_=-200, to=200, **settings)

        self.detail_kernel= tk.DoubleVar(value=10)
        detail_kernel_label = ttk.Label(pre_process_frame, text="Detail kernel")
        self.detail_kernel_box = ttk.Scale(pre_process_frame, variable=self.detail_kernel, from_=1, to=200, **settings)
        self.detail_range= tk.DoubleVar(value=0)
        detail_range_label = ttk.Label(pre_process_frame, text="Detail range")
        self.detail_range_box = ttk.Scale(pre_process_frame, variable=self.detail_range, from_=0, to=100, **settings)

        self.gamma= tk.DoubleVar(value=10)
        gamma_label = ttk.Label(pre_process_frame, text="Gamma")
        self.gamma_box = ttk.Scale(pre_process_frame, variable=self.gamma, from_=1, to=100, **settings)
        self.sharpness= tk.DoubleVar(value=0)
        sharpness_label = ttk.Label(pre_process_frame, text="Sharpness")
        self.sharpness_box = ttk.Scale(pre_process_frame, variable=self.sharpness, from_=0, to=100, **settings)
        self.denoise= tk.DoubleVar(value=0)
        denoise_label = ttk.Label(pre_process_frame, text="Denoise")
        self.denoise_box = ttk.Scale(pre_process_frame, variable=self.denoise, from_=0, to=100, **settings)

        brightness_label.grid(column=0, row=0, padx=(5,0))
        self.brightness_box.grid(column=1, row=0, padx=(0,5))
        contrast_label.grid(column=2, row=0, padx=(5,0))
        self.contrast_box.grid(column=3, row=0, padx=(0,5))

        detail_kernel_label.grid(column=4, row=0, padx=(5,0))
        self.detail_kernel_box.grid(column=5, row=0, padx=(0,5))
        detail_range_label.grid(column=6, row=0, padx=(5,0))
        self.detail_range_box.grid(column=7, row=0, padx=(0,5))

        gamma_label.grid(column=0, row=2, padx=(5,0))
        self.gamma_box.grid(column=1, row=2, padx=(0,5))
        sharpness_label.grid(column=2, row=2, padx=(5,0))
        self.sharpness_box.grid(column=3, row=2, padx=(0,5))
        denoise_label.grid(column=4, row=2, padx=(5,0))
        self.denoise_box.grid(column=5, row=2, padx=(0,5))

    def get_values(self):
        pre_process_values = {
            "brightness": self.brightness.get(),
            "contrast": self.contrast.get()/10,
            "gamma": self.gamma.get()/10,
            "detail_kernel": int(self.detail_kernel.get()),
            "detail_range": self.detail_range.get()/1000,
            "sharpness": self.sharpness.get()/10,
            "denoise": self.denoise.get()/10
        }
        return pre_process_values

class DeoldifyWindow:
    def __init__(self, root):
        self.frame = ttk.Frame(root)
        self.frame.pack(side="top", expand="yes", fill="x")

        ttk.Label(self.frame, text="Model:").pack(side="left", padx=(10,0), pady=5)
        self.model = tk.StringVar()
        self.model_menu = ttk.Combobox(self.frame, width = 10, state="readonly", justify="center", textvariable = self.model)
        self.model_menu['values'] = ('stable', 'video', 'artistic')
        self.model_menu.current(1)
        self.model_menu.pack(side="left", pady=5)

        ttk.Label(self.frame, text="Render Factor:").pack(side="left", padx=(10,0), pady=5)
        self.factor = tk.IntVar()
        self.factor.set(21)
        factor = ttk.Spinbox(self.frame, width=8, from_=1, to=100, textvariable=self.factor)
        factor.pack(side="left", pady=5)

        self.post_process = tk.IntVar()
        self.post_process.set(1)
        postProcess = ttk.Checkbutton(self.frame, text="Post-process", variable=self.post_process)
        postProcess.pack(side="left", padx=(10,0), pady=5)

        self.water_mark = tk.IntVar()
        self.water_mark.set(1)
        water_mark = ttk.Checkbutton(self.frame, text="Water-mark", variable=self.water_mark)
        water_mark.pack(side="left", padx=(10,0), pady=5)

        self.update_btn = ttk.Button(self.frame, text="Preview", width=20)
        self.update_btn.pack(side="right", padx=(0,10), pady=5)

    def disable_widgets(self):
        self.model_menu.config(state="disabled")
        self.update_btn.config(state="disabled")

    def enable_widgets(self):
        self.model_menu.config(state="readonly")
        self.update_btn.config(state="enabled")

class MainWindow:
    def __init__(self, root, monitor_size=(500,400)):
        self.root = root
        self.monitor_size = monitor_size
        self.reader_available = False
        self.auto_update = tk.BooleanVar(value=True)

        self.menu_bar = MenuBar(self.root)
        self.set_menubar_functions()
        self.staus_bar = StatusBar(self.root)

        self.monitors = Monitors(self.root, size=self.monitor_size)
        self.set_monitor_variables()
        self.set_monitor_functions()

        tabControl = ttk.Notebook(root)

        tab1 = ttk.Frame(tabControl)
        tab2 = ttk.Frame(tabControl)

        tabControl.add(tab1, text ='Deoldify')
        tabControl.add(tab2, text ='Pre-process')
        tabControl.pack(side="top", expand = 1, fill ="both")

        self.deoldify = DeoldifyWindow(tab1)
        self.pre_process = PreProcessWindow(tab2, monitor_size)

        self.model_name = self.deoldify.model.get()
        self.load_deoldify_model()
        self.deoldify.model_menu.bind('<<ComboboxSelected>>', self.load_deoldify_model)
        self.deoldify.update_btn.config(command=self.update_monitor_2)

        self.enable_auto_updates()

    def load_deoldify_model(self, *args):
        def proc():
            self.deoldify.disable_widgets()
            model_name = self.deoldify.model.get()
            if not hasattr(self, "deoldify_model") or self.model_name != model_name:
                self.set_status("Loading deoldify model...")
                self.deoldify_model = getDeoldifyModel(model=self.deoldify.model.get())
                self.model_name = model_name
                self.set_status("Deoldify model loaded")
            self.deoldify.enable_widgets()
        threading.Thread(target=proc).start()

    def set_monitor_variables(self):
        self.current_frame = tk.IntVar(value=1)
        self.start_frame = tk.IntVar(value=1)
        self.stop_frame = tk.IntVar(value=100)

        self.monitors.slider.config(variable=self.current_frame)
        self.monitors.slider.config(from_=self.start_frame.get())
        self.monitors.slider.config(to=self.stop_frame.get())
        self.monitors.start_frame_spin.config(textvariable=self.start_frame, to=1000)
        self.monitors.stop_frame_spin.config(textvariable=self.stop_frame, to=1000)

    def set_monitor_functions(self):
        m = self.monitors
        def slider_func(*args):
            m.slider.config(from_=self.start_frame.get())
            m.slider.config(to=self.stop_frame.get())
            frame = str(max(self.current_frame.get(),1))
            m.current_frame_label.config(text=frame.zfill(4))
            self.update_monitor_1()

        def trim_start_func(*args):
            m.start_frame_spin.set(self.current_frame.get())
            slider_func()

        def trim_stop_func(*args):
            m.stop_frame_spin.set(self.current_frame.get())
            slider_func()

        m.slider.config(command=slider_func)
        m.trim_start_btn.config(command=trim_start_func)
        m.trim_stop_btn.config(command=trim_stop_func)
        m.forward_btn.config(command=lambda: m.slider.set(self.current_frame.get() + 1))
        m.bacward_btn.config(command=lambda: m.slider.set(self.current_frame.get() - 1))

    def set_menubar_functions(self):
        m = self.menu_bar
        filemenu = m.filemenu

        filemenu.add_command(label="Import video", command=self.open_video)
        filemenu.add_separator()
        filemenu.add_command(label="Save video", command=lambda: threading.Thread(target=self.save_video).start())
        filemenu.add_command(label="Save as png sequence", command=lambda: threading.Thread(target=self.save_image_sequence).start())
        filemenu.add_command(label="Save screenshot", command=lambda: threading.Thread(target=self.save_screen_shot).start())
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=root.quit)

        view_menu = m.view_menu
        view_menu.add_command(label="Remove duplicate-frames", command=lambda: threading.Thread(target=self.remove_duplicate_frames).start())

        properties_menu = m.properties_menu
        properties_menu.add_checkbutton(label="Auto update", variable=self.auto_update, onvalue=1, offvalue=0, command=self.enable_auto_updates)

    def reset_monitor_controls(self):
        self.start_frame.set(1)
        self.current_frame.set(1)
        self.monitors.start_frame_spin.config(from_=1, to=99999)
        self.monitors.stop_frame_spin.config(from_=1, to=99999)
        self.monitors.slider.config(from_=1, to=99999)
        self.monitors.slider.set(1)

    def freeze_widgets(func):
        def wrapper(self, *arg, **kwargs):
            if not self.reader_available:
                #self.set_status("Operation failed!")
                return
            self.monitors.disable_widgets()
            res = func(self, *arg, **kwargs)
            self.monitors.enable_widgets()
            return res
        return wrapper

    def read_video(self, path="/home/hari/Desktop/gandhiji.mp4"):
        try:
            self.reset_monitor_controls()
            self.input_path = path
            self.cap = cv2.VideoCapture(path)
            self.input_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.stop_frame.set(self.total_frames)
            self.monitors.set_max_frame(self.total_frames)
            self.reader_available = True
            self.update_monitor_1()
            self.set_status(f"Imported: {path}")
        except Exception as e:
            print(e)
            self.reader_available = False
            self.set_status(str(e))

    def get_video_frame(self, frame=1):
        if frame <= 0 or not self.reader_available:
            return Utils.get_error_image(self.monitor_size)
        self.cap.set(1, frame - 1)
        _, image = self.cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def enable_auto_updates(self, *args):
        func = lambda x: self.set_status("Auto-update disabled")
        auto_update = self.auto_update.get()
        if auto_update:
            func = self.update_monitor_2
            self.set_status("Auto-update enabled")
        values = ("<ButtonRelease-1>", func)
        self.monitors.slider.bind(*values)
        self.monitors.forward_btn.bind(*values)
        self.pre_process.brightness_box.bind(*values)
        self.pre_process.contrast_box.bind(*values)
        self.pre_process.detail_kernel_box.bind(*values)
        self.pre_process.detail_range_box.bind(*values)
        self.pre_process.gamma_box.bind(*values)
        self.pre_process.sharpness_box.bind(*values)
        self.pre_process.denoise_box.bind(*values)
        self.root.update()

    def update_monitor_1(self, *args):
        frame = self.current_frame.get()
        if frame <= 0: return
        image = self.get_video_frame(frame=frame)
        pil_image = Image.fromarray(image).convert('RGB')
        img = Utils.make_preview_image(pil_image, expected_size=self.monitor_size, scale_to_fit=True)
        self.monitors.monitor_1_img.configure(image = img)
        self.monitors.monitor_1_img.image = img

    @freeze_widgets
    def update_monitor_2(self, *args):
        frame = self.current_frame.get()
        if frame <= 0 or not self.reader_available: return
        image = self.get_video_frame(frame=frame)
        if hasattr(self, "deoldify_model"):
            img = self.colorize(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img).convert('RGB')
            img = Utils.make_preview_image(pil_image, expected_size=self.monitor_size, scale_to_fit=True)
            self.monitors.monitor_2_img.configure(image = img)
            self.monitors.monitor_2_img.image = img

    def pre_process_update(self, *args):
        frame = self.current_frame.get()
        image = self.get_video_frame(frame=frame)
        image = PreProcess.process(image, self.pre_process.get_values())
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img).convert('RGB')
        img = Utils.make_preview_image(pil_image, expected_size=self.monitor_size, scale_to_fit=True)
        self.monitors.monitor_2_img.configure(image = img)
        self.monitors.monitor_2_img.image = img

    def colorize(self, image, pre_process=True):
        if pre_process:
            image = PreProcess.process(image, self.pre_process.get_values())
        factor = self.deoldify.factor.get()
        water_mark = self.deoldify.water_mark.get()
        post_process = self.deoldify.post_process.get()
        output = self.deoldify_model.colorize_single(image,
            render_factor=factor,
            watermark=water_mark,
            post_process=post_process)
        return output

    def open_video(self):
        file_name = filedialog.askopenfilename(initialdir = "/", title = "Select a File")
        if file_name:
            self.read_video(path=file_name)

    @freeze_widgets
    def save_video(self):
        file_path = filedialog.asksaveasfile(mode='w', filetypes=[("mp4", "*.mp4")])
        if file_path is None:
            return
        path = file_path.name
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        video_writer = cv2.VideoWriter(path, codec, self.input_fps, (width,height), True)
        start = self.start_frame.get()
        stop = self.stop_frame.get() + 1
        for i in range(start, stop):
            self.set_status(f"Saving video frame {i} of {stop}...")
            img = self.get_video_frame(frame=i)
            img = self.colorize(img)
            video_writer.write(img)
        video_writer.release()
        self.set_status("Done!")

    @freeze_widgets
    def save_image_sequence(self):
        save_dir = filedialog.askdirectory(title='Select folder to save sequence')
        if save_dir is None:
            return
        start = self.start_frame.get()
        stop = self.stop_frame.get() + 1
        fill = len(str(stop)) + 2
        for i in range(start, stop):
            num = str(i)
            self.set_status(f"Saving sequence {num} of {stop}...")
            path = os.path.join(save_dir, f"img_{num.zfill(fill)}.png")
            img = self.get_video_frame(frame=i)
            img = self.colorize(img)
            cv2.imwrite(path, img)
        self.set_status("Done!")

    @freeze_widgets
    def save_screen_shot(self):
        file_path = filedialog.asksaveasfile(mode='w', filetypes=[("jpg", "*.jpg"), ("png", "*.png")])
        if file_path is None:
            return
        num = self.current_frame.get()
        path = file_path.name
        img = self.get_video_frame(frame=num)
        img = self.colorize(img)
        cv2.imwrite(path, img)
        self.set_status("Done!")

    # needs more work
    @freeze_widgets
    def remove_duplicate_frames(self):
        file_path = filedialog.asksaveasfile(mode='w', filetypes=[("mp4", "*.mp4")])
        if file_path is None:
            return
        input_path = self.input_path
        out_path = file_path.name
        command = f'ffmpeg -y -i {input_path} -vf mpdecimate,setpts=N/FRAME_RATE/TB -an {out_path}'
        self.set_status("Removing duplicate frames...")
        process = subprocess.Popen(command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=platform.system() != 'Windows')
        for line in process.stdout:
            self.set_status(line)
        response = messagebox.askquestion("Duplicate frames removed", "Import processed video?", icon='warning')
        if response == 'yes':
            self.read_video(path=out_path)
        else:
            self.set_status("Saved as " + out_path)

    def set_status(self, text):
        self.staus_bar.status.config(text=text)

if __name__ == "__main__":
    root = tk.Tk()
    root.style = ttk.Style()
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('btn.TButton', font=('Helvetica 12 bold'))

    # hide hidden files in filedialog
    try:
        try:
            root.tk.call('tk_getOpenFile', '-foobarbaz')
        except tk.TclError:
            pass
        root.tk.call('set', '::tk::dialog::file::showHiddenBtn', '1')
        root.tk.call('set', '::tk::dialog::file::showHiddenVar', '0')
    except:
        pass

    root.title("Deoldify GUI")
    root.resizable(width=False, height=False)
    app = MainWindow(root, monitor_size=(500,400))
    root.mainloop()
