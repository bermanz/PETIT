import tkinter as tk
from pathlib import Path
import numpy as np
from PIL import ImageTk,Image

class ImageSlider:

    ## GUI infra Init:
    def __init__(self, images_dir:Path, images_format=".png") -> None:
        
        # get directory images' names and amount:
        self.base_dir = images_dir
        self.images_names = [name.stem for name in images_dir.glob("*")]
        non_numeric_idx = [i for i, name in enumerate(self.images_names) if not name.isnumeric()]
        for idx in non_numeric_idx:
            self.images_names.pop(idx)
        if all([name.isnumeric() for name in self.images_names]):
            self.images_names.sort(key=int)
        self.image_format = images_format

        # initialize gui window and components:
        self.root = tk.Tk()
        self.root.geometry('1000x800')
        self.root.resizable(False, False)
        self.root.title('Image Slider')

        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)

        self.current_idx = tk.IntVar() # slider current value
        self.image_name = tk.StringVar() # current file name
        self.image_label = "file name"
        self._set_img_name(self._get_fname())

        # other gui's components:
        self.headline = self._init_label(fontsize=24)
        self._init_canvas()
        self._init_slider()
        self._init_label(var=self.image_name)
        
    def open(self):
        """A command for programatically opening the UI (to allow post-init modifications before trigerring)"""
        self.root.mainloop()
    
    def _init_canvas(self):
        self.canvas = tk.Canvas(self.root, width = 800, height = 550)      
        self.canvas.pack()

    def _init_slider(self):
        slider = tk.Scale(
            self.root,
            from_=0,
            to=len(self.images_names)-1,
            length=900,
            orient='horizontal',  # vertical
            command=self._get_current_value,
            variable=self.current_idx,
            showvalue=False
        )
        slider.pack()

    def _init_label(self, text="Image Slider", var=None, fontsize=12):
        label = tk.Label(
            self.root,
            text=text,
            font=("Ariel", fontsize),
            textvariable=var
        )
        label.pack()
        return label

    def _get_fname(self):
        return self.images_names[self.current_idx.get()]
    
    def _load_img(self, fname):
        fpath = self.base_dir / (fname + self.image_format)
        if fpath.suffix in [".npy", ".npz"]:
            mat = np.load(fpath)
            if fpath.suffix == ".npz":
                mat = mat["image"]
            norm_mat = 255 * ((mat - mat.min()) / mat.ptp())
            image = Image.fromarray(norm_mat.astype(np.uint8))
        elif fpath.suffix in [".png", ".jpg", ".tiff"]:
            image = Image.open(fpath)
        else:
            raise Exception("Need to define a loader to convert the input format to PIL image!")
        return image

    def _set_img(self, image:Image):
        self.image = ImageTk.PhotoImage(image.resize((image.size[0] * 2, image.size[1] * 2)))
        self.canvas.create_image(75, 25, anchor=tk.NW, image=self.image)

    ## GUI callbacks:
    def _get_current_value(self, event):        
        fname = self._get_fname()
        image = self._load_img(fname)
        self._set_img(image)
        self._set_img_name(fname)
        return self.image_name

    def _set_img_name(self, name):
        self.image_name.set(f"{self.image_label}: {name}")


if __name__=="__main__":
    image_slider = ImageSlider(images_dir=Path(r"Datasets\Tau2_Aerial\pan\val"), images_format=".npy")
    image_slider.open()
