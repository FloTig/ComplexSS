import glob
from sys import argv
import copy
from PIL import Image


def make_gif(frame_folder, name):
    fnames = [fname for fname in glob.glob(f"{frame_folder}/*.png")]
    print(fnames)
    fnames.sort()
    frame_one = Image.open(fnames[0])
    images = []
    frame_one.save(f"{name}.gif", format="GIF", duration=1/10, loop=0)
    for fname in fnames[1:]:
        try:
            image = copy.copy(Image.open(fname))
        except:
            print(f"error at {fname}")
        images.append(image)
        
    frame_one.save(f"{name}.gif", format="GIF", save_all=True, 
        append_images=images, duration=1/10, loop=0)

if __name__ == "__main__":
    if len(argv) < 2:
        print("pathname ?")
        exit()

    make_gif(f"{argv[1]}", argv[1])