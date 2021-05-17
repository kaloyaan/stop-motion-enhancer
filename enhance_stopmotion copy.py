import deflicker
import video_stabilization
import resize_images
import subprocess
import argparse
import errno
import os
from gooey import Gooey, GooeyParser

@Gooey(program_name="Enhance Stop Motion",
       default_size=(710, 400),
       navigation='TABBED', 
       header_bg_color = '#2A86BF',
       body_bg_color = '#EDE1D1',)
def parseargs():
    # parser = argparse.ArgumentParser(description='Enhance stop motion animation with a single click')
    parser = GooeyParser()
    parser.add_argument('Input', widget="DirChooser", help='Select the folder with your images.')
    parser.add_argument('Output', widget="DirChooser", help='Select the folder where your output should be storted.')
    # parser.add_argument('--input', dest='input_dir', type=str)
    # parser.add_argument('--output', dest='output_dir', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseargs()
    # Define the input and output directories
    input_dir = args.Input
    output_dir = args.Output

    # Resize images
    print("Resizing images")
    try:
        os.mkdir(output_dir+"/output-resize")
        resize_images.resize_aspect_fit(input_dir, output_dir+"/output-resize")
        # os.makedirs("output-resize")
        # resize_images.resize_aspect_fit(input_dir, "output-resize")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise   
        # time.sleep might help here
        pass

    # Call deflicker function
    print("Deflickering")
    try:
        os.mkdir(output_dir+"/output-deflicker")
        deflicker.deflicker_with_files(output_dir+"/output-resize", output_dir+"/output-deflicker")
        # os.makedirs("output-resize")
        # resize_images.resize_aspect_fit(input_dir, "output-resize")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise   
        # time.sleep might help here
        pass

    # deflicker.deflicker_with_files("output-resize", "output-deflicker")
    # deflicker.deflicker_with_files(output_dir+"/output-resize", "output-deflicker")


    # Stabilize
    print("Stabilizing")
    video_stabilization.stabilize(output_dir+"/output-deflicker", output_dir+"/output-stabilizer.mp4")
    # video_stabilization.stabilize("output-deflicker", output_dir+"/output-stabilizer.mp4")

    # Double framerate
    print("Doubling framerate")
    infere = subprocess.run(["python3", "inference_video.py", "--exp=1", "--video="+output_dir+"/output-stabilizer.mp4", "--scale=0.5"])
    print("The exit code was: %d" % infere.returncode)
