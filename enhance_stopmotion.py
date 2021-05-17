import deflicker
import video_stabilization
import resize_images
import inference_video
import argparse
import errno
import os
import shutil
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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseargs()
    # Define the input and output directories
    input_dir = args.Input
    output_dir = args.Output

    #Make temp folder for images
    temp_f = output_dir+"/temp"
    try:
        os.mkdir(temp_f)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise   
        pass

    # Resize images
    print("Resizing images")
    try:
        resize_output = temp_f+"/output-resize"
        os.mkdir(resize_output)
        # os.makedirs("output-resize")
        # resize_images.resize_aspect_fit(input_dir, "output-resize")
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise   
        pass
    resize_images.resize_aspect_fit(input_dir, resize_output)
    input_dir = resize_output

    # Call deflicker function
    print("Deflickering")
    try:
        deflicker_output = temp_f+"/output-deflicker"
        os.mkdir(deflicker_output)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise   
        pass
    deflicker.deflicker_with_files(input_dir, deflicker_output)
    input_dir = deflicker_output


    # Stabilize
    print("Stabilizing")
    stabilize_output = temp_f+"/output-stabilizer.mp4"
    video_stabilization.stabilize(input_dir, stabilize_output)
    input_dir = stabilize_output

    #TODO: add a converter to mp4 if user doesn't want stabilization
    
    # Double framerate
    print("Doubling framerate")
    inference_video.double_frames(input_dir, output_dir+"/output.mp4")
    #Delete temp folder
    try:
        shutil.rmtree(temp_f)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise   
        pass