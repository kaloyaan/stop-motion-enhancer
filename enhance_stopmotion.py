import deflicker
import video_stabilization
import resize_images
import inference_video
import argparse
import errno
import os
import shutil
from gooey import Gooey, GooeyParser

@Gooey(program_name="Stop Motion Enhancer",
       program_description="Enhance stop motion animation with a single click. Just select a folder with your images and get a video file with your output!",
       default_size=(700, 600),
       header_bg_color = '#2A86BF',
       body_bg_color = '#EDE1D1',)
def parseargs():
    parser = GooeyParser()
    parser.add_argument('input', metavar="Input directory", widget="DirChooser", help='Select the folder with your images.')
    parser.add_argument('output', metavar="Output directory", widget="DirChooser", help='Select the folder where your output should be storted.')
    parser.add_argument('resize', metavar="Resize images", action='store_true', default=False, help='Makes images twice as small for quicker execution.')
    parser.add_argument('deflicker', metavar="Deflicker", action='store_true', default=True, help='Remove flickering.')
    parser.add_argument('stabilize', metavar="Stabilize", action='store_true', default=True, help='Stabilizes video to remove camera shakes.')
    parser.add_argument('interpolate', metavar="Double frames", action='store_true', default=True, help='Interpolates video to double the framerate and make everything look smoother.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parseargs()
    # Define the input and output directories
    input_dir = args.input
    output_dir = args.output

    #Make temp folder for images
    temp_f = output_dir+"/temp"
    try:
        os.mkdir(temp_f)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise   
        pass

    # Resize images
    if args.resize:
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
    if args.deflicker:
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
    if args.stabilize:
        print("Stabilizing")
        if not args.interpolate is None:
            stabilize_output = temp_f+"/output-stabilizer.mp4"
        else:
            stabilize_output = output_dir+"/output.mp4" #This will be the final output
        video_stabilization.stabilize(input_dir, stabilize_output)
        input_dir = stabilize_output
    else:
        print("Skipping stabilization")
        #TODO: turn input_dir into video sequence
    
    # Double framerate
    if args.interpolate:
        print("Doubling framerate")
        inference_video.double_frames(input_dir, output_dir+"/output.mp4")
    
    #Delete temp folder
    print("Deleting temporary folder")
    try:
        shutil.rmtree(temp_f)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise   
        pass