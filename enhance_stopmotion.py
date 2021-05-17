import deflicker
import video_stabilization
import resize_images
import inference_video
import argparse
import errno
import os
import glob
import cv2
import shutil
from gooey import Gooey, GooeyParser


@Gooey(program_name="Stop Motion Enhancer",
       program_description="Enhance stop motion animation with a single click.",
       default_size=(700, 600),
       header_bg_color='#2A86BF',
       body_bg_color='#EDE1D1',
       optional_cols=1)
def parseargs():
    parser = GooeyParser()
    parser.add_argument('input', metavar="Input directory",
                        widget="DirChooser", help='Select the folder with your images.')
    parser.add_argument('output', metavar="Output directory", widget="DirChooser",
                        help='Select the folder where your output should be storted.')
    parser.add_argument('-resize', metavar="Resize images", action='store_true',
                        help='Makes images twice as small for quicker execution.')
    parser.add_argument('-skipstabilize', metavar="Skip stabilization", action='store_true',
                        help='Do not stabilize the video. Helps when stabilization is buggy.')
    parser.add_argument('-skipinterpolate', metavar="Skip frame doubling", action='store_true',
                        help='Do not double the framerate. Makes program run a lot quicker.')

    args = parser.parse_args()
    return args


def saveFramesAsVideo(frames_folder, output):
    print("Saving video from ", frames_folder, " to ", output)
    cap = cv2.VideoCapture(frames_folder+'/%03d.JPG', cv2.CAP_IMAGES)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = 5  # Manually define fps since it's an image seq
    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Set up output video
    out = cv2.VideoWriter(output, fourcc, fps, (w, h))
    for i in range(n_frames):
        success, frame = cap.read()
        if not success:
            break
        out.write(frame)
    cap.release()
    out.release()


# def saveFramesAsVideo(frames_folder, output):
#     print("Saving video from ", frames_folder, " to ", output)
#     img_array = []
#     for filename in glob.glob(frames_folder + '/*.png'):
#         print("Reading: ", filename)

#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         size = (width, height)
#         img_array.append(img)
#         fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
#         out = cv2.VideoWriter(output, fourcc, 5, size)
#         for i in range(len(img_array)):
#             print("Writing..")
#             out.write(img_array[i])
#         out.release()


if __name__ == "__main__":
    args = parseargs()
    # Define the input and output directories
    input_dir = args.input
    output_dir = args.output

    # Make temp folder for images
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
    if not args.skipstabilize:
        print("Stabilizing")
        if not args.skipinterpolate:
            stabilize_output = temp_f+"/output-stabilizer.mp4"
        else:
            stabilize_output = output_dir+"/output.mp4"  # This will be the final output
        video_stabilization.stabilize(input_dir, stabilize_output)
        input_dir = stabilize_output
    else:
        print("Skipping stabilization")
        if args.skipinterpolate:
            saveFramesAsVideo(
                "/Users/kaloyan/Desktop/CPSC678/stop motion project/stop-motion-enhancer/output/temp/output-deflicker", output_dir+"/output.mp4")
        else:
            saveFramesAsVideo(input_dir, temp_f+"/output-deflicker.mp4")
            input_dir = temp_f+"/output-deflicker.mp4"

    # Double framerate
    if not args.skipinterpolate:
        print("Doubling framerate")
        inference_video.double_frames(input_dir, output_dir+"/output.mp4")

    print("Deleting temporary folder")
    try:
        shutil.rmtree(temp_f)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        pass
