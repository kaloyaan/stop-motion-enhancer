import deflicker
import video_stabilization
import resize_images
import subprocess
import argparse
import errno
import os
import cv2
import numpy as np


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

def list_images(input_dir):
    """ Produces a list of all the filenames in the input directory. """
    # List the file names
    filenames = sorted(os.listdir(input_dir))

    # Filter out our instructional file
    # (putting it there was a great idea)
    filenames = [f for f in filenames if "DROP_INPUT" not in f and ".DS_Store" not in f]
    # And return them	
    # Create a list of all the paths
    paths = []
    for filename in filenames:
        path = os.path.join(input_dir, filename)
        paths.append(path)

    return paths

def calculate_luminance(image):
	"""
	Calculates the luminance or brightness or whatever of a single OpenCV image.

	https://stackoverflow.com/questions/6442118/python-measuring-pixel-brightness
	"""
	# Get image dimensions
	h = image.shape[0]
	w = image.shape[1]

	# Calculate for each pixel
	brightness = []
	for y in range(0, h, int(h/50)):
		for x in range(0, w, int(w/50)):
			r,g,b = image[y, x]
			brightness.append(0.333*r + 0.333*g + 0.333*b)

	# And return an average
	return np.mean(brightness)

def movingAverage(curve, radius): 
  window_size = 2 * radius + 1
  # Define the filter 
  f = np.ones(window_size)/window_size 
  # Add padding to the boundaries 
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
  # Apply convolution 
  curve_smoothed = np.convolve(curve_pad, f, mode='same') 
  # Remove padding 
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed 

def smooth(trajectory, smoothing_radius): 
  smoothed_trajectory = np.copy(trajectory) 
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=smoothing_radius)

  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame

def stabilize_deflicker(input, output, smoothing_radius=5):

    n_frames = len(input)
    # Pre-define transformation-store array
    transforms = np.zeros((n_frames-1, 3), np.float32) 
    luminances = []

    # Read first frame
    image = cv2.imread(input[0])

    # Get width and height of video stream
    h, w = image.shape[:2]

    # Get frames per second (fps)
    fps = 5 #Manually define fps since it's an image seq

    # Define the codec for output video
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    # Set up output video
    out = cv2.VideoWriter(output, fourcc, fps, (w, h))

    # Convert frame to grayscale for stabilization
    prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # Convert frame to RGB for deflicker
    image_deflicker = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    luminances.append(calculate_luminance(image_deflicker))

    # Detect feature points in first frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                    maxCorners=200,
                                    qualityLevel=0.01,
                                    minDistance=30,
                                    blockSize=3)

    for i in range(1, n_frames-2):
        # Read frame
        curr = cv2.imread(input[i])

        # Convert to grayscale
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

        # Convert frame to RGB for deflicker
        image_deflicker = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
    
        #Calculate luminance
        luminances.append(calculate_luminance(image_deflicker))

        # Calculate optical flow (i.e. track feature points)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 

        # Sanity check
        assert prev_pts.shape == curr_pts.shape 

        # Filter only valid points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        #Find transformation matrix
        m,_ = cv2.estimateAffine2D(prev_pts, curr_pts) # will work with OpenCV>3.0

        # Extract traslation
        dx = m[0,2]
        dy = m[1,2]

        # Extract rotation angle
        da = np.arctan2(m[1,0], m[0,0])
        
        # Store transformation
        transforms[i] = [dx,dy,da]
        
        # Move to next frame
        prev_gray = curr_gray

        print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

        # Detect feature points in frame
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                maxCorners=200,
                                qualityLevel=0.01,
                                minDistance=30,
                                blockSize=3)

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0) 
    
    # Create variable to store smoothed trajectory
    smoothed_trajectory = smooth(trajectory, smoothing_radius) 

    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
    
    # Calculate newer transformation array
    transforms_smooth = transforms + difference
    
    # Write n_frames-1 transformed frames
    for i in range(n_frames-2):
        frame = cv2.imread(input[i])

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i,0]
        dy = transforms_smooth[i,1]
        da = transforms_smooth[i,2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, (w,h))

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized) 

        # Write the frame to the file
    #   frame_out = cv2.hconcat([frame, frame_stabilized])

        # If the image is too big, resize it.
    #   if(frame_out.shape[1] > 1920): 
    #     frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2))
        
    #   cv2.imshow("Before and After", frame_out)

        cv2.imwrite(output+'/%03d.png' % i, frame_stabilized, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        # cv2.waitKey(10)
        # out.write(frame_stabilized)

    out.release()
    # Close windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parseargs()
    # Define the input and output directories
    input_dir = args.Input
    output_dir = args.Output

    #Deflicker + stabilize
    image_paths = list_images(input_dir)
    stabilize_deflicker(image_paths, output_dir)

    # Resize images
    # print("Resizing images")
    # try:
    #     os.mkdir(output_dir+"/output-resize")
    #     resize_images.resize_aspect_fit(input_dir, output_dir+"/output-resize")
    #     # os.makedirs("output-resize")
    #     # resize_images.resize_aspect_fit(input_dir, "output-resize")
    # except OSError as e:
    #     if e.errno != errno.EEXIST:
    #         raise   
    #     pass

    # Call deflicker function
    # print("Deflickering")
    # try:
    #     os.mkdir(output_dir+"/output-deflicker")
    #     deflicker.deflicker_with_files(output_dir+"/output-resize", output_dir+"/output-deflicker")
    #     deflicker.deflicker_with_files(input_dir, output_dir+"/output-deflicker")
    # except OSError as e:
    #     if e.errno != errno.EEXIST:
    #         raise   
    #     pass

    # # Stabilize
    # print("Stabilizing")
    # video_stabilization.stabilize(output_dir+"/output-deflicker", output_dir+"/output-stabilizer.mp4")
    # # video_stabilization.stabilize("output-deflicker", output_dir+"/output-stabilizer.mp4")

    # # Double framerate
    # print("Doubling framerate")
    # infere = subprocess.run(["python3", "inference_video.py", "--exp=1", "--video="+output_dir+"/output-stabilizer.mp4", "--scale=0.5"])
    # print("The exit code was: %d" % infere.returncode)