import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from benchmark.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

def parseargs():
    parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
    parser.add_argument('--video', dest='video', type=str, default=None)
    parser.add_argument('--output', dest='output', type=str, default=None)
    parser.add_argument('--img', dest='img', type=str, default=None)
    parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
    parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
    parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
    parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
    parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
    parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
    parser.add_argument('--fps', dest='fps', type=int, default=None)
    parser.add_argument('--png', dest='png', action='store_true', help='whether to vid_out png format vid_outs')
    parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='vid_out video extension')
    args = parser.parse_args()
    return args

def clear_write_buffer(write_buffer, vid_out):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        vid_out.write(item[:, :, ::-1])

def build_read_buffer(read_buffer, videogen):
    try:
        for frame in videogen:
             read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

def make_inference(I0, I1, scale, n):
    middle = model.inference(I0, I1, scale)
    if n == 1:
        return [middle]
    first_half = make_inference(I0, middle, scale, n=n//2)
    second_half = make_inference(middle, I1, scale, n=n//2)
    if n%2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]

def pad_image(img, padding):
    return F.pad(img, padding)

modelDir = 'train_log'
try:
    from model.RIFE_HDv2 import Model
    model = Model()
    model.load_model(modelDir, -1)
    print("Loaded v2.x HD model.")
except:
    from model.RIFE_HD import Model
    model = Model()
    model.load_model(modelDir, -1)
    print("Loaded v1.x HD model")
model.eval()
model.device()

def double_frames(video, output, fps=5, scale=0.5):
    assert (not video is None)
    ext = 'mp4'
    exp = 1
    skip = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    if not video is None:
        videoCapture = cv2.VideoCapture(video)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()
        fpsNotAssigned = True
        fps = fps * (2 ** exp)
        videogen = skvideo.io.vreader(video)
        lastframe = next(videogen)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_path_wo_ext, ext = os.path.splitext(video)
        print('{}.{}, {} frames in total, {}FPS'.format(video_path_wo_ext, ext, tot_frame, fps))

    h, w, _ = lastframe.shape
    vid_out_name = None
    vid_out = None
    vid_out_name = output
    # vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** exp), int(np.round(fps)), ext)
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, fps, (w, h))

    tmp = max(32, int(32 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)
    skip_frame = 1
    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (write_buffer, vid_out))

    I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1, padding)

    while True:
        frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1, padding)
        I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small, I1_small)

        if ssim > 0.995:
            if skip_frame % 100 == 0:
                print("\nWarning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(skip_frame))
            skip_frame += 1
            if skip:
                pbar.update(1)
                continue

        if ssim < 0.5:
            output = []
            step = 1 / (2 ** exp)
            alpha = 0
            for i in range((2 ** exp) - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        else:
            output = make_inference(I0, I1, scale, 2**exp-1) if exp else []

        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
        pbar.update(1)
        lastframe = frame

    write_buffer.put(lastframe)
    import time
    while(not write_buffer.empty()):
        time.sleep(0.1)
    pbar.close()
    if not vid_out is None:
        vid_out.release()