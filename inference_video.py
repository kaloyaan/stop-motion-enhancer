import os
import cv2
import torch
import argparse
import numpy as np
from torch.nn import functional as F
import warnings
import _thread
import skvideo.io
from queue import Queue, Empty
from benchmark.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")


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
    if n % 2:
        return [*first_half, middle, *second_half]
    else:
        return [*first_half, *second_half]


def pad_image(img, padding):
    return F.pad(img, padding)


def loadModel():
    modelDir = 'train_log'
    global model

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
        print('{}.{}, {} frames in total, {}FPS'.format(
            video_path_wo_ext, ext, tot_frame, fps))

    h, w, _ = lastframe.shape
    vid_out_name = None
    vid_out = None
    vid_out_name = output
    # vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, (2 ** exp), int(np.round(fps)), ext)
    vid_out = cv2.VideoWriter(vid_out_name, fourcc, fps, (w, h))

    loadModel()

    tmp = max(32, int(32 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    # pbar = tqdm(total=tot_frame)
    progress = 0
    skip_frame = 1
    write_buffer = Queue(maxsize=500)
    read_buffer = Queue(maxsize=500)
    _thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
    _thread.start_new_thread(clear_write_buffer, (write_buffer, vid_out))

    I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(
        device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1, padding)

    while True:
        print("progress: {}/{}".format(progress, int(tot_frame)))
        frame = read_buffer.get()
        if frame is None:
            break
        I0 = I1
        I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(
            device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1, padding)
        I0_small = F.interpolate(
            I0, (32, 32), mode='bilinear', align_corners=False)
        I1_small = F.interpolate(
            I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small, I1_small)

        if ssim > 0.995:
            if skip_frame % 100 == 0:
                print("\nWarning: Your video has {} static frames, skipping them may change the duration of the generated video.".format(
                    skip_frame))
            skip_frame += 1
            if skip:
                progress += 1
                continue

        if ssim < 0.5:
            output = []
            step = 1 / (2 ** exp)
            alpha = 0
            for i in range((2 ** exp) - 1):
                alpha += step
                beta = 1-alpha
                output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[
                              :, :, ::-1].copy()), (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.)
        else:
            output = make_inference(I0, I1, scale, 2**exp-1) if exp else []

        write_buffer.put(lastframe)
        for mid in output:
            mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
            write_buffer.put(mid[:h, :w])
        progress += 1
        lastframe = frame

    write_buffer.put(lastframe)
    if not vid_out is None:
        vid_out.release()
