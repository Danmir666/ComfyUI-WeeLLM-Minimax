r"""Upscale a finished video 2x with an ESRGAN-style model from ComfyUI/models/upscale_models, keeping the audio.

Run it with ComfyUI's python from the portable folder:
    python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-WeeLLM\upscale_video.py ComfyUI\output\video\Minimax_00018_.mp4
"""
import argparse
import os
import sys
import time

import av
import numpy as np
import torch

COMFY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, COMFY_DIR)

import comfy.utils
from spandrel import ModelLoader

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--model", default="2x-AnimeSharpV4_Fast_RCAN_PU.safetensors", help="file name in models/upscale_models")
parser.add_argument("--output", help="default: <video>_2x.mp4 next to the input")
parser.add_argument("--crf", type=int, default=18)
args = parser.parse_args()

output = args.output or os.path.splitext(args.video)[0] + "_2x.mp4"
state = comfy.utils.load_torch_file(os.path.join(COMFY_DIR, "models", "upscale_models", args.model), safe_load=True)
model = ModelLoader().load_from_state_dict(state).eval().to("cuda")

src = av.open(args.video)
video_in = src.streams.video[0]
audio_in = src.streams.audio[0] if src.streams.audio else None
dst = av.open(output, "w")
video_out = dst.add_stream("libx264", rate=video_in.average_rate)
video_out.width = video_in.width * model.scale
video_out.height = video_in.height * model.scale
video_out.pix_fmt = "yuv420p"
video_out.options = {"crf": str(args.crf), "preset": "medium"}
audio_out = dst.add_stream_from_template(audio_in) if audio_in else None

start = time.time()
done = 0
streams = [video_in] + ([audio_in] if audio_in else [])
for packet in src.demux(*streams):
    if audio_in and packet.stream == audio_in:
        if packet.dts is None:
            continue
        packet.stream = audio_out
        dst.mux(packet)
        continue
    for frame in packet.decode():
        pixels = torch.from_numpy(frame.to_ndarray(format="rgb24")).to("cuda").permute(2, 0, 1)[None].float().div(255)
        with torch.inference_mode():
            upscaled = model(pixels).clamp(0, 1)
        array = (upscaled[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy())
        for out_packet in video_out.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
            dst.mux(out_packet)
        done += 1
        if done % 25 == 0:
            print(f"{done}/{video_in.frames} frames, {time.time() - start:.0f}s", flush=True)
for out_packet in video_out.encode():
    dst.mux(out_packet)
dst.close()
print(f"Saved {output} ({done} frames, {time.time() - start:.0f}s)")
