import os
import sys
import gc

import torch
import numpy as np
from PIL import Image, ImageOps

import comfy.utils
import comfy.model_management


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_from_comfy(tensor):
    """Convert a ComfyUI IMAGE tensor [B,H,W,C] → PIL Image (first frame)."""
    np_img = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np_img, "RGB")


def _comfy_from_pil(image: Image.Image):
    """Convert a PIL Image → ComfyUI IMAGE tensor [1,H,W,C] float32 0-1."""
    np_img = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).unsqueeze(0)


def _make_step_callback(pbar: comfy.utils.ProgressBar):
    """Return a diffusers callback_on_step_end that ticks the bar and honours cancel."""
    def _cb(pipeline, step_index, timestep, callback_kwargs):
        pbar.update(1)
        comfy.model_management.throw_exception_if_processing_interrupted()
        return callback_kwargs
    return _cb


# ---------------------------------------------------------------------------
# Node 1: WeeLLM Loader
# ---------------------------------------------------------------------------

class WeeLLMLoaderNode:
    """
    Loads the WeeLLM pipeline once and caches it in the ComfyUI workflow.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "black-forest-labs/FLUX.1-schnell"}),
                "task": (["text-to-image", "image-to-image", "video"], {"default": "text-to-image"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("WEE_PIPELINE",)
    FUNCTION = "load_pipeline"
    CATEGORY = "WeeLLM"

    def load_pipeline(self, model_path, task, dtype):
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

        if task == "text-to-image":
            from weellm import WeePipeline as PipelineClass
        elif task == "image-to-image":
            from weellm import WeeImagePipeline as PipelineClass
        elif task == "video":
            from weellm.weevideopipeline import WeeVideoPipeline as PipelineClass
        else:
            raise ValueError(f"Unknown task: {task}")

        pipe = PipelineClass.from_pretrained(model_path, torch_dtype=torch_dtype)
        return (pipe,)


# ---------------------------------------------------------------------------
# Node 2: WeeLLM Generate (Text-to-Image + Image-to-Image + Inpainting)
# ---------------------------------------------------------------------------

class WeeLLMGenerateNode:
    """
    Unified WeeLLM generation node.

    * No image connected   → Text-to-Image  (WeePipeline)
    * image connected      → Image-to-Image (WeeImagePipeline)
    * image + mask_image   → Inpainting     (WeeImagePipeline with mask)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":        ("WEE_PIPELINE",),
                "prompt":          ("STRING",  {"multiline": True, "default": "A majestic lion at golden hour"}),
                "negative_prompt": ("STRING",  {"multiline": True, "default": "", "tooltip": "Leave empty if model does not support it (e.g. FLUX)"}),
                "height":          ("INT",     {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "width":           ("INT",     {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "num_steps":       ("INT",     {"default": 4,    "min": 1,   "max": 200}),
                "guidance_scale":  ("FLOAT",   {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1, "tooltip": "-1 means use pipeline default"}),
                "image_guidance_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1, "tooltip": "-1 means use pipeline default (for InstructPix2Pix etc)"}),
                "seed":            ("INT",     {"default": 0,    "min": 0,   "max": 0xffffffffffffffff}),
                "strength":        ("FLOAT",   {"default": 0.8,  "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image":           ("IMAGE",),
                "mask_image":      ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "WeeLLM"

    def generate(self, pipeline, prompt, negative_prompt, height, width, num_steps, guidance_scale, image_guidance_scale, seed, strength, image=None, mask_image=None):

        pipe = pipeline
        pbar = comfy.utils.ProgressBar(num_steps)

        # --- Build generation kwargs ---------------------------------------
        call_kwargs = dict(
            num_inference_steps=num_steps,
            height=height,
            width=width,
            seed=seed,
            callback_on_step_end=_make_step_callback(pbar),
        )

        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        if guidance_scale >= 0.0:
            call_kwargs["guidance_scale"] = guidance_scale

        if image_guidance_scale >= 0.0:
            call_kwargs["image_guidance_scale"] = image_guidance_scale

        if image is not None:
            call_kwargs["image"] = _pil_from_comfy(image)
            call_kwargs["strength"] = strength

        if mask_image is not None:
            call_kwargs["mask_image"] = _pil_from_comfy(mask_image)

        # --- Generate ------------------------------------------------------
        if image is not None:
            result = pipe.generate(prompt, **call_kwargs)
        else:
            result = pipe.generate(prompt, **call_kwargs)

        # generate() returns a PIL Image directly
        if not isinstance(result, Image.Image):
            if hasattr(result, "images"):
                result = result.images[0]
            elif isinstance(result, list):
                result = result[0]
            else:
                raise ValueError(f"Unexpected generate() return type: {type(result)}")

        return (_comfy_from_pil(result),)


# ---------------------------------------------------------------------------
# Node 3: WeeLLM Video Generate  (currently MiniMax-H3 FL2VA)
# ---------------------------------------------------------------------------

class WeeLLMVideoGenerateNode:
    """
    WeeLLM video generation node.

    Outputs a batch of IMAGE frames that you can route to a Video Combine
    node or inspect frame-by-frame. Works with MiniMax-H3 and any other
    video model WeeLLM supports.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":    ("WEE_PIPELINE",),
                "prompt":      ("STRING", {"multiline": True, "default": "A cat walking across a wooden floor"}),
                "height":      ("INT",    {"default": 544, "min": 64, "max": 2048, "step": 8}),
                "width":       ("INT",    {"default": 960, "min": 64, "max": 2048, "step": 8}),
                "num_frames":  ("INT",    {"default": 75,   "min": 22, "max": 500}),
                "num_steps":   ("INT",    {"default": 6,    "min": 1,  "max": 100}),
                "seed":        ("INT",    {"default": 0,    "min": 0,  "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image":       ("IMAGE",),   # first frame
                "last_image":  ("IMAGE",),   # last frame (MiniMax FL2VA)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "WeeLLM"

    def generate(self, pipeline, prompt, height, width, num_frames, num_steps, seed,
                 image=None, last_image=None):

        pipe = pipeline

        generator = torch.Generator("cpu").manual_seed(seed)

        call_kwargs = dict(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            generator=generator,
        )

        if image is not None:
            call_kwargs["image"] = _pil_from_comfy(image)

        if last_image is not None:
            call_kwargs["last_image"] = _pil_from_comfy(last_image)

        out = pipe(prompt, **call_kwargs)

        # --- Unpack video frames → ComfyUI IMAGE batch --------------------
        # MiniMax returns decode_state with a "videos" key containing PIL frames
        # or a tensor of shape [T, C, H, W] / [C, T, H, W].
        frames_pil = []

        if hasattr(out, "videos") and out.videos is not None:
            raw = out.videos[0] if isinstance(out.videos, list) else out.videos
        elif isinstance(out, dict):
            raw = out.get("videos") or out.get("video")
            if isinstance(raw, list):
                raw = raw[0]
        else:
            raw = None

        if raw is None:
            raise ValueError("WeeVideoPipeline returned no video frames.")

        if isinstance(raw, list):
            # List of PIL Images
            frames_pil = [f.convert("RGB") if hasattr(f, "convert") else f for f in raw]
        elif isinstance(raw, torch.Tensor):
            # Tensor — normalise to [T, H, W, C] uint8
            v = raw.float()
            if v.shape[0] == 3 and v.ndim == 4:          # [C, T, H, W]
                v = v.permute(1, 2, 3, 0)
            elif v.ndim == 4 and v.shape[1] == 3:         # [T, C, H, W]
                v = v.permute(0, 2, 3, 1)
            v = v.clamp(0, 1)
            np_frames = (v.cpu().numpy() * 255).astype(np.uint8)
            frames_pil = [Image.fromarray(f, "RGB") for f in np_frames]
        else:
            raise ValueError(f"Unrecognised video frame type: {type(raw)}")

        # Stack into [T, H, W, C] float32 tensor (ComfyUI IMAGE batch)
        frame_tensors = [_comfy_from_pil(f) for f in frames_pil]
        frames_tensor = torch.cat(frame_tensors, dim=0)   # [T, H, W, C]

        gc.collect()
        return (frames_tensor,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "WeeLLMLoaderNode":        WeeLLMLoaderNode,
    "WeeLLMGenerateNode":      WeeLLMGenerateNode,
    "WeeLLMVideoGenerateNode": WeeLLMVideoGenerateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeLLMLoaderNode":        "WeeLLM Loader",
    "WeeLLMGenerateNode":      "WeeLLM Generate",
    "WeeLLMVideoGenerateNode": "WeeLLM Video Generate",
}
