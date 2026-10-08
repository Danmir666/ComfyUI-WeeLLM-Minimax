import os
import re
import sys
import gc
import shutil

# Reduces CUDA memory fragmentation — same flag used in native WeeLLM scripts.
# Must be set before torch initialises the CUDA allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import numpy as np
from PIL import Image, ImageOps

import comfy.utils
import comfy.model_management
import folder_paths
from server import PromptServer
from safetensors import safe_open
from safetensors.torch import save_file


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


def _resolve_weights(path, folder):
    """Accept a full path or a bare filename living in ComfyUI/models/<folder>."""
    path = path.strip().strip('"')
    if path and not os.path.exists(path):
        return folder_paths.get_full_path(folder, path) or path
    return path


MINIMAX_H3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minimax_h3")


def _minimax_h3_base_dir():
    """Assemble the config-only MiniMax-H3 folder WeeLLM expects; weights come from ComfyUI model folders."""
    base = os.path.join(folder_paths.get_temp_directory(), "weellm_minimax_h3")
    if os.path.isfile(os.path.join(base, "model_index.json")):
        return base
    shutil.copytree(MINIMAX_H3_DIR, base, dirs_exist_ok=True)
    tokenizer_dir = os.path.join(base, "tokenizer")
    for name in ("text_encoder", "processor"):
        shutil.copytree(tokenizer_dir, os.path.join(base, name), dirs_exist_ok=True)
    processor_dir = os.path.join(base, "processor")
    for name in ("chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        shutil.copy2(os.path.join(base, "text_encoder", name), processor_dir)
    return base


def _diffusers_video_vae(path):
    """The unsloth MiniMax-H3 video VAE uses the original key layout; convert it once to the diffusers layout WeeLLM loads."""
    with safe_open(path, "pt") as f:
        if "decoder.mask_token" not in f.keys():
            return path
        converted = os.path.splitext(path)[0] + "_diffusers.safetensors"
        if os.path.exists(converted):
            return converted
        state = {}
        for key in f.keys():
            if key in ("decoder.mask_token", "latents_mean", "latents_std"):
                continue
            tensor = f.get_tensor(key)
            m = re.fullmatch(r"decoder\.transformer_blocks\.(\d+)\.(attn\.to_qkv|ff\.w1|ff\.w2|attn\.to_out)\.(weight|bias)", key)
            if m:
                prefix, name, suffix = f"decoder.transformer_blocks.{m[1]}.", m[2], m[3]
                if name == "attn.to_qkv":
                    heads_qkv = tensor.unflatten(0, (32, 3, 64))
                    for i, n in enumerate("qkv"):
                        state[f"{prefix}attn.to_{n}.{suffix}"] = heads_qkv[:, i].flatten(0, 1).contiguous()
                elif name == "ff.w1":
                    state[f"{prefix}ff.net.0.proj.{suffix}"] = torch.cat(tensor.chunk(2)[::-1])
                elif name == "ff.w2":
                    state[f"{prefix}ff.net.2.{suffix}"] = tensor
                else:
                    state[f"{prefix}attn.to_out.0.{suffix}"] = tensor
                continue
            key = key.replace("decoder.x_embedder.", "decoder.proj_in.")
            key = re.sub(r"encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", key)
            key = re.sub(r"encoder\.down\.(\d+)\.downsample\.", r"encoder.down_blocks.\1.downsamplers.0.", key)
            state[key.replace(".nin_shortcut.", ".conv_shortcut.")] = tensor
        save_file(state, converted + ".tmp", metadata={"format": "pt"})
    os.replace(converted + ".tmp", converted)
    return converted


def _diffusers_audio_vae(path):
    """The unsloth MiniMax-H3 audio VAE ships fused conv weights; the model expects weight_norm's weight_g/weight_v."""
    with safe_open(path, "pt") as f:
        fused = [k for k in f.keys() if k.startswith(("encoder.", "decoder.")) and k.endswith(".weight") and len(f.get_slice(k).get_shape()) == 3]
        if not fused:
            return path
        converted = os.path.splitext(path)[0] + "_diffusers.safetensors"
        if os.path.exists(converted):
            return converted
        state = {}
        for key in f.keys():
            if key in ("latents_mean", "latents_std"):
                continue
            tensor = f.get_tensor(key)
            if key in fused:
                state[key + "_v"] = tensor
                state[key + "_g"] = torch.linalg.vector_norm(tensor.float(), dim=(1, 2), keepdim=True).to(tensor.dtype)
            else:
                state[key] = tensor
        save_file(state, converted + ".tmp", metadata={"format": "pt"})
    os.replace(converted + ".tmp", converted)
    return converted


def _free_pipeline_cache():
    """Drop ComfyUI's cached node outputs once the prompt ends; WeeLLM pipelines are single-use and keep GBs of host memory."""
    PromptServer.instance.prompt_queue.set_flag("free_memory", True)


def _make_step_callback(pbar: comfy.utils.ProgressBar):
    """Return a diffusers callback_on_step_end that ticks the bar and honours cancel."""
    def _cb(pipeline, step_index, timestep, callback_kwargs):
        pbar.update(1)
        comfy.model_management.throw_exception_if_processing_interrupted()
        return callback_kwargs
    return _cb


class _MultiLoRA:
    """WeeLLM streamers hold a single lora_loader; apply several in sequence through it."""
    def __init__(self, loaders):
        self.loaders = loaders

    def apply_to_module(self, module, shard_name):
        for loader in self.loaders:
            loader.apply_to_module(module, shard_name)


# ---------------------------------------------------------------------------
# Node 0: WeeLLM LoRA (chainable)
# ---------------------------------------------------------------------------

class WeeLLMLoraNode:
    """
    Adds one LoRA from ComfyUI/models/loras to a list. Chain several of these into the Loader's `loras` input.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "loras": ("WEE_LORAS",),
            }
        }

    RETURN_TYPES = ("WEE_LORAS",)
    FUNCTION = "add_lora"
    CATEGORY = "WeeLLM"

    def add_lora(self, lora_name, strength, loras=None):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return ((loras or []) + [(path, strength)],)


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
                "model_path": ("STRING", {"default": "black-forest-labs/FLUX.1-schnell", "tooltip": "HF repo id or local folder. Leave empty with task=video to use the bundled MiniMax-H3 configs (weights from transformer_path, vae_path, audio_vae_path and text_encoder_path)"}),
                "task": (["text-to-image", "image-to-image", "video"], {"default": "text-to-image"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            },
            "optional": {
                "vram_budget": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 128.0, "step": 0.5}),
                "ram_budget": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 256.0, "step": 0.5}),
                "text_encoder_path": ("STRING", {"default": ""}),
                "transformer_path": ("STRING", {"default": ""}),
                "unet_path": ("STRING", {"default": ""}),
                "lora_weights": ("STRING", {"default": ""}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "vae_path": ("STRING", {"default": ""}),
                "audio_vae_path": ("STRING", {"default": ""}),
                "loras": ("WEE_LORAS",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # WeeLLM offloads the text encoder and transformer to meta after each generation; a cached pipeline can't be reused
        return float("nan")

    RETURN_TYPES = ("WEE_PIPELINE",)
    FUNCTION = "load_pipeline"
    CATEGORY = "WeeLLM"

    def load_pipeline(self, model_path, task, dtype, vram_budget=4.0, ram_budget=4.0, 
                      text_encoder_path="", transformer_path="", unet_path="", lora_weights="", lora_scale=1.0,
                      vae_path="", audio_vae_path="", loras=None):
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

        model_path = model_path.strip().strip('"')
        if not model_path:
            if task != "video":
                raise ValueError("Empty model_path is only supported with task=video (bundled MiniMax-H3 configs)")
            model_path = _minimax_h3_base_dir()
        elif (os.path.isabs(model_path) or "\\" in model_path) and not os.path.isdir(model_path):
            raise FileNotFoundError(f"Local model folder not found: '{model_path}'")

        if task == "text-to-image":
            from weellm import WeeTextToImagePipeline as PipelineClass
        elif task == "image-to-image":
            from weellm import WeeImageToImagePipeline as PipelineClass
        elif task == "video":
            from weellm import WeeVideoPipeline as PipelineClass
        else:
            raise ValueError(f"Unknown task: {task}")

        kwargs = {
            "torch_dtype": torch_dtype,
            "device": "cuda",
            "vram_budget": vram_budget,
            "ram_budget": ram_budget,
        }
        
        if text_encoder_path: kwargs["text_encoder_path"] = _resolve_weights(text_encoder_path, "text_encoders")
        if transformer_path: kwargs["transformer_path"] = _resolve_weights(transformer_path, "diffusion_models")
        if unet_path: kwargs["unet_path"] = _resolve_weights(unet_path, "diffusion_models")
        if vae_path: kwargs["vae_path"] = _diffusers_video_vae(_resolve_weights(vae_path, "vae"))
        if audio_vae_path: kwargs["audio_vae_path"] = _diffusers_audio_vae(_resolve_weights(audio_vae_path, "vae"))

        pipe = PipelineClass.from_pretrained(model_path, **kwargs)

        loras = list(loras or [])
        if lora_weights:
            loras.append((_resolve_weights(lora_weights, "loras"), lora_scale))
        if loras:
            from weellm.models.loras.lora_streamer import GenericLazyLoRALoader
            transformer = getattr(pipe._pipeline, "transformer", None) or pipe._pipeline.unet
            transformer._weellm_streamer.lora_loader = _MultiLoRA([GenericLazyLoRALoader(path, scale=strength) for path, strength in loras])
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
                "height":          ("INT",     {"default": 1024, "min": 0, "max": 8192, "step": 8, "tooltip": "0 means auto-infer from input image"}),
                "width":           ("INT",     {"default": 1024, "min": 0, "max": 8192, "step": 8, "tooltip": "0 means auto-infer from input image"}),
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

    def generate(self, pipeline, prompt, height, width, num_steps, guidance_scale, image_guidance_scale, seed, strength, image=None, mask_image=None):

        pipe = pipeline
        pbar = comfy.utils.ProgressBar(num_steps)

        # --- Build generation kwargs ---------------------------------------
        call_kwargs = dict(
            num_inference_steps=num_steps,
            seed=seed,
            callback_on_step_end=_make_step_callback(pbar),
        )

        if height > 0: call_kwargs["height"] = height
        if width > 0: call_kwargs["width"] = width

        if guidance_scale >= 0.0:
            call_kwargs["guidance_scale"] = guidance_scale

        if image_guidance_scale >= 0.0:
            call_kwargs["image_guidance_scale"] = image_guidance_scale

        if image is not None:
            call_kwargs["image"] = _pil_from_comfy(image)
            call_kwargs["strength"] = strength

        if mask_image is not None:
            call_kwargs["mask_image"] = _pil_from_comfy(mask_image)

        # Free ComfyUI-held VRAM before WeeLLM takes over.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

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

        _free_pipeline_cache()
        return (_comfy_from_pil(result),)


# ---------------------------------------------------------------------------
# Node 3: WeeLLM Video Generate  (currently MiniMax-H3 FL2VA)
# ---------------------------------------------------------------------------

class WeeLLMVideoGenerateNode:
    """
    WeeLLM video generation node.

    Outputs a batch of IMAGE frames that you can route to a Video Combine
    node or inspect frame-by-frame. Works with MiniMax-H3, LTX-2.5 and any other
    video model WeeLLM supports.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline":    ("WEE_PIPELINE",),
                "prompt":      ("STRING", {"multiline": True, "default": "A cat walking across a wooden floor"}),
                "height":      ("INT",    {"default": 544, "min": 0, "max": 2048, "step": 8, "tooltip": "0 means auto-infer from first_frame"}),
                "width":       ("INT",    {"default": 960, "min": 0, "max": 2048, "step": 8, "tooltip": "0 means auto-infer from first_frame"}),
                "num_frames":  ("INT",    {"default": 75,   "min": 22, "max": 500}),
                "num_steps":   ("INT",    {"default": 6,    "min": 1,  "max": 100}),
                "seed":        ("INT",    {"default": 0,    "min": 0,  "max": 0xffffffffffffffff}),
            },
            "optional": {
                "first_frame": ("IMAGE",),   # first frame
                "last_frame":  ("IMAGE",),   # last frame (MiniMax FL2VA)
                "frame_rate":  ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "audio_guidance_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1}),
                "stg_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1}),
                "audio_stg_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1}),
                "modality_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1}),
                "audio_modality_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "WeeLLM"

    def generate(self, pipeline, prompt, height, width, num_frames, num_steps, seed,
                 first_frame=None, last_frame=None, frame_rate=24.0, audio_guidance_scale=-1.0,
                 stg_scale=-1.0, audio_stg_scale=-1.0, modality_scale=-1.0, audio_modality_scale=-1.0):

        pipe = pipeline
        generator = torch.Generator("cpu").manual_seed(seed)

        call_kwargs = dict(
            num_frames=num_frames,
            num_inference_steps=num_steps,
            generator=generator,
            frame_rate=frame_rate,
        )

        if height > 0: call_kwargs["height"] = height
        if width > 0: call_kwargs["width"] = width

        if first_frame is not None:
            call_kwargs["image"] = _pil_from_comfy(first_frame)

        if last_frame is not None:
            call_kwargs["last_image"] = _pil_from_comfy(last_frame)
            
        if audio_guidance_scale >= 0: call_kwargs["audio_guidance_scale"] = audio_guidance_scale
        if stg_scale >= 0: call_kwargs["stg_scale"] = stg_scale
        if audio_stg_scale >= 0: call_kwargs["audio_stg_scale"] = audio_stg_scale
        if modality_scale >= 0: call_kwargs["modality_scale"] = modality_scale
        if audio_modality_scale >= 0: call_kwargs["audio_modality_scale"] = audio_modality_scale

        # Free any VRAM ComfyUI is holding for its own models before handing
        # control to WeeLLM. Without this, ComfyUI's internal pool eats into
        # the 4 GB budget and causes OOM during LoRA delta computation.
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        gc.collect()

        out = pipe(prompt, **call_kwargs)

        # WeeVideoResult exposes frames (which is a list of PIL Images or a numpy array)
        if hasattr(out, "frames"):
            raw = out.frames
        elif hasattr(out, "videos") and out.videos is not None:
            raw = out.videos[0] if isinstance(out.videos, list) else out.videos
        elif isinstance(out, dict):
            raw = out.get("videos") or out.get("video")
            if isinstance(raw, list):
                raw = raw[0]
        else:
            raw = None

        if raw is None:
            raise ValueError("WeeVideoPipeline returned no video frames.")

        frames_pil = []
        if isinstance(raw, list):
            # List of PIL Images or numpy arrays
            for f in raw:
                if hasattr(f, "convert"):
                    frames_pil.append(f.convert("RGB"))
                elif isinstance(f, np.ndarray):
                    frames_pil.append(Image.fromarray(f, "RGB"))
                else:
                    frames_pil.append(f)
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
        _free_pipeline_cache()
        return (frames_tensor,)



# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "WeeLLMLoraNode":          WeeLLMLoraNode,
    "WeeLLMLoaderNode":        WeeLLMLoaderNode,
    "WeeLLMGenerateNode":      WeeLLMGenerateNode,
    "WeeLLMVideoGenerateNode": WeeLLMVideoGenerateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeLLMLoraNode":          "WeeLLM LoRA",
    "WeeLLMLoaderNode":        "WeeLLM Loader",
    "WeeLLMGenerateNode":      "WeeLLM Generate",
    "WeeLLMVideoGenerateNode": "WeeLLM Video Generate",
}
