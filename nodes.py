import os
import re
import inspect
import logging
import sys
import gc
import shutil
import textwrap

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
from huggingface_hub import constants as hf_constants

# Keep Hugging Face downloads in ComfyUI's models/diffusers instead of the user profile, unless HF_HUB_CACHE is set.
if "HF_HUB_CACHE" not in os.environ:
    hf_constants.HF_HUB_CACHE = folder_paths.get_folder_paths("diffusers")[0]


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


def _release_gpu(pipeline=None):
    """Hand the GPU back before the next node: ComfyUI's models do not know about WeeLLM's tensors, so an upscaler loading after us would otherwise compete with them."""
    if pipeline is not None:
        from weellm.io.memory import evict_module
        transformer = pipeline._pipeline.transformer
        evict_module(transformer)
        transformer._weellm_streamer._resident_loaded = False
    gc.collect()
    comfy.model_management.soft_empty_cache()


# Filled by the patched VAE decode while a video generation runs: the normalized video latent (ComfyUI's H3 LATENT convention) and whether to skip decoding it
_decode_capture = {"active": False, "skip": False, "latent": None}


def _patch_vae_decode():
    """MiniMax VAE decode on 6 GB cards: size micro-batches for the real per-tile cost (~14x, WeeLLM assumes ~5x) and stitch the decoded tiles in RAM instead of VRAM."""
    from weellm.models.vaes.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3Streamer
    module = sys.modules[AutoencoderKLMiniMaxH3Streamer.__module__]
    if getattr(AutoencoderKLMiniMaxH3Streamer, "_decode_patched", False):
        return
    source = textwrap.dedent(inspect.getsource(AutoencoderKLMiniMaxH3Streamer._stream_decode))
    gpu_tiles = "t.unsqueeze(0).to(self.device, non_blocking=True) for t in all_hs.unbind(dim=0)"
    if gpu_tiles not in source:
        logging.warning("WeeLLM's VAE decode changed; decoded tiles will be stitched in VRAM.")
        stream_decode = AutoencoderKLMiniMaxH3Streamer._stream_decode
    else:
        namespace = {}
        exec(source.replace(gpu_tiles, "t.unsqueeze(0) for t in all_hs.unbind(dim=0)"), vars(module), namespace)
        stream_decode = namespace["_stream_decode"]

    def _stream_decode(self, z):
        mem_get_info = torch.cuda.mem_get_info
        reserve = 0.6e9

        def capped(*args, **kwargs):
            free, total = mem_get_info(*args, **kwargs)
            return reserve + max(0, free - reserve) * 5 / 14, total

        torch.cuda.mem_get_info = capped
        try:
            return stream_decode(self, z)
        finally:
            torch.cuda.mem_get_info = mem_get_info

    decode = AutoencoderKLMiniMaxH3Streamer.decode

    def capture_decode(self, latents, return_dict=True, **kwargs):
        if _decode_capture["active"] and not self.is_audio:
            mean = torch.tensor(self.model.config.latents_mean).view(1, -1, 1, 1, 1)
            std = torch.tensor(self.model.config.latents_std).view(1, -1, 1, 1, 1)
            _decode_capture["latent"] = (latents.detach().float().cpu() - mean) / std
            if _decode_capture["skip"]:
                placeholder = torch.zeros(1, 3, 1, 16, 16, device=latents.device)
                return placeholder if return_dict else (placeholder,)
        return decode(self, latents, return_dict, **kwargs)

    AutoencoderKLMiniMaxH3Streamer._stream_decode = _stream_decode
    AutoencoderKLMiniMaxH3Streamer.decode = capture_decode
    AutoencoderKLMiniMaxH3Streamer._decode_patched = True


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


class _PrunedAdalnLoRA:
    """Pruned H3 bases take an 8-dim table row in adaln_proj.linear, but LoRAs such as the Turbo one carry the adaln delta in the 2688-dim silu(t_emb) space. Apply the other tensors through the wrapped loader and add B @ A @ silu(t_emb) to those projections at run time, reading silu(t_emb) from the grid bundled with ComfyUI-MiniMax-H3-Turbo."""
    GRID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ComfyUI-MiniMax-H3-Turbo", "h3_silu_temb_grid.safetensors")
    SUFFIXES = ("adaln_proj.linear.weight", "norm_out.linear.weight")
    _grid = None

    def __init__(self, loader, embedder):
        self.loader = loader
        self.embedder = embedder
        self._cached = None
        self.pairs = [p for p in loader.lora_pairs if any(t[0].endswith(self.SUFFIXES) for t in p[2])]
        loader.lora_pairs = [p for p in loader.lora_pairs if p not in self.pairs]
        if self.pairs and _PrunedAdalnLoRA._grid is None:
            if not os.path.isfile(self.GRID_PATH):
                raise FileNotFoundError(f"LoRA has adaln weights for a pruned MiniMax-H3 but '{self.GRID_PATH}' is missing; install ComfyUI-MiniMax-H3-Turbo")
            _PrunedAdalnLoRA._grid = comfy.utils.load_torch_file(self.GRID_PATH)["silu_t_emb_grid"].float()

    def _silu_temb(self, device):
        raw_t = self.embedder._raw_t
        if self._cached is None or self._cached[0] is not raw_t:
            t = raw_t.float().clamp(0.0, 1.0).cpu() * 1024.0
            lo = t.long().clamp(0, 1023)
            grid = _PrunedAdalnLoRA._grid
            self._cached = (raw_t, torch.lerp(grid[lo], grid[lo + 1], (t - lo.float()).unsqueeze(-1)).to(device))
        return self._cached[1]

    def _patch(self, lin, key):
        """Add this LoRA's delta to lin's forward; deltas of several LoRAs on the same projection accumulate."""
        if not hasattr(lin, "_adaln_deltas"):
            lin._adaln_deltas = {}
            base_forward = lin.forward

            def forward(x):
                out = base_forward(x)
                st = self._silu_temb(x.device)
                for a, b in lin._adaln_deltas.values():
                    out = out + (b.to(x.device, out.dtype) @ (a.to(x.device, out.dtype) @ st.to(out.dtype).T)).T
                return out
            lin.forward = forward
        lin._adaln_deltas[id(self)] = key

    def _factors(self, f, a_key, b_key):
        a = self.loader._read_tensor(f, a_key, "cpu")
        b = self.loader._read_tensor(f, b_key, "cpu")
        scale = self.loader.scale
        alpha_key = a_key.rsplit(".lora_A", 1)[0] + ".alpha"
        if alpha_key in self.loader._header:
            scale *= self.loader._read_tensor(f, alpha_key, "cpu").item() / a.size(0)
        return a.bfloat16(), (b * scale).bfloat16()

    def _apply(self, root, prefix):
        with open(self.loader.lora_path, "rb") as f:
            for a_key, b_key, targets in self.pairs:
                for t_key, _ in targets:
                    if t_key.startswith(prefix):
                        lin = root.get_submodule(t_key[len(prefix):].removesuffix(".weight"))
                        self._patch(lin, self._factors(f, a_key, b_key))

    def apply_to_module(self, module, shard_name):
        self.loader.apply_to_module(module, shard_name)
        self._apply(module, shard_name.removeprefix("transformer.") + ".")

    def patch_resident(self, transformer):
        self._apply(transformer.norm_out, "norm_out.")


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
            _patch_vae_decode()
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
            loaders = [GenericLazyLoRALoader(path, scale=strength) for path, strength in loras]
            if hasattr(getattr(transformer, "time_embedder", None), "table"):
                loaders = [_PrunedAdalnLoRA(loader, transformer.time_embedder) for loader in loaders]
                for loader in loaders:
                    loader.patch_resident(transformer)
            transformer._weellm_streamer.lora_loader = _MultiLoRA(loaders)
        return (pipe,)


# ---------------------------------------------------------------------------
# Node 2: WeeLLM Generate (Text-to-Image + Image-to-Image + Inpainting)
# ---------------------------------------------------------------------------

MAX_REFERENCE_IMAGES = 10


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
                **{f"image_{i}": ("IMAGE", {"tooltip": "Extra reference image (models that accept several, e.g. Qwen Image 2.1 edit)"}) for i in range(2, MAX_REFERENCE_IMAGES + 1)},
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "WeeLLM"

    def generate(self, pipeline, prompt, height, width, num_steps, guidance_scale, image_guidance_scale, seed, strength, image=None, mask_image=None, **extra_images):

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
            references = [_pil_from_comfy(image)] + [_pil_from_comfy(extra_images[f"image_{i}"]) for i in range(2, MAX_REFERENCE_IMAGES + 1) if extra_images.get(f"image_{i}") is not None]
            call_kwargs["image"] = references if len(references) > 1 else references[0]
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
                "decode_frames": ("BOOLEAN", {"default": True, "tooltip": "Chooses ONE output, never both (outputting both can run out of memory).\nOn: decodes the video. Use frames (and audio) to preview or save it; latent stays empty.\nOff: skips the slow VAE decode. Use latent with the H3 Latent Upscaler and WeeLLM Latent Decode to scale the video; frames is then a 1-frame placeholder.\nSwitching modes runs the generation again (same seed, same prompt)."}),
                "audio_guidance_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1}),
                "stg_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1}),
                "audio_stg_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1}),
                "modality_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1}),
                "audio_modality_scale": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "LATENT")
    RETURN_NAMES = ("frames", "audio", "latent")
    FUNCTION = "generate"
    CATEGORY = "WeeLLM"

    def generate(self, pipeline, prompt, height, width, num_frames, num_steps, seed,
                 first_frame=None, last_frame=None, frame_rate=24.0, audio_guidance_scale=-1.0,
                 stg_scale=-1.0, audio_stg_scale=-1.0, modality_scale=-1.0, audio_modality_scale=-1.0,
                 decode_frames=True):

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

        _decode_capture.update(active=not decode_frames, skip=True, latent=None)
        try:
            out = pipe(prompt, **call_kwargs)
        except BaseException:
            _free_pipeline_cache()
            raise
        finally:
            _decode_capture["active"] = False
        latent = _decode_capture["latent"]
        if latent is None and not decode_frames:
            raise ValueError("No video latent was captured; latent output needs the bundled MiniMax-H3 setup (empty model_path)")

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

        waveform, sample_rate = getattr(out, "audio", None), getattr(out, "audio_rate", 32000)
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.float().cpu()
            if waveform.ndim < 3:
                waveform = waveform.reshape(1, -1, waveform.shape[-1])
        else:
            logging.warning("WeeLLM returned no audio; the audio output is silent.")
            waveform = torch.zeros(1, 2, int(sample_rate * frames_tensor.shape[0] / frame_rate))

        _release_gpu(pipe)
        _free_pipeline_cache()
        return (frames_tensor, {"waveform": waveform, "sample_rate": sample_rate}, None if latent is None else {"samples": latent})



# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Node 4: WeeLLM Latent Decode (MiniMax-H3)
# ---------------------------------------------------------------------------

class WeeLLMLatentDecodeNode:
    """
    Decodes a MiniMax-H3 video latent (for example the output of the H3 Latent Upscaler) with the loaded
    pipeline's streamed VAE. Connect the same pipeline that produced the latent.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("WEE_PIPELINE",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "decode"
    CATEGORY = "WeeLLM"

    def decode(self, pipeline, latent):
        comfy.model_management.unload_all_models()
        _release_gpu(pipeline)

        underlying = pipeline._pipeline
        streamer = underlying.vae_streamer
        config = streamer.model.config
        mean = torch.tensor(config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(config.latents_std).view(1, -1, 1, 1, 1)
        with torch.no_grad():
            video = streamer._stream_decode((latent["samples"].float().cpu() * std + mean).to(streamer.device, streamer.dtype))
        pixel_mean = torch.tensor(underlying.pixel_mean).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(underlying.pixel_std).view(1, -1, 1, 1, 1)
        frames = (video.float().cpu() * pixel_std + pixel_mean).clamp(0, 1)[0].permute(1, 2, 3, 0).contiguous()

        _release_gpu()
        _free_pipeline_cache()
        return (frames,)


NODE_CLASS_MAPPINGS = {
    "WeeLLMLoraNode":          WeeLLMLoraNode,
    "WeeLLMLoaderNode":        WeeLLMLoaderNode,
    "WeeLLMGenerateNode":      WeeLLMGenerateNode,
    "WeeLLMVideoGenerateNode": WeeLLMVideoGenerateNode,
    "WeeLLMLatentDecodeNode":  WeeLLMLatentDecodeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WeeLLMLoraNode":          "WeeLLM LoRA",
    "WeeLLMLoaderNode":        "WeeLLM Loader",
    "WeeLLMGenerateNode":      "WeeLLM Generate",
    "WeeLLMVideoGenerateNode": "WeeLLM Video Generate",
    "WeeLLMLatentDecodeNode":  "WeeLLM Latent Decode",
}
