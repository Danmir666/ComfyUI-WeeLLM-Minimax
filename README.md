<div align="center">
	<img src="docs/banner.png" alt="ComfyUI-WeeLLM" width="100%" style="max-width: 1200px;" />
	<br />
	<a href="https://camo.githubusercontent.com/5e3ae50ed74ba1fea6f7b2002fe729b6dac58900dde6ba310eaa26b56dd071dd/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374617475732d4163746976652d737563636573732e737667"><img src="https://camo.githubusercontent.com/5e3ae50ed74ba1fea6f7b2002fe729b6dac58900dde6ba310eaa26b56dd071dd/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374617475732d4163746976652d737563636573732e737667" alt="Status: Active" /></a>
	<a href="https://camo.githubusercontent.com/7013272bd27ece47364536a221edb554cd69683b68a46fc0ee96881174c4214c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667"><img src="https://camo.githubusercontent.com/7013272bd27ece47364536a221edb554cd69683b68a46fc0ee96881174c4214c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c6963656e73652d4d49542d626c75652e737667" alt="License: MIT" /></a>
	<a href="https://camo.githubusercontent.com/dd0b24c1e6776719edb2c273548a510d6490d8d25269a043dfabbd38419905da/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5052732d77656c636f6d652d627269676874677265656e2e737667"><img src="https://camo.githubusercontent.com/dd0b24c1e6776719edb2c273548a510d6490d8d25269a043dfabbd38419905da/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5052732d77656c636f6d652d627269676874677265656e2e737667" alt="PRs Welcome" /></a>
</div>

A native ComfyUI custom node wrapper for [WeeLLM](https://github.com/Jit-Roy/weellm) — bringing ultra-low VRAM layer-streaming inference directly into your Comfy workflows.

With ComfyUI-WeeLLM, you can run massive diffusion models (like FLUX, Stable Diffusion, Stable Diffusion XL, Z image, Krea2, Qwen Image, Ltx2.5 and MiniMax-H3) on GPUs with **less than 4GB of VRAM**, without requiring to quantize. 

- GGUFs and Safetensors - Everything is Supported.
- LoRa Support Available.

## Workflows

### Workflow 1: Image to Image

This workflow uses an input image as the starting point and generates a revised
image according to the prompt and `strength` value.

![Workflow 1: Image to Image](docs/workflow1.png)

[Download the Image to Image workflow](workflows/Weellm%20Image%20to%20Image.json)

1. Import the workflow into ComfyUI and load an input image.
2. In **WeeLLM Loader**, set `model_path` to a supported Hugging Face model ID
	or local model directory and set `task` to `image-to-image`.
3. Connect the input image to the `image` input on **WeeLLM Generate**.
4. Adjust `strength`, then queue the prompt.

### Workflow 2: Image to Video

This workflow uses an input image as the first frame of a generated video.
The resulting frame batch can be sent to a Video Combine node.

![Workflow 2: Image to Video](docs/workflow2.png)

[Download the Image to Video workflow](workflows/Weellm%20Video.json)

1. Import the workflow into ComfyUI and load an input image.
2. In **WeeLLM Loader**, set `task` to `video`.
3. Connect the input image to the `first_frame` input on **WeeLLM Video Generate**.
4. Connect the generated frames to a Video Combine node, then queue the prompt.


## Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open the ComfyUI Manager.
2. Click **Install Custom Nodes**.
3. Search for `ComfyUI-WeeLLM` and click Install.
4. Restart ComfyUI.

### Method 2: Manual Git Clone
Navigate to your `ComfyUI/custom_nodes/` directory and run:
```bash
git clone https://github.com/Jit-Roy/ComfyUI-WeeLLM.git
cd ComfyUI-WeeLLM
pip install -r requirements.txt
```

## Nodes

### 1. WeeLLM Loader
Responsible for configuring and caching the model pipeline. 
- **`model_path`**: Provide the Hugging Face repo ID (e.g. `black-forest-labs/FLUX.1-schnell`) or a local directory.
- **`task`**: Select the type of pipeline to initialize (`text-to-image`, `image-to-image`, `video`).
- **`dtype`**: The precision to use (e.g., `bfloat16`).

### 2. WeeLLM Generate
Generates images using the loaded pipeline.
- Connect the `WEE_PIPELINE` from the Loader to this node.
- Supports optional **`image`** and **`mask_image`** inputs for automatic Image-to-Image and Inpainting.

### 3. WeeLLM Video Generate
Generates video batches for models like MiniMax-H3 FL2VA.
- Connect the `WEE_PIPELINE` from the Loader (with `video` task selected) to this node.
- Outputs an `IMAGE` batch that can be sent directly to Video Combine nodes.
