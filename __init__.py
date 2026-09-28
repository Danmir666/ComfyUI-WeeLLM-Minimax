from .install import setup_triton_headers, setup_cuda_alloc_conf, setup_disable_cuda_malloc

setup_triton_headers()
setup_cuda_alloc_conf()
setup_disable_cuda_malloc()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
