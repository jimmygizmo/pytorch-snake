# monitor.py  -  Importable module for usage by the ML training agent.
# Monitor resource usage by the GP. Pretty much only for NVidia GPUs at this time.

import torch


# ###############################################    CONFIGURATION    ##################################################

MONITOR: bool = False

# #############################################    FUNCTION DEFINITIONS    #############################################

def hardware_gpu_check():  # Specific to CUDA/NVidia
    if not MONITOR: return
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    cuda_devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    for device in cuda_devices:
        print(f"CUDA device name: [{torch.cuda.get_device_name(device)}]")
# TODO: Check out this module for testing GPU:
# https://pypi.org/project/test-pytorch-gpu/
# UPDATE: Ran fine on my machine. My GPU is: NVIDIA RTX A5000 Laptop GPU


def gpu_resource_sample():  # Specific to CUDA/NVidia
    if not MONITOR: return
    memory_allocated = round(torch.cuda.memory_allocated(0)/1024**3,1)
      # Returns the current GPU memory usage by tensors in bytes. (Device 0)
    memory_reserved = round(torch.cuda.memory_reserved(0)/1024**3,1)
    print(f"GPU Memory:    Allocated: {memory_allocated} GB    Reserved: {memory_reserved} GB")
    # OTHER EXPERIMENTAL STATS BELOW HERE:
    max_memory_reserved = round(torch.cuda.max_memory_reserved(0)/1024**3,1)
    print(f"GPU Memory:    Max Reserved: {max_memory_reserved} GB")
      # Returns the maximum GPU memory managed by the caching allocator in bytes for a given device.
# end def gpu_resource_sample()  -  # TODO: Troubleshoot all three memory values always 0.0.
# Othere tools can briefly see a small amount of allocated usage. Perhaps this programs just uses no GPU memory?
#   No I have a feeling maybe the values just exist for a very short amount of time or there is some issue with the
#   monitoring libs/calls. Not sure yet. I do want robust resource monitoring however, especially for other projects.
# NOTE: 'reserved' used to be called 'cached' everywhere in older CUDA stuff.


##
#

# TODO: Later this module might support various GPUs or even CPU resource usage, by
#   auto-detecting hardware or being configurable.


##
#
