"""Checks that PyTorch can really use your NVIDIA GPU (important on RTX 50-series / Blackwell cards,
which need a PyTorch build made for CUDA 12.8 or newer).   Run:  python scripts/check_gpu.py"""
import sys

try:
    import torch
except ImportError:
    print("PyTorch is not installed. Run setup.bat first.")
    sys.exit(1)

print(f"PyTorch {torch.__version__}  |  built for CUDA {torch.version.cuda}")

if not torch.cuda.is_available():
    print("\nCUDA is NOT available to PyTorch.")
    print("- Update your NVIDIA driver (570 or newer for RTX 50-series), then re-run this check.")
    print("- Make sure the CUDA build of PyTorch is installed (not the CPU build):")
    print("    pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
print(f"GPU: {name}  |  compute capability {major}.{minor} ({arch})  |  {total_gb:.1f} GB memory")

arches = torch.cuda.get_arch_list()
if arch not in arches:
    print(f"\nWARNING: this PyTorch build lists {arches} and does not include {arch}.")
    print("You will probably see 'no kernel image is available'. Reinstall with the cu128 (or newer) index:")
    print("    pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
y = (x @ x).float().sum().item()
print("GPU math test passed." if y == y else "GPU math test returned NaN (unexpected).")
print(f"bfloat16 supported: {torch.cuda.is_bf16_supported()}")
if total_gb < 7.5:
    print("Note: less than 8 GB of GPU memory. Use clips.backend: kenburns or lower clips.long_side.")
print("\nAll good.")
