import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import importlib.metadata
import importlib
import ctypes
import subprocess
import sys

def check_import(module_name, pip_name=None):
    pip_name = pip_name or module_name
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__',
                  importlib.metadata.version(pip_name))
        print(f"  OK  {pip_name:<35} {version}")
        return mod
    except ImportError as e:
        print(f"  FAIL {pip_name:<35} ImportError: {e}")
        return None
    except Exception as e:
        print(f"  WARN {pip_name:<35} {e}")
        return None

def check_cudnn():
    try:
        lib = ctypes.cdll.LoadLibrary('libcudnn.so.9')
        ver = lib.cudnnGetVersion()
        major = ver // 10000
        minor = (ver % 10000) // 100
        patch = ver % 100
        print(f"  OK  {'libcudnn.so.9':<35} {major}.{minor}.{patch} (raw: {ver})")
        return ver
    except Exception as e:
        print(f"  FAIL {'libcudnn.so.9':<35} {e}")
        return None

def check_nvidia_smi():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  OK  {'nvidia-smi':<35} {result.stdout.strip()}")
        else:
            print(f"  FAIL nvidia-smi failed")
    except Exception as e:
        print(f"  FAIL nvidia-smi: {e}")

def check_onnxruntime_gpu():
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_cuda = 'CUDAExecutionProvider' in providers
        status = "OK  " if has_cuda else "WARN"
        print(f"  {status} {'onnxruntime-gpu':<35} {ort.__version__} | providers: {providers}")
        return ort
    except Exception as e:
        print(f"  FAIL {'onnxruntime-gpu':<35} {e}")
        return None

def check_pynvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        print(f"  OK  {'pynvml':<35} {importlib.metadata.version('pynvml')}")
        print(f"       GPU: {name}")
        print(f"       Memory: {mem.used/1024**2:.0f} / {mem.total/1024**2:.0f} MB")
        print(f"       Utilization: {util.gpu}% GPU, {util.memory}% Memory")
        print(f"       Power: {power:.1f} W")
        pynvml.nvmlShutdown()
        return True
    except Exception as e:
        print(f"  FAIL {'pynvml':<35} {e}")
        return False

def check_nvtx():
    try:
        import nvtx
        with nvtx.annotate("test_range", color="green"):
            x = sum(range(1000))
        print(f"  OK  {'nvtx':<35} {importlib.metadata.version('nvtx')} | annotation works")
        return True
    except Exception as e:
        print(f"  FAIL {'nvtx':<35} {e}")
        return False

def check_tf2onnx():
    try:
        import tf2onnx
        print(f"  OK  {'tf2onnx':<35} {tf2onnx.__version__}")
        return True
    except Exception as e:
        print(f"  FAIL {'tf2onnx':<35} {e}")
        return False

def check_tensorrt():
    try:
        import tensorrt as trt
        print(f"  OK  {'tensorrt':<35} {trt.__version__}")
        return True
    except ImportError:
        print(f"  WARN {'tensorrt':<35} not installed (optional)")
        return False
    except Exception as e:
        print(f"  FAIL {'tensorrt':<35} {e}")
        return False

def check_cuda_python():
    try:
        import cuda
        print(f"  OK  {'cuda-python':<35} {importlib.metadata.version('cuda-python')}")
        return True
    except ImportError:
        print(f"  WARN {'cuda-python':<35} not installed (optional)")
        return False

def check_cuda_python_old():
    try:
        from cuda import cuda, nvrtc
        print(f"  OK  {'cuda-python':<35} {importlib.metadata.version('cuda-python')}")
        return True
    except ImportError:
        print(f"  WARN {'cuda-python':<35} not installed (optional)")
        return False
    except Exception as e:
        print(f"  FAIL {'cuda-python':<35} {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("ML TOOLS BACKEND VERIFICATION")
print("="*60)

print("\n[Core ML Frameworks]")
check_import('keras')
check_import('tensorflow')
check_import('torch')
check_import('jax')

print("\n[CUDA / cuDNN]")
check_cudnn()
check_import('nvidia.cudnn', 'nvidia-cudnn-cu12')
check_nvidia_smi()

print("\n[ONNX Stack]")
check_import('onnx')
check_onnxruntime_gpu()
check_tf2onnx()

print("\n[TF Optimization]")
check_import('tensorflow_model_optimization', 'tensorflow-model-optimization')

print("\n[GPU Monitoring]")
check_pynvml()
check_import('GPUtil', 'gputil')
check_import('psutil')
check_import('cpuinfo', 'py-cpuinfo')

print("\n[NVIDIA Profiling Tools]")
check_nvtx()
check_cuda_python()

print("\n[TensorRT (optional)]")
check_tensorrt()

print("\n[pip package versions]")
packages = [
    'nvidia-cudnn-cu12', 'jaxlib', 'jax-cuda12-plugin', 'keras',
    'onnx', 'onnxruntime-gpu', 'tf2onnx',
    'tensorflow-model-optimization', 'pynvml', 'nvtx',
    'gputil', 'psutil', 'py-cpuinfo', 'cuda-python', 'tensorrt'
]
for pkg in packages:
    try:
        print(f"  {pkg:<40} {importlib.metadata.version(pkg)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"  {pkg:<40} not installed")

print("\n" + "="*60)
print("Verification complete.")
print("="*60 + "\n")
