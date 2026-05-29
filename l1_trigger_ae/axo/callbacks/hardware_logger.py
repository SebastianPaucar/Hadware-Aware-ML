import time
import numpy as np
import pynvml
import tensorflow as tf

try:
    from pynvml import (nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
                        nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates,
                        nvmlDeviceGetPowerUsage, NVMLError)
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


def _safe_nvml(fn, fallback=0.0):
    try:
        return fn()
    except Exception:
        return fallback


class HardwareLogger(tf.keras.callbacks.Callback):
    """
    Keras callback that tracks GPU power, memory, utilization
    and epoch time per epoch. 
    """

    def __init__(self, device_id=0):
        super().__init__()
        self.handle = None
        if NVML_AVAILABLE:
            try:
                nvmlInit()
                self.handle = nvmlDeviceGetHandleByIndex(device_id)
            except Exception as e:
                print(f"[HardwareLogger] NVML init failed: {e}. Hardware logging disabled.")


        self.t_start = None
        self.history = {
            "epoch_time_s":    [],
            "gpu_util_pct":    [],
            "gpu_mem_used_mb": [],
            "gpu_power_w":     [],
        }

    def on_epoch_begin(self, epoch, logs=None):
        self.t_start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.perf_counter() - self.t_start

        if self.handle is not None:
            mem   = _safe_nvml(lambda: nvmlDeviceGetMemoryInfo(self.handle))
            util  = _safe_nvml(lambda: nvmlDeviceGetUtilizationRates(self.handle))
            power = _safe_nvml(lambda: nvmlDeviceGetPowerUsage(self.handle) / 1000)
            mem_mb  = mem.used / 1024**2 if hasattr(mem, 'used') else 0.0
            gpu_pct = util.gpu if hasattr(util, 'gpu') else 0.0
        else:
            mem_mb = gpu_pct = power = 0.0

        # Store in callback history
        self.history["epoch_time_s"].append(elapsed)
        self.history["gpu_util_pct"].append(gpu_pct)
        self.history["gpu_mem_used_mb"].append(mem_mb)
        self.history["gpu_power_w"].append(power)

        # Inject into Keras logs so it appears in model.history
        # and can be stored by utilities.store_axo
        if logs is not None:
            logs["epoch_time_s"]    = elapsed
            logs["gpu_util_pct"]    = gpu_pct
            logs["gpu_mem_used_mb"] = mem_mb
            logs["gpu_power_w"]     = power

        print(f"\n  [HW] time={elapsed:.2f}s | "
              f"GPU={gpu_pct}% | "
              f"mem={mem_mb:.0f}MB | "
              f"power={power:.1f}W")


    def on_train_end(self, logs=None):
        times  = self.history["epoch_time_s"]
        utils  = self.history["gpu_util_pct"]
        powers = self.history["gpu_power_w"]
        mems   = self.history["gpu_mem_used_mb"]

        print("\n" + "="*60)
        print("HARDWARE METRICS SUMMARY")
        print("="*60)
        print(f"  Avg epoch time : {np.mean(times):.2f}s")
        print(f"  Avg GPU util   : {np.mean(utils):.1f}%")
        print(f"  Avg power      : {np.mean(powers):.1f}W")
        print(f"  Peak power     : {max(powers):.1f}W")
        print(f"  Peak GPU memory: {max(mems):.0f}MB")
        print("="*60)

    def __del__(self):
        if NVML_AVAILABLE:
            try:
                nvmlShutdown()
            except Exception:
                pass
