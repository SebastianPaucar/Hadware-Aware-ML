import time
import numpy as np
import pynvml
import tensorflow as tf


class HardwareLogger(tf.keras.callbacks.Callback):
    """
    Keras callback that tracks GPU power, memory, utilization
    and epoch time per epoch. 
    """

    def __init__(self, device_id=0):
        super().__init__()
        pynvml.nvmlInit()
        self.handle  = pynvml.nvmlDeviceGetHandleByIndex(device_id)
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
        mem     = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        util    = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        power   = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000

        # Store in callback history
        self.history["epoch_time_s"].append(elapsed)
        self.history["gpu_util_pct"].append(util.gpu)
        self.history["gpu_mem_used_mb"].append(mem.used / 1024**2)
        self.history["gpu_power_w"].append(power)

        # Inject into Keras logs so it appears in model.history
        # and can be stored by utilities.store_axo
        if logs is not None:
            logs["epoch_time_s"]    = elapsed
            logs["gpu_util_pct"]    = util.gpu
            logs["gpu_mem_used_mb"] = mem.used / 1024**2
            logs["gpu_power_w"]     = power

        print(f"\n  [HW] time={elapsed:.2f}s | "
              f"GPU={util.gpu}% | "
              f"mem={mem.used/1024**2:.0f}MB | "
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
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
