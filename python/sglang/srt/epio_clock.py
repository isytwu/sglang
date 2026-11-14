"""Anchor the GPU wall clock (wall_clock64, the counter the MoRI kernel profiler stamps) to
host time: launch a 1-thread kernel that reads it, on a high-priority stream, and bracket the
launch with time.time(). Compiled once with hiprtc through ctypes (no build step).

  gpu_wall_clock_anchor() -> {"host_before", "host_after", "gpu_ticks", "freq_hz", "err_s"}
  host_time = host_mid + (ticks - gpu_ticks) / freq_hz
"""

import ctypes
import threading
import time

_SRC = b"""
extern "C" __global__ void epio_wall_clock(unsigned long long* out) {
  out[0] = wall_clock64();
}
"""
_lock = threading.Lock()
_state = {}


def _load():
    import torch

    dev = torch.cuda.current_device()
    if dev in _state:
        return _state[dev]
    hip = ctypes.CDLL("libamdhip64.so")
    rtc = ctypes.CDLL("libhiprtc.so")
    prog = ctypes.c_void_p()
    assert rtc.hiprtcCreateProgram(ctypes.byref(prog), _SRC, b"epio_wall_clock.hip", 0, None, None) == 0
    arch = torch.cuda.get_device_properties(dev).gcnArchName.split(":")[0]
    opts = (ctypes.c_char_p * 1)(f"--offload-arch={arch}".encode())
    rc = rtc.hiprtcCompileProgram(prog, 1, opts)
    if rc != 0:
        n = ctypes.c_size_t()
        rtc.hiprtcGetProgramLogSize(prog, ctypes.byref(n))
        log = ctypes.create_string_buffer(n.value)
        rtc.hiprtcGetProgramLog(prog, log)
        raise RuntimeError(f"hiprtc compile failed: {log.value.decode(errors='replace')}")
    n = ctypes.c_size_t()
    rtc.hiprtcGetCodeSize(prog, ctypes.byref(n))
    code = ctypes.create_string_buffer(n.value)
    rtc.hiprtcGetCode(prog, code)
    mod = ctypes.c_void_p()
    assert hip.hipModuleLoadData(ctypes.byref(mod), code) == 0
    fn = ctypes.c_void_p()
    assert hip.hipModuleGetFunction(ctypes.byref(fn), mod, b"epio_wall_clock") == 0
    out = torch.zeros(1, dtype=torch.int64, device=f"cuda:{dev}")
    stream = torch.cuda.Stream(device=dev, priority=-1)
    st = dict(hip=hip, fn=fn, out=out, stream=stream, code=code, mod=mod, freq_hz=100e6)
    try:
        import mori.cpp as mc

        st["freq_hz"] = mc.get_cur_device_wall_clock_freq_mhz() * 1e3  # returns kHz (100000)
    except Exception:  # noqa: BLE001
        pass
    _state[dev] = st
    return st


def gpu_wall_clock_anchor(n=5):
    """Best (smallest bracket) of n launches."""
    with _lock:
        st = _load()
        hip = st["hip"]
        best = None
        for _ in range(n):
            arg = ctypes.c_void_p(st["out"].data_ptr())
            params = (ctypes.c_void_p * 1)(ctypes.addressof(arg))
            sh = ctypes.c_void_p(st["stream"].cuda_stream)
            t0 = time.time()
            rc = hip.hipModuleLaunchKernel(st["fn"], 1, 1, 1, 1, 1, 1, 0, sh, params, None)
            hip.hipStreamSynchronize(sh)
            t1 = time.time()
            if rc != 0:
                raise RuntimeError(f"hipModuleLaunchKernel rc={rc}")
            ticks = int(st["out"].item())
            if best is None or (t1 - t0) < (best["host_after"] - best["host_before"]):
                best = dict(host_before=t0, host_after=t1, gpu_ticks=ticks, freq_hz=st["freq_hz"], err_s=(t1 - t0) / 2)
        return best
