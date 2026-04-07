# import torch
# import numpy as np
# import scipy.fft as cpu_fft
# import time
#
# # Your specific grid size
# N = 2**12
# z_steps = 2**9
# iterations = z_steps
#
# print(f"Benchmarking 1D FFTs for array size {N} over {iterations} iterations...")
#
# # --- 1. SciPy CPU Benchmark (Multi-threaded) ---
# E_cpu = np.random.rand(N) + 1j * np.random.rand(N)
# operator_cpu = np.exp(1j * np.random.rand(N))
#
# start_time = time.time()
# for _ in range(iterations):
#     # Notice the workers=-1, this forces SciPy to use all M1 CPU cores
#     spectrum = cpu_fft.fft(E_cpu, workers=-1)
#     spectrum *= operator_cpu
#     E_cpu = cpu_fft.ifft(spectrum, workers=-1)
#     E_cpu *= np.abs(E_cpu)**2
#
# cpu_time = time.time() - start_time
# print(f"M1 CPU Time (SciPy): {cpu_time:.4f} seconds")
#
#
# # --- 2. PyTorch GPU Benchmark (MPS) ---
# if not torch.backends.mps.is_available():
#     print("MPS device not found. Ensure PyTorch is installed correctly for Apple Silicon.")
# else:
#     mps_device = torch.device("mps")
#
#     # Move arrays to the M1 GPU
#     E_gpu = torch.tensor(E_cpu, device=mps_device)
#     operator_gpu = torch.tensor(operator_cpu, device=mps_device)
#
#     # Warm-up run
#     _ = torch.fft.fft(E_gpu)
#
#     start_time = time.time()
#     for _ in range(iterations):
#         spectrum = torch.fft.fft(E_gpu)
#         spectrum *= operator_gpu
#         E_gpu = torch.fft.ifft(spectrum)
#         E_gpu *= torch.abs(E_gpu)**2
#
#     # Wait for M1 GPU to finish
#     torch.mps.synchronize()
#     gpu_time = time.time() - start_time
#
#     print(f"M1 GPU Time (PyTorch MPS): {gpu_time:.4f} seconds")
#     print(f"--> Speedup (GPU vs CPU): {cpu_time / gpu_time:.2f}x")

import torch
import numpy as np
import scipy.fft as cpu_fft
import time

N = 2**12
iterations = 2**9

print(f"Benchmarking 1D FFTs for array size {N} over {iterations} iterations...")

# Forcing complex64 (single precision) so the Apple M1 GPU accepts it
E_cpu = (np.random.rand(N) + 1j * np.random.rand(N)).astype(np.complex64)
operator_cpu = np.exp(1j * np.random.rand(N)).astype(np.complex64)

start_time = time.time()
for _ in range(iterations):
    spectrum = cpu_fft.fft(E_cpu, workers=-1)
    spectrum *= operator_cpu
    E_cpu = cpu_fft.ifft(spectrum, workers=-1)
    E_cpu *= 0.99  # Damping to prevent overflow

cpu_time = time.time() - start_time
print(f"M1 CPU Time (SciPy): {cpu_time:.4f} seconds")

if not torch.backends.mps.is_available():
    print("MPS device not found.")
else:
    mps_device = torch.device("mps")

    # Send to GPU
    E_gpu = torch.tensor(E_cpu, device=mps_device, dtype=torch.complex64)
    operator_gpu = torch.tensor(operator_cpu, device=mps_device, dtype=torch.complex64)

    _ = torch.fft.fft(E_gpu)  # warmup

    start_time = time.time()
    for _ in range(iterations):
        spectrum = torch.fft.fft(E_gpu)
        spectrum *= operator_gpu
        E_gpu = torch.fft.ifft(spectrum)
        E_gpu *= 0.99  # Damping to prevent overflow

    torch.mps.synchronize()
    gpu_time = time.time() - start_time

    print(f"M1 GPU Time (PyTorch MPS): {gpu_time:.4f} seconds")
    print(f"--> Speedup (GPU vs CPU): {cpu_time / gpu_time:.2f}x")
