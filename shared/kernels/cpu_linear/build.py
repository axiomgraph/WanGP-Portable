# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Rebuilds the CPU linear libraries after a change to linear.cpp or ggml_x86_port.c: python -m shared.kernels.cpu_linear.build
On Windows it builds cpu_linear_win_amd64.dll with MSVC (Visual Studio or Build Tools with the C++ workload), then cpu_linear_linux_x86_64.so
in WSL when WSL is installed (CPU_LINEAR_WSL_PYTHON: the Python run there, python3 by default; CXX / CC: the compilers there). On Linux it
builds the .so with g++ and gcc (AVX2/FMA/F16C), the C++ runtime linked statically: it needs glibc."""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = ["linear.cpp", "ggml_x86_port.c"]


def build_windows():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    from mmgp.allocator.build import _msvc_environment
    env, cl = _msvc_environment()
    target = "cpu_linear_win_amd64.dll"
    result = subprocess.run([cl, "/nologo", "/O2", "/EHsc", "/LD", "/MD", "/std:c++17", "/arch:AVX2", *SOURCES, f"/Fe{target}"], cwd=HERE, env=env)
    for name in os.listdir(HERE):
        if name.startswith(("linear.", "ggml_x86_port.", "cpu_linear_win_amd64.")) and name.endswith((".obj", ".lib", ".exp")):
            os.remove(os.path.join(HERE, name))
    return result.returncode, target


def build_linux():
    cxx = os.environ.get("CXX", "g++")
    cc = os.environ.get("CC", re.sub(r"g\+\+$", "gcc", cxx))
    for compiler in (cxx, cc):
        if shutil.which(compiler) is None:
            raise RuntimeError(f"{compiler} is required (C++17 / C11).")
    target, flags = "cpu_linear_linux_x86_64.so", ["-O3", "-fPIC", "-fvisibility=hidden", "-mavx2", "-mfma", "-mf16c", "-pthread"]
    result = subprocess.run([cc, "-std=c11", "-c", *flags, "ggml_x86_port.c", "-o", "ggml_x86_port.o"], cwd=HERE)
    if result.returncode == 0:
        result = subprocess.run([cxx, "-std=c++17", "-shared", *flags, "-ffunction-sections", "-fdata-sections", "linear.cpp", "ggml_x86_port.o", "-static-libstdc++",
                                 "-static-libgcc", "-Wl,--gc-sections", "-s", "-o", target], cwd=HERE)
    if os.path.exists(os.path.join(HERE, "ggml_x86_port.o")):
        os.remove(os.path.join(HERE, "ggml_x86_port.o"))
    if result.returncode == 0 and shutil.which("objdump"):
        symbols = subprocess.run(["objdump", "-T", os.path.join(HERE, target)], capture_output=True, text=True).stdout
        versions = sorted({tuple(int(x) for x in v.split(".")) for v in re.findall(r"GLIBC_([0-9.]+)", symbols)})
        if versions:
            print(f"{target} needs glibc {'.'.join(map(str, versions[-1]))} or newer")
    return result.returncode, target


def build_linux_in_wsl():
    if shutil.which("wsl") is None:
        print("WSL is not installed: the Linux library was not rebuilt (run python3 shared/kernels/cpu_linear/build.py on Linux).")
        return 0
    folder = subprocess.run(["wsl", "-e", "wslpath", "-a", HERE.replace("\\", "/")], capture_output=True, text=True).stdout.strip()
    return subprocess.run(["wsl", "-e", os.environ.get("CPU_LINEAR_WSL_PYTHON", "python3"), f"{folder}/build.py"]).returncode


def main():
    if sys.platform == "win32":
        code, target = build_windows()
        print(f"{'built' if code == 0 else 'failed to build'} {os.path.join(HERE, target)}")
        code = code or build_linux_in_wsl()
    elif sys.platform.startswith("linux"):
        code, target = build_linux()
        print(f"{'built' if code == 0 else 'failed to build'} {os.path.join(HERE, target)}")
    else:
        raise RuntimeError("The CPU linear kernels support Windows and Linux.")
    sys.exit(code)


if __name__ == "__main__":
    main()
