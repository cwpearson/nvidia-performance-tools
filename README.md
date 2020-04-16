# Nvidia Performance Tools

[![Build Status](https://travis-ci.com/cwpearson/nvidia-performance-tools.svg?branch=master)](https://travis-ci.com/cwpearson/nvidia-performance-tools)

## Docker images with Nvidia's Performance Tools

[cwpearson/nvidia-performance-tools on Docker Hub](https://hub.docker.com/repository/docker/cwpearson/nvidia-performance-tools).

```bash
docker pull cwpearson/nvidia-performance-tools:latest-amd64   # for x86
docker pull cwpearson/nvidia-performance-tools:latest-ppc64le # for POWER
```

Typically, you'll want the `latest-amd64` or `latest-ppc64le` tags.
If you are developing a workflow and want stability, choose a tag like `amd64-10.1-master-ce03360`, which describes the architecture, CUDA version, branch, and short SHA of the corresponding git commit for [cwpearson/nvidia-performance-tools on Github](github.com/cwpearson/nvidia-performance-tools).

## Presentations

* ECE 408 Spring 2020 guest lecture for Professor Lumetta.
  * [Slides](docs/20200416_ece408.pdf)
  * Recorded Lecture (75 mins)
    * [Part 1: Intro]()
    * [Part 2: CUDA Events](https://youtu.be/yI137sSOlkU)
    * [Part 3: Nsight Compute]()
    * [Part 4: Nsight Systems]()

## Installing Nsight Systems and Nsight Compute

There is a command-line (CLI) and graphical (GUI) version of each tool.
They will be installed together, unless a CLI-only version is downloaded.

* macOS: You probably don't have CUDA installed, so download the Nsight Systems or Compute installer from the Nvidia website.
* Windows with CUDA:
  * with CUDA: You may already find Nsight Systems or Compute in your start menu. You can download a more recent release from the Nvidia website. If you install it, you will have two entries in the start menu for different versions.
  * without CUDA: Download the Nsight Systems or Compute installer from the CUDA website.
* Linux
  * with CUDA: you may already have Nsight Systems and Compute (check `/usr/local/cuda/bin/nsight-sys` and `/usr/local/cuda/bin/nv-nsight-cu`). If so, you can still download the Nsight Systems or Compute `.deb` package to update. It may override the package that was installed with CUDA. You can also use the `.run` file, which you should install to a directory not managed by the package manager, and add the location of the resulting binary files to your path.
  * without CUDA: 
    * `.deb`: Download the `.deb` package and install it. Requires root privileges
    * `.run`: Download the `.run` package and execute it. Choose a file system that you have permission to install to, and then add the resulting binary directory to your path.

## Preparing for Profiling

### Source code annotations

```c++
#include <nvToolsExt.h>

nvtxRangePush("span 1");
nvtxRangePush("a nested span");
nvtxRangePop(); // end nested span
nvtxRangePop(); // end span 1
```

Also link with `-lnvToolsExt`.

### nvcc

Compile with optimizations turned on, and without debug information.
The most linkely relevant flags for `nvcc` are below:

```
--profile                                       (-pg)                           
        Instrument generated code/executable for use by gprof (Linux only).

--debug                                         (-g)                            
        Generate debug information for host code.

--device-debug                                  (-G)                            
        Generate debug information for device code. Turns off all optimizations.
        Don't use for profiling; use -lineinfo instead.

--generate-line-info                            (-lineinfo)                     
        Generate line-number information for device code.
```

So, change `nvcc -g/-pg/-G ...` to `nvcc <your optimization flags> -lineinfo ...`.

### cuda-memcheck

If your code overwrites unallocated memory, it may corrupt the profiling process.
If profiling fails, try running your code under `cuda-memcheck`.
This will instrument your binary to detect bad GPU memory activity.
Fix any errors that occur, and try profiling again.
This will cause ~100x slowdown usually, so try a small dataset first.

```
cuda-memcheck my-binary
```

### Nsight Systems Environment Check

Run `nsys status -e`. You should see something like 

```
Sampling Environment Check
Linux Kernel Paranoid Level = 2: OK
Linux Distribution = Ubuntu
Linux Kernel Version = 4.16.15-41615: OK
Linux perf_event_open syscall available: OK
Sampling trigger event available: OK
Intel(c) Last Branch Record support: Available
Sampling Environment: OK
```

Errors may reduce the amount of information collected, or cause profiling to fail.
Consult documentation for troubleshooting steps.

## Capturing a Profile with CLI

Under this scheme, we 
* use the CLI on the target to record a profiling file
* transfer that file to the client
* use the GUI on the client to analyze the record

### Nsight Compute

This command will
* Generate `a.nsight-cuprof-report` with recorded profiling information
* Measure metrics associated with all sections
* Profile the 6th invocation of `__global__ void kernel_name(...)`
* Run a.out

```bash
nv-nsight-cu-cli \ 
  -o a \
  --sections ".*" \
  --kernel-id ::kernel_name:6 \
  a.out
```

To see sections that will be recorded for a command, add `--list-sections`.

```
nv-nsight-cu-cli --list-sections
---------------------------- ------------------------------- --------------------------------------------------
Identifier                    Display Name                    Filename                                          
----------------------------- ------------------------------- --------------------------------------------------
ComputeWorkloadAnalysis       Compute Workload Analysis       .../../../sections/ComputeWorkloadAnalysis.section
InstructionStats              Instruction Statistics          ...64/../../sections/InstructionStatistics.section
LaunchStats                   Launch Statistics               ...1_3-x64/../../sections/LaunchStatistics.section
MemoryWorkloadAnalysis        Memory Workload Analysis        ...4/../../sections/MemoryWorkloadAnalysis.section
MemoryWorkloadAnalysis_Chart  Memory Workload Analysis Chart  ..../sections/MemoryWorkloadAnalysis_Chart.section
MemoryWorkloadAnalysis_Tables Memory Workload Analysis Tables .../sections/MemoryWorkloadAnalysis_Tables.section
Occupancy                     Occupancy                       ...ibc_2_11_3-x64/../../sections/Occupancy.section
SchedulerStats                Scheduler Statistics            ...-x64/../../sections/SchedulerStatistics.section
SourceCounters                Source Counters                 ..._11_3-x64/../../sections/SourceCounters.section
SpeedOfLight                  GPU Speed Of Light              ..._2_11_3-x64/../../sections/SpeedOfLight.section
WarpStateStats                Warp State Statistics           ...-x64/../../sections/WarpStateStatistics.section
```

To see supported metrics on a device, do `nv-nsight-cu-cli --devices 0 --query-metrics`

The `--kernel-id` flag takes a string like `context-id:stream-id:[name-operator:]kernel-name:invocation-nr`.
Commonly, we might only use `kernel-name`, to select kernels to profile by name, and `invocation-nr`, to select which invocation of the kernels to profile.

### Nsight Systems

This command will
* Record profiling info to `a.qdreq` 
* Run a.out

```bash
nsys profile \
  -o a
  a.out
```

## Using the GUI on a client to view a recorded file from the target

In **Nsight Compute**:

File > Open File ... > file.nsight-cuprof-report

In **Nsight Systems**:

File > Open > file.qdrep

## Using the GUI on the client to Control Remote Profiling on the target

*instructions to come*

## Managing docker images

* `docker ps -a`
* `docker rm `docker ps -a -q``
* `docker system prune`

Run a profiling container:
```bash
docker run cwpearson/nvidia-performance-tools:latest-amd64
```

Resume a previously exited container:
```bash
* docker ps -a       # find the ID
* docker start <ID>  # resume the exited container
* docker attach <ID> # attach a terminal to the container
```

## For Developers

See [DEVELOPING.md](DEVELOPING.md)

## Resources

* [Nvidia Nsight Systems Docs](https://docs.nvidia.com/nsight-systems/)
* [Nvidia Nsight Compute Docs](https://docs.nvidia.com/nsight-compute/)
* NVIDIA Devloper Blog
  * [Nsight Systems Exposes GPU Optimization (May 30 2018)](https://devblogs.nvidia.com/nsight-systems-exposes-gpu-optimization/)
  * [Using Nsight Compute to Inspect your Kernels (Sep 16 2019)](https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/)
  * [Using Nvidia Nsight Systems in Containers and the Cloud (Jan 29 2020)](https://devblogs.nvidia.com/nvidia-nsight-systems-containers-cloud/)
* Interpreting Nsight Compute Results
  * Workload Memory Analysis
    * [CUDA Memory Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)
    * [Device Memory Access Performance Guidelines](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)
  * Stall Reasons
    * [Nsight Graphics Docs: Stall Reasons](https://docs.nvidia.com/drive/drive_os_5.1.12.0L/nsight-graphics/activities/#shaderprofiler_stallreasons)
  * Issue Efficiency
    * [Issue Efficiency Nsight Visual Studio Edition](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/issueefficiency.htm)
  * Occupancy
    * [Nsight Visual Studio Edition](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm)
* Slides
  * [docs/GEMM-joint-tiling.ppt](docs/GEMM-joint-tiling.ppt): Joint-tiling slide deck from ECE 508 Spring 2017
