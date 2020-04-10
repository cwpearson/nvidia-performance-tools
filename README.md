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

[ECE 408 Spring 2020 - Introduction to Nvidia Performance Tools](https://docs.google.com/presentation/d/1A5i3Zdh7ltOLdW7qHZ2tviXYcyl1sKvM7kRpnzOD7tQ/edit?usp=sharing)

## nvcc

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

## Nsight Compute

```bash
nv-nsight-cu-cli a.out
nv-nsight-cu-cli --csv a.out
```

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

**Creating a report**

```
nv-nsight-cu-cli -o report ...
```

Then open the report in the NVIDIA Nsight Compute GUI: 

File > Open File > `report.nsight-cuprof-report`


**Only certain kernels:**

The `--kernel-id` flag takes a string like `context-id:stream-id:[name-operator:]kernel-name:invocation-nr`.
Commonly, we might only use `kernel-name`, to select kernels to profile by name, and `invocation-nr`, to select which invocation of the kernels to profile.

In this example, we profile the `mygemm` kernel's 6th invocation.

```
nv-nsight-cu-cli --kernel-id "::mygemm:6" sgemm-basic
```

Get supported metrics
```
nv-nsight-cu-cli --devices 0 --query-metrics >my_metrics.txt
```



## Nsight Systems

```bash
nsys profile a.out
```

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

* `docs/GEMM-joint-tiling.ppt`: Joint-tiling slide deck from ECE 508 Spring 2017
* [Nsight Graphics Stall Reasons](https://docs.nvidia.com/drive/drive_os_5.1.12.0L/nsight-graphics/activities/#shaderprofiler_stallreasons)

* NVIDIA Devloper Blog
  * [Nsight Systems Exposes GPU Optimization (May 30 2018)](https://devblogs.nvidia.com/nsight-systems-exposes-gpu-optimization/)
  * [Using Nsight Compute to Inspect your Kernels](https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/)
  * [Using Nvidia Nsight Systems in Containers and the Cloud](https://devblogs.nvidia.com/nvidia-nsight-systems-containers-cloud/)