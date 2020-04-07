# Nvidia Performance Tools

[![Build Status](https://travis-ci.com/cwpearson/nvidia-performance-tools.svg?branch=master)](https://travis-ci.com/cwpearson/nvidia-performance-tools)

## Docker images with Nvidia's Performance Tools

[cwpearson/nvidia-performance-tools on Docker Hub](https://hub.docker.com/repository/docker/cwpearson/nvidia-performance-tools).

```bash
docker pull cwpearson/nvidia-performance-tools/latest-amd64
docker pull cwpearson/nvidia-performance-tools/latest-ppc64le
```

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

## Resources

* [Nvidia Nsight Systems Docs](https://docs.nvidia.com/nsight-systems/)
* [Nvidia Nsight Compute Docs](https://docs.nvidia.com/nsight-compute/)
* [Using Nvidia Nsight Systems in Containers and the Cloud](https://devblogs.nvidia.com/nvidia-nsight-systems-containers-cloud/)
* [Using Nsight Compute to Inspect your Kernels](https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/)