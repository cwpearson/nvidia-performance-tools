# Building a Development Container

First, download any dependencies required in `ci/install_deps.sh`.

Build a test image
```
docker build -f amd64_10-1.dockerfile -t npt-dev .
```

Run an image
```
docker run -it --rm npt-dev
```

Delete all unused docker data:
```
docker prune system
```

## Travis

CI is done through travis-ci.com.
Travis builds the example code, as well as all docker images.
Carl Pearson's docker hub account is used to push images up to [cwpearson/nvidia-performance-tools on Docker Hub](https://hub.docker.com/repository/docker/cwpearson/nvidia-performance-tools).

## Resources

* [Nvidia Docker Image Definitions](https://gitlab.com/nvidia/container-images/cuda/)

## Roadmap

* [ ] Using Nsight Compute and Nsight Systems on EWS
* [ ] Instructions for remote profiling
* [ ] Nsight Systems: How to load missing source file
* [ ] Definitions for Various Performance Terms
  * [ ] Occupancy
  * [ ] Memory Hierarchy
  * [ ] Scheduling
    * [ ] Stall reasons
  * [ ] cudaStreams, cudaEvents
* [ ] CUDA Event and Stream timing examples
  * [ ] single-device
  * [ ] multi-device
* [ ] interacting with `.qdrep` files.
* [ ] interacting with `.nsight-cuprof-report` files.
* [ ] Best Practices
  * [ ] Fixing GPU frequency
  * [ ] initial CUDA runtime cost
  * [ ] Warmup Kernels
  * [ ] `cuda-memcheck` race condition and sync check?
* Is stream 0 the default stream?
* Nsight System with MPI
* Nsight System with multi-GPU


## Inspirations

* https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s22141-what-the-profiler-is-telling-you-how-to-get-the-most-performance-out-of-your-hardware.pdf
  * https://developer.nvidia.com/gtc/2020/video/s22141
* https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21351-scaling-the-transformer-model-implementation-in-pytorch-across-multiple-nodes.pdf
  * https://developer.nvidia.com/gtc/2020/video/s21351
