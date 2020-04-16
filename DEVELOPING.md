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