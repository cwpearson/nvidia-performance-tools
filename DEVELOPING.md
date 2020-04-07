# Building a Development Container

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

## Resources

* [Nvidia Docker Image Definitions](https://gitlab.com/nvidia/container-images/cuda/)