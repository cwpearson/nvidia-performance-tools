set -x
set -e

source ci/env.sh

# deps for building docker images
if [[ $BUILD_DOCKER == "1" ]]; then
  cd $TRAVIS_BUILD_DIR

  if [[ $TRAVIS_CPU_ARCH == ppc64le ]]; then
    wget -qSL https://uofi.box.com/shared/static/vfxflckdjixxkc524qltme4sx8kt3w9d.deb -O NVIDIA_Nsight_Systems_Power_CLI_Only_2020.2.1.71.deb;
    wget -qSL https://uofi.box.com/shared/static/swjp2bjr7xj153vzw8mvutv2tqomypxu.run -O nsight-compute-PPC64LE-2019.5.0.14-27346997.run;
  elif [[ $TRAVIS_CPU_ARCH == amd64 ]]; then 
    wget -qSL https://uofi.box.com/shared/static/zjsv2rayiotyrdix6a6yd3w8cre56lo0.deb -O NVIDIA_Nsight_Systems_Linux_2020.2.1.71.deb;
    wget -qSL https://uofi.box.com/shared/static/4fuf3wws1uplhf29ndcq4s91kl3jyl7z.run -O nsight-compute-linux-2019.5.0.14-27346997.run;
  fi
fi

# deps for building code
if [[ $BUILD_TYPE != '' ]]; then
    cs $HOME

    ## install CMake
    wget -qSL https://github.com/Kitware/CMake/releases/download/v3.8.2/cmake-3.8.2-Linux-x86_64.tar.gz -O cmake.tar.gz
    mkdir -p $CMAKE_PREFIX
    tar -xf cmake.tar.gz --strip-components=1 -C $CMAKE_PREFIX
    rm cmake.tar.gz

    ## install CUDA
    sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    CUDA102="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.2.89-1_amd64.deb"
    wget -SL $CUDA102 -O cuda.deb
    sudo dpkg -i cuda.deb
    sudo apt-get update 
    sudo apt-get install -y --no-install-recommends \
        cuda-toolkit-10-2
    rm cuda.deb
fi