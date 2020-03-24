set -x
set -e

sudo apt-get update
sudo apt-get install -q -y wget

if [[ $TRAVIS_CPU_ARCH == ppc64le ]]; then
  wget -qSL https://uofi.box.com/shared/static/vfxflckdjixxkc524qltme4sx8kt3w9d.deb -O docker/NVIDIA_Nsight_Systems_Power_CLI_Only_2020.2.1.71.deb;
  wget -qSL https://uofi.box.com/shared/static/swjp2bjr7xj153vzw8mvutv2tqomypxu.run -O docker/nsight-compute-PPC64LE-2019.5.0.14-27346997.run;
elif [[ $TRAVIS_CPU_ARCH == amd64 ]]; then 
  wget -qSL https://uofi.box.com/shared/static/zjsv2rayiotyrdix6a6yd3w8cre56lo0.deb -O docker/NVIDIA_Nsight_Systems_Linux_2020.2.1.71.deb;
  wget -qSL https://uofi.box.com/shared/static/4fuf3wws1uplhf29ndcq4s91kl3jyl7z.run -O docker/nsight-compute-linux-2019.5.0.14-27346997.run;
fi