set -x

set -e

cd docker
ls -halt

echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin

TRAVIS_COMMIT=${TRAVIS_COMMIT:0:7}
DOCKER_REPO=nvidia-performance-tools
DOCKER_SLUG=$DOCKER_USER/$DOCKER_REPO
DOCKER_TAG=$TRAVIS_CPU_ARCH-10.2-$TRAVIS_BRANCH-$TRAVIS_COMMIT


docker build -f $TRAVIS_CPU_ARCH.dockerfile -t $DOCKER_SLUG:$DOCKER_TAG .
docker push $DOCKER_SLUG:$DOCKER_TAG


if [[ $TRAVIS_BRANCH == master ]]; then
    docker tag $DOCKER_SLUG:$DOCKER_TAG $DOCKER_SLUG:latest-$TRAVIS_CPU_ARCH
    docker push $DOCKER_SLUG:latest-$TRAVIS_CPU_ARCH
else
    docker tag $DOCKER_SLUG:$DOCKER_TAG $DOCKER_SLUG:$TRAVIS_BRANCH-$TRAVIS_CPU_ARCH
    docker push $DOCKER_SLUG:$TRAVIS_BRANCH-$TRAVIS_CPU_ARCH
fi

# remove the login key from the image
rm -fv $HOME/.docker/config.json
