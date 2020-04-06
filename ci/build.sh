set -x
set -e

source ci/env.sh

if [[ $BUILD_DOCKER == "1" ]]; then
    cd $TRAVIS_BUILD_DIR

    echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin

    TRAVIS_COMMIT=${TRAVIS_COMMIT:0:7}
    DOCKER_REPO=nvidia-performance-tools
    DOCKER_SLUG=$DOCKER_USER/$DOCKER_REPO
    DOCKER_TAG=${TRAVIS_CPU_ARCH}-10.1-$TRAVIS_BRANCH-$TRAVIS_COMMIT


    docker build -f ${TRAVIS_CPU_ARCH}_10-1.dockerfile -t $DOCKER_SLUG:$DOCKER_TAG .
    docker push $DOCKER_SLUG:$DOCKER_TAG


    if [[ $TRAVIS_BRANCH == master ]]; then
        docker tag $DOCKER_SLUG:$DOCKER_TAG $DOCKER_SLUG:latest-${TRAVIS_CPU_ARCH}
        docker push $DOCKER_SLUG:latest-${TRAVIS_CPU_ARCH}
    else
        docker tag $DOCKER_SLUG:$DOCKER_TAG $DOCKER_SLUG:$TRAVIS_BRANCH-$TRAVIS_CPU_ARCH
        docker push $DOCKER_SLUG:$TRAVIS_BRANCH-${TRAVIS_CPU_ARCH}
    fi
fi


if [[ $BUILD_TYPE != '' ]]; then
    cd $TRAVIS_BUILD_DIR
    cd sgemm
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE
    make VERBOSE=1
fi