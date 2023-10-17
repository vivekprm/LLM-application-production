#!/bin/bash

# Building the deltalake docker image:
# docker build -t delta_quickstart -f Dockerfile_delta_quickstart .
docker run --name delta_quickstart --rm -it -p 8888-8889:8888-8889 -p 5001:5000 delta_quickstart