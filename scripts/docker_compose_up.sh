#!/bin/bash
cd /home/clearpeakegroup/finrl-platform
# Build and start containers using the correct context
export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1
docker-compose -f infra/docker-compose.yml build
docker-compose -f infra/docker-compose.yml up -d 