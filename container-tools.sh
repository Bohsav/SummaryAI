#!/bin/bash

# Quick scripts to:
# Build a container image
# Create a container with interactive tty
# Start a container and attach to it

# Arg1 is the Containerfile path
# Arg2 is the name of the image
build-image ()
{
  podman build -t "$2" "$1"
}

# Arg1 is the Containerfile path
# Arg2 is the name of the container
create-container ()
{
  podman create --tty --gpus="all" --name="$2" "$1"
}

# Arg1 is the name or id of the container
run-container()
{
  podman run --interactive --attach "$1"
}

