#!/bin/bash
# Run after EVERY RunPod restart — system packages don't persist.
# libglu1-mesa is the critical one: without it the RTX material system fails
# silently and every camera returns empty frames.
set -e
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq xvfb ffmpeg libglu1-mesa libegl1
echo "OK: xvfb=$(which xvfb-run) ffmpeg=$(which ffmpeg) libGLU=$(ls /usr/lib/x86_64-linux-gnu/libGLU.so.1)"
echo "Remember: export OMNI_KIT_ACCEPT_EULA=yes"
