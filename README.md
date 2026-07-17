# Isaac Sim City Demo

A minimal NVIDIA Isaac Sim scene: an NVIDIA Leatherback car parked on a street in the **Rivermark** outdoor city environment, captured by a camera doing a slow 360° orbit, rendered headless and encoded to video.

![Demo](demo.gif)

*Full-quality video: [simple_city_car.mp4](simple_city_car.mp4)*

## What it does

[`simple_city_car.py`](simple_city_car.py):

1. Starts Isaac Sim headless (`SimulationApp({"headless": True})`)
2. Loads the Rivermark city environment (`Isaac/Environments/Outdoor/Rivermark/rivermark.usd`)
3. Spawns an NVIDIA Leatherback car (`Isaac/Robots/NVIDIA/Leatherback/leatherback.usd`) on a road surface located by probing the bounding boxes of the map's roadmark tiles
4. Adds a distant "sun" light and dome light so the shot is well exposed
5. Orbits a 1280×720 RGB camera 360° around the car over 200 frames (10 s @ 20 fps), saving each frame as a JPEG
6. Frames are encoded to MP4 with ffmpeg

## Running it

Tested with **Isaac Sim 5.1.0** (pip install) on Ubuntu 22.04 with an RTX 4000 Ada (RunPod), Python 3.11.

```bash
sudo apt-get install -y xvfb ffmpeg libglu1-mesa libegl1
export OMNI_KIT_ACCEPT_EULA=yes
xvfb-run -a -s "-screen 0 1280x720x24" python -u simple_city_car.py
ffmpeg -framerate 20 -i simple_city_frames/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p simple_city_car.mp4
```

> Note: `libglu1-mesa` is required — without it the RTX material system (MDL) fails to load and the camera silently returns empty frames.
