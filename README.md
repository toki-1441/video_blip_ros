Script to acquire camera information and convert it into language using VideoBLIP(ros2)
## How to install
```bash
mkdir -p video_blip_ws/src
cd ~/video_blip_ws/src
git clone https://github.com/toki-1441/video_blip_ros.git
```
## Build & Run
```bash
cd ~/video_blip_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch video_blip_ros video_blip_launch.py
```
## What is the VideoBLIP?
Please go [here](https://github.com/yukw777/VideoBLIP).
## Learned model
[video-blip-opt-2.7b-ego4d](https://huggingface.co/kpyu/video-blip-opt-2.7b-ego4d)
[video-blip-flan-t5-xl-ego4d](https://huggingface.co/kpyu/video-blip-flan-t5-xl-ego4d)
