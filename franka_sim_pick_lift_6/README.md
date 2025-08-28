---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v2.1",
    "robot_type": null,
    "total_episodes": 30,
    "total_frames": 802,
    "total_tasks": 1,
    "total_videos": 60,
    "total_chunks": 1,
    "chunks_size": 1000,
    "fps": 10,
    "splits": {
        "train": "0:30"
    },
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [
                18
            ],
            "names": null
        },
        "action": {
            "dtype": "float32",
            "shape": [
                4
            ],
            "names": [
                "delta_x_ee",
                "delta_y_ee",
                "delta_z_ee",
                "gripper_delta"
            ]
        },
        "next.reward": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "next.done": {
            "dtype": "bool",
            "shape": [
                1
            ],
            "names": null
        },
        "complementary_info.discrete_penalty": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": [
                "discrete_penalty"
            ]
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": [
                3,
                128,
                128
            ],
            "names": [
                "channels",
                "height",
                "width"
            ],
            "info": {
                "video.height": 128,
                "video.width": 128,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 10,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": [
                3,
                128,
                128
            ],
            "names": [
                "channels",
                "height",
                "width"
            ],
            "info": {
                "video.height": 128,
                "video.width": 128,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 10,
                "video.channels": 3,
                "has_audio": false
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    }
}
```


## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```
df.shape/size information:

[observation.state	action	next.reward	next.done	complementary_info.discrete_penalty	timestamp	frame_index	episode_index	index	task_index]