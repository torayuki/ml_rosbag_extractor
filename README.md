# Rosbag Extractor for Machine Learning

This package is extract the rosbag for dataset of machine learning

* Maintainer : Yuki Toramatsu ([toramatsu.yuki@em.ci.ritsumei.ac.jp](mailto:toramatsu.yuki@em.ci.ritsumei.ac.jp))
* Author : Yuki Toramatsu ([toramatsu.yuki@em.ci.ritsumei.ac.jp](mailto:toramatsu.yuki@em.ci.ritsumei.ac.jp))


## Content

[[_TOC_]]

## Purpose

This package create the npy files which include the time-series aligned data from rosbag.
We can use the npy files as the dataset of MRSSM directly.

## Usage

### Extract single rosbag

Fill in the rosbag path and excute this
```Bash
rosrun ml_rosbag_extractor single_rosbag_extractor.py rosbag_path
```

### Extract multi rosbag

Move the rosbags into the scripts directory
```
ml_rosbag_extractor
└── scripts
    ├── (bash scripts)
    ├── (python scripts)
    ├── ***.bag
    ├── ...
    └── ***.bag
```

Then excute this.
```Bash
roscd ml_rosbag_extractor/scripts
bash multi_rosbag_extractor.bash
```

## Preprocess options

You can adjust the variable preprocess options. 
If you want to change the value of preprocess options, you must read this section.


### Common


```
target_topics = dict(
    npy_dict_key=dict(
        topic_name      = (String  | name of topic),
        topic_msg_type  = (String  | type of topic),
        buf_clear       = (Boolean | True or False),
        delta_option    = (Boolean | True or False),
        delta_dict_key  = (String  | the name using as key when the delta values add the observation dict),
        rpy_option      = (Boolean | True or False),
        rpy_dict_key    = (String  | the name using as key when the rpy values add the observation dict),
    ),
    ...other observations...
)
```

#### buffer clear option



#### delta option



#### rpy option



### Image option



#### Clip mode 1



```Python
img = img[options["clip_width"][0]:options["clip_width"][1], options["clip_height"][0]:options["clip_height"][1]]
```
This is a example of the option in clip mode 1.
```Python
npy_dict_key=dict(
    topic_name = "/camera_side/color/image_raw/compressed",
    topic_msg_type = "sensor_msgs/CompressedImage",
    image_options = dict(
        clip_mode = 1,
        clip_width = (None,None),
        clip_height = (80, 560),
        resize = (80, 80),
    ),
    buf_clear = True,
)
```

#### Clip mode 2



```Python
w = options["clip_width"]
center_h = options["clip_height_center"]
center_w = options["clip_width_center"]
img = img[int(center_h-w/2):int(center_h+w/2),int(center_w-w/2):int(center_w+w/2)]
```
This is a example of the option in clip mode 2.
```Python
npy_dict_key=dict(
    topic_name = "/camera_side/color/image_raw/compressed",
    topic_msg_type = "sensor_msgs/CompressedImage",
    image_options = dict(
        clip_mode = 2,
        clip_width = 320,
        clip_height_center = 320,
        clip_width_center = 420,
        resize = (80, 80),
    ),
    buf_clear = True,
)
```

### Audio option

This is a example of the option.
```Python
npy_dict_key=dict(
    topic_name = "/audio/audio",
    topic_msg_type = "audio_common_msgs/AudioData",
    sound_info_topic = "/audio/audio_info",
    sound_option = dict(
        convert_mlsp = False,
        library = "torchaudio",
        device = "cuda:0",
    ),
    buf_clear = False,
)
```

## TODO
- Create config files
- Test multiprocessing scripts (Python)
- Add the function separate successful episodes and failed episodes
