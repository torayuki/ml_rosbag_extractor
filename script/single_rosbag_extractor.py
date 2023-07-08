#!/usr/bin/env python2
import os
import sys
import numpy as np
import rosbag

from recoder_utils import TopicRecoder


if __name__ == "__main__":
    target_hz = 10
    target_topics = dict(
        image_horizon=dict(
            topic_name="/camera_side/color/image_raw/compressed",
            image_options=dict(
                # clip_mode = 1,
                # clip_width = (None,None),
                # clip_height = (80, 560),
                clip_mode=2,
                clip_width=320,
                clip_height_center=320,
                clip_width_center=370,
                resize=(80, 80),
            ),
            buf_clear=True,
        ),
        image_vertical=dict(
            topic_name="/camera_top/color/image_raw/compressed",
            image_options=dict(
                # clip_mode = 1,
                # clip_width = (280, 530),
                # clip_height = (250, 500),
                clip_mode=2,
                clip_width=200,
                clip_height_center=340,
                clip_width_center=450,
                resize=(80, 80),
            ),
            buf_clear=True,
        ),
        # image_vertical_high_resolution=dict(
        #     topic_name = "/usb_cam/image_raw/compressed",
        #     image_options = dict(
        #         clip_mode = 2,
        #         clip_width = 320,
        #         clip_height_center = 915,
        #         clip_width_center = 1100,
        #         resize = (None, None),
        #     ),
        #     buf_clear = True,
        # ),
        sound=dict(
            topic_name="/audio/audio",
            sound_info_topic="/audio/audio_info",
            sound_option=dict(
                convert_mlsp=False,
                library="torchaudio",
                device="cuda:0",
            ),
            buf_clear=False,
        ),
        joint_states=dict(
            topic_name="/arm2/joints/get/joint_states",
            buf_clear=True,
            # delta_option = True,
            # delta_dict_key = "d_joint_states",
        ),
        pose_quat=dict(
            topic_name="/arm2_kinematics/get/pose",
            buf_clear=True,
            # rpy_option = True,
            # rpy_dict_key = "pose_rpy",
        ),
        desired_pose_quat=dict(
            topic_name="/arm2_kinematics/set/desired_pose",
            buf_clear=True,
            # rpy_option = True,
            # rpy_dict_key = "pose_rpy",
        ),
        # servo_value = dict(
        #     topic_name = "/servo_server/delta_twist_cmds",
        #     buf_clear = True
        # ),
        weight_value=dict(
            topic_name="/ekew_i_driver/output",
            buf_clear=True,
        ),
    )

    bag_path = sys.argv[1]
    bag = rosbag.Bag(bag_path)

    # collect bag info
    t_s = bag.get_start_time()
    t_e = bag.get_end_time()
    type_topic_info = bag.get_type_and_topic_info()[1]

    read_msg_list = []
    for key in target_topics.keys():
        target_topics[key]["topic_msg_type"] = type_topic_info[target_topics[key]["topic_name"]].msg_type
        read_msg_list.append(target_topics[key]["topic_name"])

    # init recoder
    recoder = TopicRecoder(target_hz, target_topics)
    if "sound" in target_topics.keys():
        recoder.init_audio(bag)

    # collect topic
    for topic, msg, t in bag.read_messages(topics=read_msg_list):
        percentage = (t.to_sec() - t_s) / (t_e - t_s)
        sys.stdout.write("\r{0:4.2f}%".format(100 * percentage))
        t = t.to_sec() - t_s
        recoder(topic, msg, t)
    sys.stdout.flush()

    # dataset interpolation
    recoder.obs_complement()

    # save dataset
    out_path = os.path.splitext(bag_path)[0] + ".npy"
    recoder.save_dataset(out_path)

    print("\nDone!")
