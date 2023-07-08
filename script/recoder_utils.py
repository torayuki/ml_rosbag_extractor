#!/usr/bin/env python2
import numpy as np

from topic_utils import *


class TopicRecoder:
    def __init__(self, target_hz, target_topics, start_margin=0.02):
        self.target_hz = target_hz
        self.time_dulation = float(1.0 / self.target_hz)
        self.target_topics = target_topics
        self.topic_buffer = dict()
        self.observations = dict()
        self.observations_times = dict()
        self.start_margin = start_margin
        self.target_time = start_margin

        for key in self.target_topics.keys():
            self.topic_buffer[key] = dict(
                data=[],
                time=[],
            )
            self.observations[key] = []
            self.observations_times[key] = []

        print("Topic Recoder initialize complete")

    def init_audio(self, bag):
        print("audio setting ...")
        self.target_time = self.target_time + self.time_dulation
        audio_info = get_audio_info(bag, self.target_topics["sound"]["sound_info_topic"])

        self.sr = audio_info["sampling_rate"]
        self.frame_period = 5  # ms
        self.hop_length = int(0.001 * self.sr * self.frame_period)
        self.frame_num = int((1 / self.target_hz) / (0.001 * self.frame_period))
        self.sound_length = int(self.sr / self.target_hz)

        print("audio initialize complete")

    def __call__(self, topic, msg, time):
        if time > self.target_time:
            self.append_latest_data()
            self.target_time = self.target_time + self.time_dulation
        for key in self.target_topics.keys():
            if topic == self.target_topics[key]["topic_name"]:
                self.append_topic_buffer(key, msg, time)

    def append_topic_buffer(self, key, msg, time):
        msg_type = self.target_topics[key]["topic_msg_type"]
        if msg_type == "sensor_msgs/Image":
            img = image_raw_msg2opencv(msg)
            self.topic_buffer[key]["data"].append(img)
            self.topic_buffer[key]["time"].append(time)
        elif msg_type == "sensor_msgs/CompressedImage":
            img = image_cmp_msg2opencv(msg)
            self.topic_buffer[key]["data"].append(img)
            self.topic_buffer[key]["time"].append(time)
        elif msg_type == "audio_common_msgs/AudioData":
            wav = float32_to_pcm(np.frombuffer(msg.data, dtype=np.float32))
            self.topic_buffer[key]["data"].extend(wav)
            # self.topic_buffer[key]["time"].append(time)
        elif msg_type == "sensor_msgs/JointState":
            joint_state = jointstate_converter(msg)["position"]
            self.topic_buffer[key]["data"].append(joint_state)
            self.topic_buffer[key]["time"].append(time)
        elif msg_type == "geometry_msgs/PoseStamped":
            pose_data = posestamped_converter(msg)["pose"]
            self.topic_buffer[key]["data"].append(pose_data)
            self.topic_buffer[key]["time"].append(time)
        elif msg_type == "geometry_msgs/Vector3Stamped":
            vector_data = vector3stamped_converter(msg)["vector"]
            self.topic_buffer[key]["data"].append(vector_data)
            self.topic_buffer[key]["time"].append(time)
        elif msg_type == "geometry_msgs/TwistStamped":
            twist_data = twiststamped_converter(msg)["twist"]
            # twist_data = normalize_twist_data(twist_data)
            self.topic_buffer[key]["data"].append(twist_data)
            self.topic_buffer[key]["time"].append(time)
        elif msg_type == "and_scale_ros/WeightStamped":
            weight_data = weight_stamped_converter(msg)["weight"][0]
            self.topic_buffer[key]["data"].append(weight_data)
            self.topic_buffer[key]["time"].append(time)
        else:
            raise NotImplementedError("please check the type of topic")

    def append_latest_data(self):
        for key in self.target_topics.keys():
            if "image" in key:
                if self.topic_buffer[key]["data"] != []:
                    img = image_clip_resize(
                        self.topic_buffer[key]["data"][-1], self.target_topics[key]["image_options"]
                    )
                    self._obs_append_checker(key, img, self.topic_buffer[key]["time"][-1])
            elif "sound" in key:
                self.observations[key].append(self.topic_buffer[key]["data"][-self.sound_length :])
            else:
                if self.topic_buffer[key]["data"] != []:
                    self._obs_append_checker(
                        key, self.topic_buffer[key]["data"][-1], self.topic_buffer[key]["time"][-1]
                    )

            if self.target_topics[key]["buf_clear"] == True:
                self.clear_key_buffer(key)

    def _obs_append_checker(self, key, data, time):
        if self.observations_times[key] == []:
            for i in range(int((self.target_time - self.start_margin) * self.target_hz) - 1):
                self.observations[key].append(None)
        elif self.observations_times[key][-1] < self.target_time - self.time_dulation:
            for i in range(int((self.target_time - self.observations_times[key][-1]) * self.target_hz) - 1):
                self.observations[key].append(None)
        self.observations[key].append(data)
        self.observations_times[key].append(time)

    def _show_num_missing_value(self):
        print("count NOT miss value")
        for key in self.target_topics.keys():
            print(
                "{0}:{1}/{2}".format(
                    key, sum(x is not None for x in self.observations[key]), len(self.observations[key])
                )
            )

    def _show_num_double_missing_value(self):
        print("count double miss value")
        for key in self.target_topics.keys():
            count = 0
            for i in range(len(self.observations[key]) - 1):
                if (self.observations[key][i] is None) and (self.observations[key][i + 1] is None):
                    count = count + 1
            print("{0}:{1}/{2}".format(key, count, len(self.observations[key])))

    def _show_obs_shape(self):
        for key in self.observations.keys():
            print("{0}:{1}".format(key, self.observations[key].shape))

    def obs_complement(self):
        # # None search
        # self._show_num_missing_value()
        # self._show_num_double_missing_value()

        # remove first and last miss values
        obs_index_list = []
        init_miss_value_list = []
        for key in self.target_topics.keys():
            init_miss_value_count = 0
            obs_index_list.append(len(self.observations[key]))
            for i in range(len(self.observations[key])):
                if self.observations[key][i] is not None:
                    break
                else:
                    init_miss_value_count += 1
            init_miss_value_list.append(init_miss_value_count)

        for key in self.target_topics.keys():
            self.observations[key] = self.observations[key][: min(obs_index_list)]
        for _ in range(max(init_miss_value_list)):
            self._remove_obs_item_idx(0)
        del obs_index_list, init_miss_value_list

        # interpolation
        for key in self.target_topics.keys():
            for i in range(len(self.observations[key])):
                if self.observations[key][i] is None:
                    self.observations[key][i] = self.observations[key][i - 1]

        # convert to numpy array
        for key in self.target_topics.keys():
            self.observations[key] = np.array(self.observations[key])

        for key in self.target_topics.keys():
            # waveform convert to mlsp
            if "sound" in key:
                if self.target_topics[key]["sound_option"]["convert_mlsp"] == True:
                    self.observations[key] = wav2mlsp_converter(
                        self.observations[key],
                        self.target_topics[key]["sound_option"]["library"],
                        self.target_topics[key]["sound_option"]["device"],
                    )
            # delta option
            if "delta_option" in self.target_topics[key].keys():
                if self.target_topics[key]["delta_option"] == True:
                    self._generate_delta_values(key, self.target_topics[key]["delta_dict_key"])
            # rpy option
            if "rpy_option" in self.target_topics[key].keys():
                if self.target_topics[key]["rpy_option"] == True:
                    self._generate_rpy_values(key, self.target_topics[key]["rpy_dict_key"])

        self._show_obs_shape()

        # sync data
        index_list = []
        for key in self.observations.keys():
            index_list.append(len(self.observations[key]))
        min_index = min(index_list)
        for key in self.observations.keys():
            self.observations[key] = self.observations[key][:min_index]
        action, reward, done = make_dummy(data_length=min_index)
        self.observations["action"] = action
        self.observations["reward"] = reward
        self.observations["done"] = done

    def save_dataset(self, out_path):
        np.save(out_path, self.observations, allow_pickle=True, fix_imports=True)

    def get_dataset(self):
        return self.observations

    def get_dataset_time(self):
        return self.observations_times

    def clear_all_buffer(self):
        for key in self.topic_buffer.keys():
            self.clear_key_buffer(key)

    def clear_all_buffer_obs(self):
        for key in self.target_topics.keys():
            self.clear_buffer_key(key)
            self.observations[key] = []
            self.observations_times[key] = []

    def clear_buffer_key(self, key):
        self.topic_buffer[key] = dict(
            data=[],
            time=[],
        )

    def _remove_obs_item_idx(self, idx):
        for key in self.observations.keys():
            self.observations[key].pop(idx)

    def _generate_delta_values(self, key, new_key):
        self.observations[new_key] = self.observations[key][1:, :] - self.observations[key][:-1, :]

    def _generate_rpy_values(self, quat_key, rpy_key):
        self.observations[rpy_key] = convert_quaternion2euler(self.observations[quat_key])
