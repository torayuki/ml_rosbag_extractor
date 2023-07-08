#!/usr/bin/env python2
import cv2
import math
import numpy as np


#################################################################################
#                                   image                                       #
#################################################################################
# def image2msg(image):
#     img_msg = Image()
#     (img_msg.width, img_msg.height, channel) = image.shape
#     img_msg.encoding = "bgr8"
#     img_msg.step = img_msg.width * channel
#     img_msg.data = image.tostring()
#     return img_msg


def image_raw_msg2opencv(image_msg):
    image_np = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, -1)
    image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np


def image_cmp_msg2opencv(image_msg):
    image_np = np.fromstring(image_msg.data, dtype=np.uint8)
    image_np = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image_np


def bilinear_interpolation(data1, data2, dtype=np.uint8):
    data_sum = np.array(data1, dtype=dtype) + np.array(data2, dtype=dtype)
    return np.divide(data_sum, 2, dtype=dtype)


#################################################################################
#                                   audio                                       #
#################################################################################
def sound_postprocess(sound):
    sound_mel = np.multiply(sound, -80)
    return sound_mel


def float32_to_pcm(float, bit_depth=16):
    return np.multiply(np.floor_divide(float, 1.0 / (2 ** (bit_depth - 1))), 1.0 / (2 ** (bit_depth - 1)))


def wav2mlsp_converter(wav_list, library="librosa", device="cpu"):
    # fft parameter
    sr = 16000
    fft_size = 1024
    frame_period = 5  # ms
    target_hz = 10
    n_mels = 128
    hop_length = int(0.001 * sr * frame_period)
    frame_num = int((1 / target_hz) / (0.001 * frame_period))
    top_db = 80.0
    multiplier = 10.0
    amin = 1e-10
    ref_value = np.max
    # db_multiplier = math.log10(max(amin, ref_value))
    device = torch.device(device)
    sound_array = []
    if library == "librosa":
        import librosa

        for i in range(len(wav_list)):
            mlsp = librosa.feature.melspectrogram(
                y=wav_list[i], sr=sr, n_fft=fft_size, hop_length=hop_length, htk=False
            )
            mlsp = librosa.power_to_db(mlsp, ref=ref_value)
            sound_array.append(mlsp[:, :frame_num])
        # sound preprocess [-0 ~ -80] -> [0 ~ 1]
        sound_array = np.array(sound_array).astype(np.float32)
        sound_array = np.divide(np.abs(sound_array), 80).astype(np.float32)
    elif library == "torchaudio":
        import torch
        import torchaudio

        trans_mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=fft_size,
            win_length=None,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="slaney",
        ).to(device=device)
        for i in range(len(wav_list)):
            temp = torch.FloatTensor(wav_list[i]).to(device=device)
            mlsp_power = trans_mel(temp)
            ref_value = mlsp_power.max(dim=1)[0].max(dim=0)[0]
            mlsp = torchaudio.functional.amplitude_to_DB(
                trans_mel(temp), multiplier, amin, math.log10(max(amin, ref_value)), top_db
            )
            # sound preprocess [-0 ~ -80] -> [0 ~ 1]
            mlsp = torch.narrow(mlsp.abs().float().div_(80), 1, 0, frame_num).cpu().detach().numpy()
            sound_array.append(mlsp)
        del trans_mel, temp, mlsp_power, ref_value
    else:
        raise NotImplementedError("Error : please select library torchaudio or librosa")
    return np.array(sound_array)


#################################################################################
#                                   messages                                    #
#################################################################################
def convert_quaternion2euler(array):
    idx_start = 3
    result = np.zeros(shape=(array.shape[0], array.shape[1] - 1))
    for i in range(len(array)):
        result[i, :idx_start] = array[i, :idx_start]
        result[i, idx_start], result[i, idx_start + 1], result[i, idx_start + 2] = quaternion2euler_numpy(
            array[i, idx_start], array[i, idx_start + 1], array[i, idx_start + 2], array[i, idx_start + 3]
        )
    return result


def quaternion2euler_numpy(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.degrees(np.arctan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = np.where(t2 > +1.0, +1.0, t2)
    t2 = np.where(t2 < -1.0, -1.0, t2)
    pitch_y = np.degrees(np.arcsin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.degrees(np.arctan2(t3, t4))
    return roll_x, pitch_y, yaw_z  # in radians


def normalize_twist_data(data):
    for i in range(3):
        data[i] = data[i] * 20 + 0.5
    return data


def postprocess_normalize_twist_data(data):
    for i in range(len(data)):
        data[i] = (data[i] - 0.5) / 20.0


def get_audio_info(bag, audio_info_topic):
    result_dict = dict()
    for topic, msg, t in bag.read_messages(topics=audio_info_topic):
        result_dict["n_channel"] = msg.channels
        result_dict["sampling_rate"] = msg.sample_rate
        result_dict["sample_format"] = msg.sample_format
        result_dict["bitrate"] = msg.bitrate
        result_dict["coding_format"] = msg.coding_format
    if result_dict.keys() == []:
        print("Error: Audio Info Topic Not found")
        quit()
    return result_dict


def image_clip_resize(img, options):
    if options["clip_mode"] == 1:
        img = img[
            options["clip_width"][0] : options["clip_width"][1], options["clip_height"][0] : options["clip_height"][1]
        ]
    elif options["clip_mode"] == 2:
        w = options["clip_width"]
        center_h = options["clip_height_center"]
        center_w = options["clip_width_center"]
        img = img[int(center_h - w / 2) : int(center_h + w / 2), int(center_w - w / 2) : int(center_w + w / 2)]
    if options["resize"][0] != None and options["resize"][0] != None:
        img = cv2.resize(img, options["resize"])
    return img


def make_dummy(data_length):
    # dummy data
    action = np.zeros((data_length, 1)).astype(np.float32)
    reward = np.zeros((data_length,)).astype(np.float32)
    done = np.zeros((data_length,)).astype(np.float32)
    done[-1] = 1.0
    return action, reward, done


def sync_data(data):
    index_list = []
    for key in data.keys():
        index_list.append(len(data[key]))
    min_index = np.min(index_list)
    for key in data.keys():
        data[key] = data[key][:min_index]
    action, reward, done = make_dummy(data_length=min_index)
    data["action"] = action
    data["reward"] = reward
    data["done"] = done
    return data


#################################################################################
#                                   converter                                   #
#################################################################################
def stdheader_converter(msg) -> float:
    return msg.stamp.to_sec()


def jointstate_converter(msg) -> dict:
    time_stamp = stdheader_converter(msg.header)
    # naem : [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_gripper]
    name_list = msg.name
    position_list = msg.position
    velocity_list = msg.velocity
    effort_list = msg.effort
    return dict(
        time_stamp=time_stamp,
        name=name_list,
        position=position_list,
        velocity=velocity_list,
        effort=effort_list,
    )


def posestamped_converter(msg) -> dict:
    time_stamp = stdheader_converter(msg.header)
    pose_list = pose_converter(msg.pose)
    return dict(
        time_stamp=time_stamp,
        pose=pose_list,
    )


def pose_converter(msg) -> list:
    position_list = geometry_msgs_vector3d_converter(msg.position)
    orientation_list = geometry_msgs_quaternion_converter(msg.orientation)
    return position_list + orientation_list


def geometry_msgs_vector3d_converter(msg) -> list:
    return [msg.x, msg.y, msg.z]


def geometry_msgs_quaternion_converter(msg) -> list:
    return [msg.x, msg.y, msg.z, msg.w]


def vector3stamped_converter(msg) -> dict:
    time_stamp = stdheader_converter(msg.header)
    vector = geometry_msgs_vector3d_converter(msg.vector)
    return dict(
        time_stamp=time_stamp,
        vector=vector,
    )


def pose_vector_converter(msg) -> dict:
    if hasattr(msg, "pose"):
        data = posestamped_converter(msg)["pose"]
    elif hasattr(msg, "vector"):
        data = vector3stamped_converter(msg)["vector"]
    else:
        raise NotImplementedError()
    return data


def twiststamped_converter(msg) -> dict:
    time_stamp = stdheader_converter(msg.header)
    twist = geometry_msgs_twist_converter(msg.twist)
    return dict(
        time_stamp=time_stamp,
        twist=twist,
    )


def geometry_msgs_twist_converter(msg) -> list:
    linear = geometry_msgs_vector3d_converter(msg.linear)
    angular = geometry_msgs_vector3d_converter(msg.angular)
    return linear + angular


def weight_stamped_converter(msg) -> dict:
    time_stamp = stdheader_converter(msg.header)
    weight = weight_msg_converter(msg.weight)
    return dict(
        time_stamp=time_stamp,
        weight=weight,
    )


def weight_msg_converter(msg) -> list:
    value = msg.value
    stable = msg.stable
    return [value, stable]
