import cv2
import zipfile
import json
from types import SimpleNamespace as Namespace
import numpy as np
import os
from skvideo import io
from moviepy.editor import VideoFileClip, concatenate_videoclips
import re
import time


def visualize_frame():
    pass


all_thickness = [[2, 1, 2, 2], [8, 4, 6, 6]]
pose_seq = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
pose_colors = [[255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0]] \
              + [[85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255]] \
              + [[0, 85, 255], [0, 0, 255], [255, 0, 170], [170, 0, 255], [255, 0, 255]]
face_seq = list(zip(range(16), range(1, 17))) + list(zip(range(17, 21), range(18, 22)))\
           + list(zip(range(22, 26), range(23, 27))) + list(zip(range(27, 30), range(28, 31))) \
           + list(zip(range(31, 35), range(32, 36))) + list(zip(range(36, 41), range(37, 42))) \
           + [(41, 36)] + list(zip(range(42, 47), range(43, 48))) + [(47, 42)] \
           + list(zip(range(48, 59), range(49, 60))) + [(59, 48)] \
           + list(zip(range(60, 67), range(61, 68))) + [(67, 60)]
face_colors = [[255, 255, 255]] * 68
hand_seq = list(zip(range(4), range(1, 5))) + [(0, 5)] + list(zip(range(5, 8), range(6, 9))) + [(0, 9)] \
           + list(zip(range(9, 12), range(10, 13))) + [(0, 13)] + list(zip(range(13, 16), range(14, 17))) \
           + [(0, 17)] + list(zip(range(17, 20), range(18, 21)))
hand_colors = [[100, 100, 100], [100, 0, 0], [150, 0, 0], [200, 0, 0], [255, 0, 0], [100, 100, 0]] \
              + [[150, 150, 0], [200, 200, 0], [255, 255, 0], [0, 100, 50], [0, 150, 75], [0, 200, 100]] \
              + [[0, 255, 125], [0, 50, 100], [0, 75, 150], [0, 100, 200], [0, 125, 255], [100, 0, 100]] \
              + [[150, 0, 150], [200, 0, 200], [255, 0, 255]]
all_seqs = [pose_seq, face_seq, hand_seq, hand_seq]
all_colors = [pose_colors, face_colors, hand_colors, hand_colors]


def color_video(json_zip, vid_file, out_file, hd=True, max_frames=None):
    video_capture = cv2.VideoCapture(vid_file)
    output_wihout_ext, output_type = os.path.splitext(out_file)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    colored_frames = []
    last_time = time.time()
    total_time = 0
    with zipfile.ZipFile(json_zip) as zip_file:
        json_files = [file for file in zip_file.namelist() if file.endswith(".json")]
        json_files.sort()
        json_count = len(json_files)
        frame_count = min(frame_count, json_count)
        splitted = max_frames < frame_count
        if splitted:
            output_path, output_name = os.path.split(output_wihout_ext)
            output_path = os.path.join(output_path, "splitted")
            output_wihout_ext = os.path.join(output_path, output_name)
            if not os.path.isdir(output_path):
                os.mkdir(output_path)
        json_files = json_files[:frame_count]
        video_number = 1
        for i, file_name in enumerate(json_files):
            ret, canvas = video_capture.read()
            print(i)
            if canvas is None:
                print("No canvas for file: ", file_name)
                break
            written_canvas = canvas_for_frame(canvas, file_name, zip_file, hd=hd)
            canvas = cv2.addWeighted(canvas, 0.1, written_canvas, 0.9, 0)
            colored_frames.append(canvas[:, :, [2, 1, 0]])
            if max_frames and (i + 1) % max_frames == 0 and max_frames != frame_count:
                current_time = time.time()
                needed_time = current_time - last_time
                total_time += needed_time
                print("Needed {} s".format(needed_time))
                last_time = current_time
                colored_frames = np.array(colored_frames)
                io.vwrite("{}{}_{}".format(output_wihout_ext, video_number, output_type), colored_frames)  # WRITE VIDEO
                video_number += 1
                colored_frames = []
    if len(colored_frames) > 0:
        colored_frames = np.array(colored_frames)
        video_number = "_" + str(video_number) if splitted else ""
        io.vwrite("{}{}{}".format(output_wihout_ext, video_number, output_type), colored_frames)  # WRITE VIDEO

    if splitted:
        # noinspection PyUnboundLocalVariable
        combine_videos(output_path, False)
    print(total_time)


def combine_videos(path, delete=False):
    file_names = os.listdir(path)
    file_names.sort(key=lambda _file: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', _file)])
    if len(file_names) == 0:
        return
    total_file_names = [os.path.join(path, video_name) for video_name in file_names]
    video_clips = [VideoFileClip(video_name) for video_name in total_file_names]
    final_clip = concatenate_videoclips(video_clips)
    new_path, _ = os.path.split(path)
    file_name, ext = os.path.splitext(file_names[0])
    new_name = os.path.join(new_path, file_name[:-2] + ".mp4")
    final_clip.write_videofile(new_name)
    if delete:
        for file in total_file_names:
            os.remove(file)
        os.rmdir(path)


def canvas_for_frame(current_canvas, frame_json, zip_file, hd=True):
    with zip_file.open(frame_json, "r") as json_file:
        json_obj = json.loads(json_file.read().decode("utf-8"), object_hook=lambda d: Namespace(**d))
    people = json_obj.people
    for person in people:
        pose_key_points = np.array(person.pose_keypoints_2d).reshape(-1, 3)
        face_key_points = np.array(person.face_keypoints_2d).reshape(-1, 3)
        hand_left_key_points = np.array(person.hand_left_keypoints_2d).reshape(-1, 3)
        hand_right_key_points = np.array(person.hand_right_keypoints_2d).reshape(-1, 3)
        all_key_points = [pose_key_points, face_key_points, hand_left_key_points, hand_right_key_points]
        thick = all_thickness[0] if not hd else all_thickness[1]
        for key_points, seq, colors, thickness in zip(all_key_points, all_seqs, all_colors, thick):
            for joints, color in zip(seq, colors):
                points = key_points[joints, :]
                x = points[:, 0]
                y = points[:, 1]
                c = points[:, 2]
                if any(c < 0.05):
                    continue
                canvas_copy = current_canvas.copy()
                cv2.line(canvas_copy, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), color, thickness)
                current_canvas = cv2.addWeighted(current_canvas, 0.1, canvas_copy, 0.9, 0)
                canvas_copy = current_canvas.copy()
                point_size = thickness + (3 if not hd else 12)
                cv2.line(canvas_copy, (int(x[0]), int(y[0])), (int(x[0]), int(y[0])), color, thickness=point_size)
                current_canvas = cv2.addWeighted(current_canvas, 0.1, canvas_copy, 0.9, 0)
    return current_canvas


if __name__ == '__main__':
    _path = os.path.join(".", "data")
    # _video_filename = os.path.join(_path, "2018-05-29_2200_US_KNBC_The_Ellen_DeGeneres_Show_672-1147.mp4")
    # _json_zip = os.path.join(_path, "2905_small_json.zip")
    # _out_path = os.path.join(_path, "test_colored_video_small.avi")
    _video_filename = os.path.join(_path, "2018-05"
                                        "-29_2200_US_KNBC_The_Ellen_DeGeneres_Show_repaired_compressed_only_video_672"
                                        "-1147_HD.mp4")
    _json_zip = os.path.join(_path, "2905_HD_reduced_to_-1_368.zip")
    _out_path = os.path.join(_path, "test_colored_video_hd.avi")
    color_video(_json_zip, _video_filename, _out_path, hd=True, max_frames=500)
    # combine_videos(os.path.join(_path, "splitted"))
