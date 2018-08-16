import cv2
import zipfile
import json
from types import SimpleNamespace as Namespace
import numpy as np
import os
from skvideo import io


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


def color_video(json_zip, vid_file, out_file, start=0, end=25000, hd=True):
    video_capture = cv2.VideoCapture(vid_file)
    arr = []
    with zipfile.ZipFile(json_zip) as zip_file:
        json_files = [file for file in zip_file.namelist() if file.endswith(".json")]
        json_files.sort()
        for i, file_name in enumerate(json_files):
            if not file_name.endswith(".json"):
                print("Not a json file: {}".format(file_name))
                continue
            ret, canvas = video_capture.read()
            if i > 200:
                break
            print(i)
            if canvas is None:
                print("No canvas for file: ", file_name)
                break

            written_canvas = canvas_for_frame(canvas, file_name, zip_file, hd=hd)
            canvas = cv2.addWeighted(canvas, 0.1, written_canvas, 0.9, 0)
            arr.append(canvas[:, :, [2, 1, 0]])
    arr = np.array(arr)
    io.vwrite(out_file, arr)  # WRITE VIDEO


def canvas_for_frame(current_canvas, frame_json, zip_file, hd=True):
    with zip_file.open(frame_json, "r") as json_file:
        json_obj = json.loads(json_file.read(), object_hook=lambda d: Namespace(**d))
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
                cv2.line(canvas_copy, (int(x[0]), int(y[0])), (int(x[0]), int(y[0])), color, thickness=thickness+3)
                current_canvas = cv2.addWeighted(current_canvas, 0.1, canvas_copy, 0.9, 0)
    return current_canvas


if __name__ == '__main__':
    path = os.path.join(".", "data")
    # video_filename = os.path.join(path, "2018-05-29_2200_US_KNBC_The_Ellen_DeGeneres_Show_672-1147.mp4")
    # json_zip = os.path.join(path, "2905_small_json.zip")
    # out_path = os.path.join(path, "test_colored_video_small.avi")
    video_filename = os.path.join(path, "2018-05"
                                        "-29_2200_US_KNBC_The_Ellen_DeGeneres_Show_repaired_compressed_only_video_672"
                                        "-1147_HD.mp4")
    json_zip = os.path.join(path, "2905_HD_reduced_to_-1_368.zip")
    out_path = os.path.join(path, "test_colored_video_hd.avi")
    color_video(json_zip, video_filename, out_path, hd=False)
