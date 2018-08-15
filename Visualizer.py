import cv2
import zipfile
import json
from types import SimpleNamespace as Namespace
import numpy as np
import os
from skvideo import io


def visualize_frame():
    pass


pose_seq = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
# face_seq =
pose_colors = [[255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0]] \
              + [[85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255]] \
              + [[0, 85, 255], [0, 0, 255], [255, 0, 170], [170, 0, 255], [255, 0, 255], [85, 0, 255]]


def color_video(json_zip, vid_file, out_file, start=0, end=25000, thickness=8):
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

            written_canvas = canvas_for_frame(canvas, file_name, zip_file, thickness)
            canvas = cv2.addWeighted(canvas, 0.1, written_canvas, 0.9, 0)
            arr.append(canvas[:, :, [2, 1, 0]])
    arr = np.array(arr)
    io.vwrite(out_file, arr)  # WRITE VIDEO


def canvas_for_frame(current_canvas, frame_json, zip_file, thickness = 8):
    with zip_file.open(frame_json, "r") as json_file:
        json_obj = json.loads(json_file.read(), object_hook=lambda d: Namespace(**d))
    people = json_obj.people
    for person in people:
        pose_key_points = np.array(person.pose_keypoints_2d).reshape(-1, 3)
        # face_key_points = np.array(person.face_keypoints_2d).reshape(-1, 3)
        # hand_left_key_points = np.array(person.hand_left_keypoints_2d).reshape(-1, 3)
        # hand_right_key_points = np.array(person.hand_right_keypoints_2d).reshape(-1, 3)
        for joints, color in zip(pose_seq, pose_colors):
            points = pose_key_points[joints, :]
            x = points[:, 0]
            y = points[:, 1]
            c = points[:, 2]
            if any(c < 0.05):
                continue
            canvas_copy = current_canvas.copy()
            cv2.line(canvas_copy, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), color, thickness)
            current_canvas = cv2.addWeighted(current_canvas, 0.1, canvas_copy, 0.9, 0)
    return current_canvas


def from_json(file):
    coordinates = ["x", "y"]
    joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist",
                   "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "neck",
                   "right_eye", "right_ear", "left_eye", "left_ear"]
    with open(file, 'r') as inf:
        out = json.load(inf)

    liste = []
    for fr in out["frames"]:
        l_joints = []
        for j in joints_list[:12]:
            l_coo = []
            # print("liste",list(fr.keys()), "joint", j, "in dic?", str(j) in list(fr.keys()), list(fr.keys())[-1])
            for xy in coordinates:
                l_coo.append(fr[j][xy])
            l_joints.append(l_coo)
        liste.append(l_joints)
    return np.array(liste)


if __name__ == '__main__':
    path = os.path.join(".", "data")
    video_filename = os.path.join(path, "2018-05-29_2200_US_KNBC_The_Ellen_DeGeneres_Show_repaired_compressed_only_video_672-1147_HD.mp4")
    json_zip = os.path.join(path, "2905_HD_reduced_to_-1_368.zip")
    out_path = os.path.join(path, "test_colored_video.avi")
    color_video(json_zip, video_filename, out_path)
