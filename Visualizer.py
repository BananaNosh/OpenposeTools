import cv2
import zipfile
import json
from types import SimpleNamespace as Namespace
import numpy as np
import os
import re
import argparse
import tempfile
import atexit
import sys
from skvideo import io
from moviepy.editor import VideoFileClip, concatenate_videoclips
import enum
import subprocess
import time

class OpenposeOutputFormat(enum.Enum):
    COCO = "coco",
    BODY_25 = "body_25"


HD_THRESHOLD = 1300
# line thickness for different body parts for Non-HD and HD
ALL_THICKNESS = [[2, 1, 2, 2], [8, 4, 6, 6]]
# PoseConnection and colors can be found here
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/pose/poseParametersRender.hpp
# but they have to be corrected at least for B25
# Pose sequence (the connected keypoints) in COCO
POSE_SEQ_COCO = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
                 (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
# Pose colors for COCO
POSE_COLORS_COCO = [[255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0],
                    [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                    [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
                    [0, 85, 255], [0, 0, 255], [255, 0, 170], [170, 0, 255], [255, 0, 255]]
# Pose sequence (the connected keypoints) in BODY_25
POSE_SEQ_B25 = [(1, 8), (1, 2), (1, 5), (2, 3),
                (3, 4), (5, 6), (6, 7), (8, 9),
                (9, 10), (10, 11), (8, 12), (12, 13),
                (13, 14), (1, 0), (0, 15), (15, 17),
                (0, 16), (16, 18), (14, 19), (19, 20),
                (14, 21), (11, 22), (22, 23), (11, 24)]
# Pose colors for BODY_25
POSE_COLORS_B25 = [[255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
                   [255, 0, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
                   [0, 170, 255], [0, 85, 255], [0, 0, 255], [255, 0, 170],
                   [170, 0, 255], [255, 0, 255], [85, 0, 255], [0, 0, 255],
                   [0, 0, 255], [0, 0, 255], [0, 255, 255], [0, 255, 255], [0, 255, 255]]

# Face sequences (the connected keypoints) for the old and the new format
FACE_SEQ = list(zip(range(16), range(1, 17))) + list(zip(range(17, 21), range(18, 22))) \
           + list(zip(range(22, 26), range(23, 27))) + list(zip(range(27, 30), range(28, 31))) \
           + list(zip(range(31, 35), range(32, 36))) + list(zip(range(36, 41), range(37, 42))) \
           + [(41, 36)] + list(zip(range(42, 47), range(43, 48))) + [(47, 42)] \
           + list(zip(range(48, 59), range(49, 60))) + [(59, 48)] \
           + list(zip(range(60, 67), range(61, 68))) + [(67, 60)]
FACE_COLORS = [[255, 255, 255]] * 68
HAND_SEQ = list(zip(range(4), range(1, 5))) + [(0, 5)] + list(zip(range(5, 8), range(6, 9))) + [(0, 9)] \
           + list(zip(range(9, 12), range(10, 13))) + [(0, 13)] + list(zip(range(13, 16), range(14, 17))) \
           + [(0, 17)] + list(zip(range(17, 20), range(18, 21)))
HAND_COLORS = [[100, 100, 100], [100, 0, 0], [150, 0, 0], [200, 0, 0], [255, 0, 0], [100, 100, 0]] \
              + [[150, 150, 0], [200, 200, 0], [255, 255, 0], [0, 100, 50], [0, 150, 75], [0, 200, 100]] \
              + [[0, 255, 125], [0, 50, 100], [0, 75, 150], [0, 100, 200], [0, 125, 255], [100, 0, 100]] \
              + [[150, 0, 150], [200, 0, 200], [255, 0, 255]]
# the combination of all seqs for COCO and BODY_25 format
ALL_SEQS = [[POSE_SEQ_COCO, FACE_SEQ, HAND_SEQ, HAND_SEQ],
            [POSE_SEQ_B25, FACE_SEQ, HAND_SEQ, HAND_SEQ]]
ALL_COLORS = [[POSE_COLORS_COCO, FACE_COLORS, HAND_COLORS, HAND_COLORS],
              [POSE_COLORS_B25, FACE_COLORS, HAND_COLORS, HAND_COLORS]]


def color_video(frames_json_folder, vid_file, out_file, temp_folder, out_images=None, max_frames=None,
                frame_range=None, openpose_format=OpenposeOutputFormat.BODY_25, no_bg=False):
    """
    Create a video from the vid_file with the poses given in the frames_json_folder colored on it
    Args:
        frames_json_folder(str): path to folder of json files (might me a zip folder)
        vid_file(str): path to the video file
        out_file(str): filename of the output video
        temp_folder(str): path to the temp folder to store videos in in intermediate steps
        out_images(str|None): path to folder to store rendered images
        max_frames(int): the maximum number of frames before the video is splitted during the process
        frame_range(Range): the range of frames which should be used to write the new video
        openpose_format(OpenposeOutputFormat): the used output format of openpose
    """
    video_capture = cv2.VideoCapture(vid_file)
    output_without_ext, output_type = os.path.splitext(out_file)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_string = subprocess.Popen('ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate'.split(' ')+[vid_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0].decode('UTF-8')[:-1]
    if fps_string != '0/0':
        fps = eval(fps_string)
    else:
        fps = video_capture.get(cv2.CAP_PROP_FPS)
    all_colored_frames = []
    colored_frames = []
    json_files = get_json_files_from_folder(frames_json_folder)
    json_count = len(json_files)
    frame_count = min(frame_count, json_count)
    splitted = max_frames and max_frames < frame_count
    if splitted:
        temp_folder = generate_temp_folder(temp_folder)
    temp_name = os.path.join(temp_folder, os.path.basename(output_without_ext))
    json_files = json_files[:frame_count]
    frame_range = frame_range or range(frame_count)
    enumerating = zip(frame_range, json_files[frame_range.start:frame_range.stop:frame_range.step])
    if frame_range.start > 0:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_range.start)
    video_number = None
    for i, file_name in enumerating:
        if i % 10 + 1 == 0:
            print("{}/{} frames ready       ".format(i, frame_count), end='\r')
            sys.stdout.flush()
        video_number = (i + 1) // max_frames if max_frames else None
        canvas = None
        for j in range(frame_range.step):
            _, canvas = video_capture.read()
        if no_bg:
            canvas = np.zeros(canvas.shape, dtype=np.uint8)
        elif canvas is None:
            print("")
            print("No canvas for file: ", file_name)
            break
        written_canvas = canvas_for_frame(canvas, file_name, frames_json_folder, openpose_format=openpose_format)
        canvas = cv2.addWeighted(canvas, 0.1, written_canvas, 0.9, 0)
        all_colored_frames.append((file_name, canvas))
        colored_frames.append(canvas[:, :, [2, 1, 0]])
        if max_frames and (i + 1) % max_frames == 0 and max_frames != frame_count:
            colored_frames = np.array(colored_frames)
            write_file(colored_frames, fps, output_type, temp_name, video_number)
            colored_frames = []
    if len(colored_frames) > 0:
        colored_frames = np.array(colored_frames)
        video_number = video_number + 1 if splitted else None
        write_file(colored_frames, fps, output_type, temp_name if splitted else output_without_ext, video_number)
    if out_images is not None:
        write_out_images(out_images, all_colored_frames)
    if splitted:
        combine_videos(out_file, temp_folder, True)


def write_out_images(out_images_path, all_colored_frames):
    for json_name, frame in all_colored_frames:
        name, _ = os.path.splitext(json_name)
        _, name = os.path.split(name)
        name = name if not name.endswith("_keypoints") else name[:-10]
        name += "_rendered.png"
        path = os.path.join(out_images_path, name)
        cv2.imwrite(path, frame)


def get_json_files_from_folder(frames_json_folder):
    json_ext = ".json"
    _, ext = os.path.splitext(frames_json_folder)
    if ext == ".zip":
        with zipfile.ZipFile(frames_json_folder) as zip_file:
            json_files = [file for file in zip_file.namelist() if file.endswith(json_ext)]
    else:
        json_files = [file for file in os.listdir(frames_json_folder) if file.endswith(json_ext)]
    json_files.sort()
    return json_files


def canvas_for_frame(current_canvas, frame_json, frames_json_folder, openpose_format=OpenposeOutputFormat.BODY_25):
    def load_json_from_file(json_string):
        return json.loads(json_string, object_hook=lambda d: Namespace(**d))

    hd = current_canvas.shape[1] > HD_THRESHOLD
    _, ext = os.path.splitext(frames_json_folder)
    if ext == ".zip":
        zip_file = zipfile.ZipFile(frames_json_folder)
        with zip_file.open(frame_json, "r") as json_file:
            json_obj = load_json_from_file(json_file.read().decode("utf-8"))
    else:
        with open(os.path.join(frames_json_folder, frame_json), "r", encoding="utf-8") as json_file:
            json_obj = load_json_from_file(json_file.read())
    canvas_copy = current_canvas.copy()
    people = json_obj.people
    for person in people:
        pose_key_points = np.array(person.pose_keypoints_2d).reshape(-1, 3)
        face_key_points = np.array(person.face_keypoints_2d).reshape(-1, 3)
        hand_left_key_points = np.array(person.hand_left_keypoints_2d).reshape(-1, 3)
        hand_right_key_points = np.array(person.hand_right_keypoints_2d).reshape(-1, 3)
        all_key_points = [pose_key_points, face_key_points, hand_left_key_points, hand_right_key_points]
        thick = ALL_THICKNESS[0] if not hd else ALL_THICKNESS[1]
        format_index = int(openpose_format == OpenposeOutputFormat.BODY_25)
        for key_points, seq, colors, thickness in zip(all_key_points,
                                                      ALL_SEQS[format_index],
                                                      ALL_COLORS[format_index],
                                                      thick):
            for joints, color in zip(seq, colors):
                points = key_points[joints, :]
                x = points[:, 0]
                y = points[:, 1]
                c = points[:, 2]
                if any(c < 0.05):
                    continue
                color = [color[2], color[1], color[0]]
                cv2.line(canvas_copy, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), color=color, thickness=thickness)
                point_size = thickness + (3 if not hd else 12)
                cv2.line(canvas_copy, (int(x[0]), int(y[0])), (int(x[0]), int(y[0])), color=color, thickness=point_size)
    current_canvas = cv2.addWeighted(current_canvas, 0.1, canvas_copy, 0.9, 0)
    return current_canvas


def generate_temp_folder(temp_folder):
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
    elif len(os.listdir(temp_folder)) > 0:
        temp_folder = tempfile.mkdtemp(dir=temp_folder)

    def delete_temp():
        if os.path.exists(temp_folder):
            remove_all_files_in_folder(temp_folder)
            os.rmdir(temp_folder)

    atexit.register(delete_temp)
    return temp_folder


def remove_all_files_in_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def write_file(colored_frames, fps, output_type, output_wihout_ext, video_number):
    name = "{}_{}{}".format(output_wihout_ext, video_number, output_type) \
        if video_number else "{}{}".format(output_wihout_ext, output_type)
    fps_string = "{:.2f}".format(fps)
    io.vwrite(name, colored_frames, inputdict={"-framerate": fps_string},
              outputdict={"-r": fps_string})  # WRITE VIDEO
    print("\nwritten to {}".format(name))


def combine_videos(outfile, temp_path, delete=False):
    file_names = os.listdir(temp_path)
    file_names.sort(key=lambda _file: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', _file)])
    if len(file_names) == 0:
        return
    total_file_names = [os.path.join(temp_path, video_name) for video_name in file_names]
    video_clips = [VideoFileClip(video_name) for video_name in total_file_names]
    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile(outfile)
    if delete:
        remove_all_files_in_folder(temp_path)
        os.rmdir(temp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw the lines given with the json data on to the given video.')
    parser.add_argument("videofile", type=argparse.FileType(mode="r"), help="the video file to write on")
    parser.add_argument("json", help="the folder of json files with the data for each frame (might be zipped)")
    parser.add_argument("outfile", help='the output video file')
    parser.add_argument("-oi --out_images", dest="out_images", help='the output folder for the images')
    parser.add_argument("-coco", dest="coco_format",
                        help='add if the COCO openpose format is used instead of body_25', action='store_true')
    parser.add_argument("-t --temp", metavar='tempfolder', dest="temp", help="folder for saving the temp files")
    parser.add_argument("--maxframes", metavar="maxframes", type=int, default=100,
                        help="maximal number of frames before splitting the video sequence - default to 100")
    parser.add_argument('--noBG', help='Include to show skeleton only.', action='store_true')
    args = parser.parse_args()
    video, json_folder, out_video_path, out_images_path, use_coco_format, temp, no_bg \
        = args.videofile.name, args.json, args.outfile, args.out_images, args.coco_format, args.temp, args.noBG

    if not os.path.exists(json_folder):
        print("Json folder not found!")
        exit(-1)

    _, out_extension = os.path.splitext(out_video_path)
    if out_extension != ".mp4":
        print("So far only .mp4 extension allowed for outfile!")
        exit(-1)
    out_folder, _ = os.path.split(out_video_path)
    os.makedirs(out_folder, exist_ok=True)
    if out_images_path is not None:
        os.makedirs(out_images_path, exist_ok=True)

    if not temp:
        temp = tempfile.mkdtemp()

    used_format = OpenposeOutputFormat.COCO if use_coco_format else OpenposeOutputFormat.BODY_25
    color_video(json_folder, video, out_video_path, temp_folder=temp, out_images=out_images_path,
                max_frames=args.maxframes, openpose_format=used_format, no_bg=no_bg)
