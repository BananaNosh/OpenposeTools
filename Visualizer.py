import cv2
import zipfile
import json
from types import SimpleNamespace as Namespace
import numpy as np
import os
import re
import argparse
import tempfile
import sys
from skvideo import io
from moviepy.editor import VideoFileClip, concatenate_videoclips


HD_THRESHOLD = 1300
ALL_THICKNESS = [[2, 1, 2, 2], [8, 4, 6, 6]]
POSE_SEQ = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
            (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
POSE_COLORS = [[255, 0, 85], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0]] \
              + [[85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255]] \
              + [[0, 85, 255], [0, 0, 255], [255, 0, 170], [170, 0, 255], [255, 0, 255]]
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
ALL_SEQS = [POSE_SEQ, FACE_SEQ, HAND_SEQ, HAND_SEQ]
ALL_COLORS = [POSE_COLORS, FACE_COLORS, HAND_COLORS, HAND_COLORS]


def color_video(frames_json_zip, vid_file, out_file, temp_folder, max_frames=None, frame_range=None):
    video_capture = cv2.VideoCapture(vid_file)
    output_without_ext, output_type = os.path.splitext(out_file)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    colored_frames = []
    with zipfile.ZipFile(frames_json_zip) as zip_file:
        json_files = [file for file in zip_file.namelist() if file.endswith(".json")]
        json_files.sort()
        json_count = len(json_files)
        frame_count = min(frame_count, json_count)
        splitted = max_frames and max_frames < frame_count
        if splitted:
            temp_folder = generate_temp_folder(temp_folder)
        temp_name = os.path.join(temp_folder, os.path.basename(output_without_ext))
        json_files = json_files[:frame_count]
        enumerating = enumerate(json_files) if not frame_range \
            else zip(frame_range, json_files[frame_range.start:frame_range.stop:frame_range.step])
        if frame_range:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_range.start)
        video_number = None
        for i, file_name in enumerating:
            if i % 10 == 0:
                print("{}/{} frames ready       ".format(i, frame_count), end='\r')
                sys.stdout.flush()
            video_number = (i + 1) // max_frames if max_frames else None
            ret, canvas = video_capture.read()
            if canvas is None:
                print("")
                print("No canvas for file: ", file_name)
                break
            written_canvas = canvas_for_frame(canvas, file_name, zip_file)
            canvas = cv2.addWeighted(canvas, 0.1, written_canvas, 0.9, 0)
            colored_frames.append(canvas[:, :, [2, 1, 0]])
            if max_frames and (i + 1) % max_frames == 0 and max_frames != frame_count:
                colored_frames = np.array(colored_frames)
                write_file(colored_frames, fps, output_type, temp_name, video_number)
                colored_frames = []
    if len(colored_frames) > 0:
        colored_frames = np.array(colored_frames)
        video_number = video_number + 1 if splitted else None
        write_file(colored_frames, fps, output_type, temp_name, video_number)
    if splitted:
        combine_videos(out_file, temp_folder, True)


def canvas_for_frame(current_canvas, frame_json, zip_file):
    hd = current_canvas.shape[1] > HD_THRESHOLD
    with zip_file.open(frame_json, "r") as json_file:
        json_obj = json.loads(json_file.read().decode("utf-8"), object_hook=lambda d: Namespace(**d))
    canvas_copy = current_canvas.copy()
    people = json_obj.people
    for person in people:
        pose_key_points = np.array(person.pose_keypoints_2d).reshape(-1, 3)
        face_key_points = np.array(person.face_keypoints_2d).reshape(-1, 3)
        hand_left_key_points = np.array(person.hand_left_keypoints_2d).reshape(-1, 3)
        hand_right_key_points = np.array(person.hand_right_keypoints_2d).reshape(-1, 3)
        all_key_points = [pose_key_points, face_key_points, hand_left_key_points, hand_right_key_points]
        thick = ALL_THICKNESS[0] if not hd else ALL_THICKNESS[1]
        for key_points, seq, colors, thickness in zip(all_key_points, ALL_SEQS, ALL_COLORS, thick):
            for joints, color in zip(seq, colors):
                points = key_points[joints, :]
                x = points[:, 0]
                y = points[:, 1]
                c = points[:, 2]
                if any(c < 0.05):
                    continue
                cv2.line(canvas_copy, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), color, thickness)
                point_size = thickness + (3 if not hd else 12)
                cv2.line(canvas_copy, (int(x[0]), int(y[0])), (int(x[0]), int(y[0])), color, thickness=point_size)
    current_canvas = cv2.addWeighted(current_canvas, 0.1, canvas_copy, 0.9, 0)
    return current_canvas


def generate_temp_folder(temp_folder):
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
    elif len(os.listdir(temp_folder)) > 0:
        temp_folder = tempfile.mkdtemp(dir=temp_folder)
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
    parser.add_argument("json", type=argparse.FileType(mode="r"),
                        help="the zip of json files with the data for each frame")
    parser.add_argument("outfile", help='the output file')
    parser.add_argument("-t --temp", metavar='tempfolder', dest="temp", help="folder for saving the temp files")
    args = parser.parse_args()
    video, json_zip, out, temp = args.videofile.name, args.json.name, args.outfile, args.temp
    _, out_extension = os.path.splitext(out)
    if out_extension != ".mp4":
        print("So far only .mp4 extension allowed for outfile!")
        exit(-1)
    if not temp:
        temp = tempfile.mkdtemp()
    color_video(json_zip, video, out, temp_folder=temp, max_frames=100, frame_range=range(500))
