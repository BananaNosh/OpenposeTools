import zipfile
import json
import csv
from types import SimpleNamespace as Namespace
import numpy as np
import os


def json_zip_to_csv(zip_file_name):
    csv_name = zip_file_name[:-3] + "csv"
    data = []
    with zipfile.ZipFile(zip_file_name) as zip_file:
        for i, file_name in enumerate(zip_file.namelist()):
            if not file_name.endswith(".json"):
                print("Not a json file: {}".format(file_name))
                continue
            data.append((process_json_file(zip_file, file_name)))
        print(len(zip_file.namelist()))
    write_data_to_csv(csv_name, data)


def write_data_to_csv(filename, data):
    headers = ["people", "people_points", "faces", "face_points", "hands", "hand_points"]
    with open(filename, "w+", encoding="utf8", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(headers)
        for data_values in data:
            writer.writerow(data_values)


def process_json_file(zip, file_name):
    with zip.open(file_name, "r") as json_file:
        json_obj = json.loads(json_file.read(), object_hook=lambda d: Namespace(**d))
        people = json_obj.people
        people_count = len(people)
        recognized_faces = people_count
        recognized_hands = people_count * 2
        total_pose_points = 0
        total_face_points = 0
        total_hand_points = 0
        for person in people:
            pose_key_points = count_key_points(person.pose_keypoints_2d, 0.05)
            face_key_points = count_key_points(person.face_keypoints_2d, 0.4)
            left_hand_key_points = count_key_points(person.hand_left_keypoints_2d, 0.2)
            right_hand_key_points = count_key_points(person.hand_right_keypoints_2d, 0.2)
            if face_key_points == 0:
                recognized_faces -= 1
            if left_hand_key_points == 0:
                recognized_hands -= 1
            if right_hand_key_points == 0:
                recognized_hands -= 1
            total_pose_points += pose_key_points
            total_face_points += face_key_points
            total_hand_points += left_hand_key_points + right_hand_key_points
    return people_count, total_pose_points, recognized_faces, total_face_points, recognized_hands, total_hand_points


def count_key_points(key_points_array, threshold=0.0):
    return np.count_nonzero(np.array(key_points_array[0::3]) > threshold)


def evaluate_csvs(small_file, large_file):
    small_data = np.loadtxt(small_file, delimiter="\t", skiprows=1)
    large_data = np.loadtxt(large_file, delimiter="\t", skiprows=1)
    small_data_count = small_data.shape[0]
    large_data_count = large_data.shape[0]
    if large_data_count != small_data_count:
        print("not same size: small-{} and large-{}".format(small_data.shape, large_data.shape))
        more_large = large_data_count > small_data_count
        if more_large:
            large_data = large_data[:small_data_count]
        else:
            small_data = small_data[:large_data_count]
    total_gain = np.sum(large_data, axis=0) / np.sum(small_data, axis=0)
    gain = np.divide(large_data, small_data, out=np.zeros_like(small_data), where=small_data != 0)
    average_gain = np.mean(gain, axis=0)
    print("total_gain is \t\t{}".format(total_gain))
    print("average_gain is \t{}".format(average_gain))


if __name__ == '__main__':
    path = "./data/"
    for file_name in os.listdir(path):
        if file_name.endswith(".zip"):
            json_zip_to_csv(path + file_name)
    evaluate_csvs(path + "2905_small_json.csv", path + "2905_HD_reduced_to_-1_368.csv")

