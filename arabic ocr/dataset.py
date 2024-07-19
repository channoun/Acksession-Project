from roboflow import Roboflow
from dotenv import load_dotenv

import os
import json
import shutil

def process_classes(dataset_dir, unwanted_classes):
    """Filter out classes unrelated to Arabic OCR"""
    dirs = ["train/", "test/", "valid/"]

    for dir in dirs:
        path = dataset_dir + dir + "labels/"
        for label in os.listdir(path):
            with open(path + label, "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    class_id = int(lines[i].split(" ")[0])
                    if class_id in unwanted_classes:
                        lines[i] = ""

            with open(path + label, "w") as f:
                f.writelines(lines)

def clean_dataset(dataset_dir):
    dirs = ["train/", "test/", "valid/"]
    for dir in dirs:
        path = dataset_dir + dir + "labels/"
        for label in os.listdir(path):
            f = open(path + label, "r")
            lines = f.readlines()
            f.close()
            if len(lines) == 0:
                os.remove(path + label)
                os.remove(dataset_dir + dir + "images/" + label[:-3] + "jpg")

def build_dirs(dataset):
    os.makedirs(dataset + "train/images/")
    os.makedirs(dataset + "train/labels/")

    os.makedirs(dataset + "test/images/")
    os.makedirs(dataset + "test/labels/")

    os.makedirs(dataset + "valid/images/")
    os.makedirs(dataset + "valid/labels/")


def compile_dataset(dataset_json):
    """Compile datasets from Roboflow into a single directory"""
    build_dirs("arabic_compiled_dataset/")
    with open(dataset_json, "r") as f:
        data = json.load(f)
        for dataset in data["Roboflow"]:
            dataset_dir = "../" + dataset["project"] + "/"
            dirs = ["train/", "test/", "valid/"]
            for dir in dirs:
                for image in os.listdir(dataset_dir + dir + "images/"):
                    shutil.copy(dataset_dir + dir + "images/" + image, "./arabic_compiled_dataset/" + dir + "images/")
                    label = image[:-3] + "txt"
                    label_file = dataset_dir + dir + "labels/" + label
                    if dataset["class_map"] is not None:
                        with open(label_file, "r") as label_f:
                            lines = label_f.readlines()
                            for i in range(len(lines)):
                                class_id = lines[i].split(" ")[0]
                                class_id = dataset["class_map"][class_id]
                                lines[i] = str(class_id) + " " + " ".join(lines[i].split(" ")[1:])
                        with open(label_file, "w") as label_f:
                            label_f.writelines(lines)
                    shutil.copy(dataset_dir + dir + "labels/" + label, "./arabic_compiled_dataset/" + dir + "labels/")


def remove_duplicates(dataset_dir):
    dirs = ["train/", "test/", "valid/"]
    for dir in dirs:
        images = os.listdir(dataset_dir + dir + "images/")
        for i in range(1, len(images)):
            old_name = images[i-1].split(".rf")[0]
            next_name = images[i].split(".rf")[0]
            label_name = images[i][:-3] + "txt"

            if old_name == next_name:
                os.remove(dataset_dir + dir + "images/" + images[i])
                os.remove(dataset_dir + dir + "labels/" + label_name)


def download_roboflow_datasets(dataset_json, rf):
    """Download datasets from Roboflow to local directory."""
    with open(dataset_json) as f:
        datasets = json.load(f)  # load datasets from json file
        for rf_dataset in datasets["Roboflow"]:  # iterate through Roboflow datasets
            project = rf.workspace(rf_dataset["workspace"]).project(rf_dataset["project"])
            version = project.version(rf_dataset["version"])
            dataset = version.download("yolov5", "./" + rf_dataset["project"] + "/")
            #process_classes("./" + rf_dataset["project"] + "/")

def main():
    #Load environment variables from .env file
    load_dotenv()

    #Get Roboflow API key from environment variables
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    download_roboflow_datasets("./arabic ocr/arabic_datasets.json", rf)

if __name__ == "__main__":
    main()