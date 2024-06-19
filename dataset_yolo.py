from roboflow import Roboflow
from dotenv import load_dotenv
from helpers import make_dataset_dir, git_ignore

import os
import json
import random
import shutil


def download_roboflow_datasets(dataset_json, rf, dest_dir=None):
    """Download datasets from Roboflow to local directory."""
    with open(dataset_json) as f:
        datasets = json.load(f)  # load datasets from json file
        for rf_dataset in datasets["Roboflow"]:  # iterate through Roboflow datasets
            project = rf.workspace(rf_dataset["workspace"]).project(rf_dataset["project"])
            version = project.version(rf_dataset["version"])
            dataset = version.download("yolov5", "./" + rf_dataset["project"] + "/")
            git_ignore(rf_dataset["project"])
            adjust_labels(rf_dataset)
            if dest_dir:
                process_dataset("./"+rf_dataset["project"]+"/", dest_dir, rf_dataset["downsample"])


def adjust_labels(metadata):
    "Adjust classification labels"
    dest_dir = "./" + metadata["project"] + "/"
    for label in metadata["labels"]:
        for d in metadata["labels"][label]:
            dir = d["src"]
            if len(d["imgs"]) > 0:
                img_dir = d["imgs"]
            else:
                img_dir = os.listdir(dest_dir + dir + "/images/")
            for image in img_dir:
                content = 0
                file = image[:-3] + "txt"
                print(file)
                with open(dest_dir + dir + "/labels/" + file, "r") as f:
                    content = f.readlines()
                if content != 0:
                    for l in range(len(content)):
                        if len(content[l]) > 0:
                            content[l] = label + content[l][1:]
                with open(dest_dir + dir + "/labels/" + file, "w") as f:
                    f.writelines(content)




def process_dataset(dir, dest_dir, downsample=1.0):
    """
    Compile downloaded datasets.
    Takes download directory as input.
    """
    if random.random() < downsample: 
        src_dirs = [dir + "train/", dir + "test/", dir + "valid/"]

        if not os.path.exists(dest_dir):
            make_dataset_dir(dest_dir)
        
        for src_dir in src_dirs:
            tmp = src_dir.split("/")[2] + "/"
            for image in os.listdir(src_dir + "images/"):
                shutil.copy(src_dir + "images/" + image, dest_dir + tmp+ "images/")
            for label in os.listdir(src_dir + "labels/"):
                shutil.copy(src_dir + "labels/" + label, dest_dir + tmp + "labels/")

def main():
    #Load environment variables from .env file
    load_dotenv()

    #Get Roboflow API key from environment variables
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    download_roboflow_datasets("datasets.json", rf, "./yolov8_compiled_dataset/")


if __name__ == "__main__":
    main()