from roboflow import Roboflow
from dotenv import load_dotenv
from helpers import make_dataset_dir, git_ignore

import os
import json
import shutil


def download_roboflow_datasets(dataset_json, rf, dest_dir=None):
    """Download datasets from Roboflow to local directory."""
    with open(dataset_json) as f:
        datasets = json.load(f)  # load datasets from json file
        for rf_dataset in datasets["Roboflow"]:  # iterate through Roboflow datasets
            project = rf.workspace(rf_dataset["workspace"]).project(rf_dataset["project"])
            version = project.version(rf_dataset["version"])
            dataset = version.download("yolov8", "./" + rf_dataset["project"] + "/")
            git_ignore(rf_dataset["project"])
            if dest_dir:
                process_dataset("./"+rf_dataset["project"]+"/", dest_dir)



def process_dataset(dir, dest_dir):
    """
    Compile downloaded datasets.
    Takes download directory as input.
    """
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