from roboflow import Roboflow
from dotenv import load_dotenv

import os
import json



def download_roboflow_datasets(dataset_json, rf, dest_dir=None):
    """Download datasets from Roboflow to local directory."""
    with open(dataset_json) as f:
        datasets = json.load(f)  # load datasets from json file
        for rf_dataset in datasets["Roboflow"]:  # iterate through Roboflow datasets
            project = rf.workspace(rf_dataset["workspace"]).project(rf_dataset["project"])
            version = project.version(rf_dataset["version"])
            dataset = version.download("yolov5", "../" + rf_dataset["project"] + "/")
            #git_ignore(rf_dataset["project"])
            #if dest_dir:
                #process_dataset("./"+rf_dataset["project"]+"/", dest_dir, rf_dataset["downsample"])

def main():
    #Load environment variables from .env file
    load_dotenv()

    #Get Roboflow API key from environment variables
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)

    download_roboflow_datasets("datasets.json", rf, "./yolov8_compiled_dataset/")