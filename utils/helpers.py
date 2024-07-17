import os

def make_dataset_dir(dir):
    os.makedirs(dir + "train/images/")
    os.makedirs(dir + "train/labels/")

    os.makedirs(dir + "test/images/")
    os.makedirs(dir + "test/labels/")

    os.makedirs(dir + "valid/images/")
    os.makedirs(dir + "valid/labels/")

def git_ignore(ignored_folder):
    with open(".gitignore", "a") as f:
        f.write("\n" + ignored_folder + "/")

