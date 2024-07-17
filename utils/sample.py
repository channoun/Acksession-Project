from roboflow import Roboflow

rf = Roboflow(api_key="UoagwIR5U5GWnbgWV62P")
project = rf.workspace("graduationproject-t4eec").project("old-iraq-number-plate")
version = project.version(4)
dataset = version.download("tfrecord", "./tensors")
