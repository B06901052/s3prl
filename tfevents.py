import os
import shutil

os.makedirs("./tfevents", exist_ok=True)


for d in os.walk("./result/downstream"):
    for f in d[2]:
        if "tfevents" in f:
            path = "./tfevents/{}/".format(d[0][9:])
            os.makedirs(path, exist_ok=True)
            shutil.copy(os.path.join(d[0], f), os.path.join(path,f))
            
            
