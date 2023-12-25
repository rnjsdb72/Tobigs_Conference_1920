import os
import argparse
import pickle
from PIL import Image

from utils.score import score_func
from utils.cal_coords2combine import cal_coords_func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir")
    args = parser.parse_args()

    files = os.listdir(args.inputdir)
    for file in files:
        img_bgr = Image.open(os.path.join(args.inputdir, file, "inpainted", "inpaint_out.png"))
        img_human = Image.open(os.path.join(args.inputdir, file, "composition.png"))
        with open(os.path.join(args.inputdir, file, "keypoints", "keypoints.pkl"), "rb") as f:
            kpts = pickle.load(f)
            
        score = score_func()
        coords_to_combine = cal_coords_func()
        
        img_bgr.paste(img_human, coords_to_combine)
        
        img_bgr.save(os.path.join(args.inputdir, file, "final_output.png"))
        print(f"{file}'s Score: ", score)