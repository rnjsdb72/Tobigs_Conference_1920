import os
import argparse
from tqdm import tqdm
import pickle
from PIL import Image
import numpy as np

from utils.position_cal import main_process, find_neck_center

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir")
    args = parser.parse_args()

    files = os.listdir(args.inputdir)
    for file in tqdm(files):
        img = Image.open(os.path.join(args.inputdir, file, "inpaint_out.png"))
        img_bgr = Image.open(os.path.join(args.inputdir, file, "inpaint_out.png"))
        img_human = Image.open(os.path.join(args.inputdir, file, "composition.png")).convert("RGBA")
        # img_alpha = Image.open(os.path.join(args.inputdir, file, "alpha.png"))
        with open(os.path.join(args.inputdir, file, "keypoints", "keypoints.pkl"), "rb") as f:
            kpts = pickle.load(f)
        
        img.paste(img_human, mask=img_human)
        mouth_point = kpts[0][0]
        left_eye_pos, right_eye_pos = kpts[0][1], kpts[0][2]
        left_shoulder_pos, right_shoulder_pos = kpts[0][5], kpts[0][6]
        left_knee_pos, right_knee_pos = kpts[0][13], kpts[0][14]
        left_ankle_pos, right_ankle_pos = kpts[0][15], kpts[0][16]
        pelvic = (kpts[0][11] + kpts[0][12])/2
        
        shoulder_points = [left_shoulder_pos, right_shoulder_pos]
        knee = left_knee_pos
        ankle = left_ankle_pos
        
        output = main_process(np.array(img), shoulder_points, mouth_point, left_eye_pos, right_eye_pos,
                     pelvic, knee, ankle)
        if len(output) == 4:
            optimal_left_eye, optimal_right_eye, best_score_eye, first_score_eye = output
            score = first_score_eye
            coords_to_combine = optimal_left_eye - left_eye_pos
            coords_to_combine = (int(coords_to_combine[0]), int(coords_to_combine[1]))
        elif len(output) == 5:
            optimal_center_neck, best_score_neck, best_score_bottom, first_score_neck, first_score_bottom = output
            score = first_score_neck + first_score_bottom
            coords_to_combine = optimal_center_neck - find_neck_center(shoulder_points, mouth_point)
            coords_to_combine = (int(coords_to_combine[0]), int(coords_to_combine[1]))
        
        img_bgr.paste(img_human, coords_to_combine, img_human)
        
        img_bgr.save(os.path.join(args.inputdir, file, "final_output.png"))
        print(f"{file}'s Score: ", score)