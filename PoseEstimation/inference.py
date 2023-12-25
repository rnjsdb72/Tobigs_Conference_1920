import argparse
import os
from glob import glob
from tqdm import tqdm
import pickle
from PIL import Image

from mmpose.apis import inference_topdown, init_model
from mmpose.apis import visualize

def estimate(model, img_path, args):
    res = inference_topdown(model, img_path)

    preds = res[0].pred_instances

    keypoints = preds.keypoints
    keypoint_scores = preds.keypoint_scores

    res = visualize(
        img_path,
        keypoints,
        keypoint_scores,
        metainfo=args.vis_cfg,
        show=False)

    img_res = res[:, int(res.shape[1]/2):, :]
    
    return res, img_res

def save_res(img_res, fn, args):
    savedir = os.path.join(args.output_dir, fn, "keypoints")
    os.makedirs(savedir, exist_ok=True)
    
    with open(os.path.join(savedir, "keypoints.pkl"), "wb") as f:
        pickle.dump(res, f)
        
    img_res = Image.fromarray(img_res)
    img_res.save(os.path.join(savedir, "keypoints.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-cfg")
    parser.add_argument("--ckpt")
    parser.add_argument("--input-dir")
    parser.add_argument("--apply-matted", default=False)
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--vis-cfg")
    args = parser.parse_args()

    model = init_model(args.model_cfg, args.ckpt, device=args.device)
    
    files = glob(os.path.join(args.input_dir, "*"))
    for img_path in tqdm(files):
        if args.apply_matted:
            fn = img_path.split("/")[-1]
            img_path = os.path.join(img_path, "composition.png")
        else:
            fn = os.path.basename(img_path)
            fn = fn.split(".")[0]
        res, img_res = estimate(model, img_path, args)
        save_res(img_res, fn, args)