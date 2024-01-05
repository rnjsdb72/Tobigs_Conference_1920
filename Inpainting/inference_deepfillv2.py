import os
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as T

from model.networks_deepfillv2 import Generator

def parse_args():
    parser = argparse.ArgumentParser(description='Test inpainting')
    parser.add_argument("--image", type=str,
                        default="examples/inpaint/case1.png", help="path to the image file")
    parser.add_argument("--mask", type=str,
                        default="examples/inpaint/case1_mask.png", help="path to the mask file")
    parser.add_argument("--out", type=str,
                        default="examples/inpaint/case1_out_test.png", help="path for the output file")
    parser.add_argument("--checkpoint", type=str,
                        default="./ckpt/states_pt_places2.pth", help="path to the checkpoint file")
    args = parser.parse_args()
    
    return args

def inference(fn_img, fn_mask, generator, device, out):
    # load image and mask
    image = Image.open(fn_img)
    mask = Image.open(fn_mask)

    # prepare input
    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    image = (image*2 - 1.).to(device)  # map image values to [-1, 1] range
    mask = (mask > 0.5).to(dtype=torch.float32,
                           device=device)  # 1.: masked 0.: unmasked

    image_masked = image * (1.-mask)  # mask image

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x*mask],
                  dim=1)  # concatenate channels

    with torch.inference_mode():
        _, x_stage2 = generator(x, mask)

    # complete image
    image_inpainted = image * (1.-mask) + x_stage2 * mask

    # save inpainted image
    img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
    img_out = img_out.to(device='cpu', dtype=torch.uint8)
    img_out = Image.fromarray(img_out.numpy())
    img_out.save(out)

def main():
    args = parse_args()
    
    out = args.out
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up network
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)

    generator_state_dict = torch.load(args.checkpoint, map_location=device)['G']
    generator.load_state_dict(generator_state_dict, strict=True)
    
    if os.path.isdir(args.image):
        files = glob(f"{args.image}/*")
        for fn_img in tqdm(files):
            img_nm = os.path.basename(fn_img).split(".")[0]
            fn_mask = os.path.join(out, img_nm, "alpha.png")
            
            save_root = os.path.join(out, img_nm, "inpaint_out.png")
            
            inference(fn_img, fn_mask, generator, device, save_root)
    else:
        fn_img = args.image
        img_nm = os.path.basename(fn_img).split(".")[0]
        fn_mask = os.path.join(out, img_nm, "alpha.png")
        
        save_root = os.path.join(out, img_nm, "inpaint_out.png")
        
        inference(fn_img, fn_mask, generator, device, save_root)

if __name__ == '__main__':
    main()
