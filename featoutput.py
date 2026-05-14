import glob
import os

import torch

from vggt.utils.load_fn import load_and_preprocess_images_square


def main():
    image_dir = r"D:/Code/My_Work/VGGT2RGT/examples/kitchen/images"
    model_path = "模型路径"

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    print(f"Found {len(image_paths)} images in {image_dir}")

    images, _ = load_and_preprocess_images_square(image_paths, target_size=518)
    print(f"Images loaded: shape = {images.shape}")

    model = VGGT.from_pretrained(model_path)
    model.eval()

    dtype = torch.float16
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        images_batch = images[None]
        features = model.get_last_block_features(images_batch, target_size=378)

    print(f"Output features shape: {features.shape}")
    print(f"  B (batch):       {features.shape[0]}")
    print(f"  S (frames):      {features.shape[1]}")
    print(f"  num_patches:     {features.shape[2]}")
    print(f"  2*embed_dim:     {features.shape[3]}")


if __name__ == "__main__":
    main()
