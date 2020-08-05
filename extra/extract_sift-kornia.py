import argparse

import cv2

import itertools

import kornia

import numpy as np

import os

import torch

import torchvision.transforms as transforms

import tqdm


def get_transforms():
    transform = transforms.Compose([
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)),
        transforms.Lambda(lambda x: cv2.resize(x, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)),
        transforms.Lambda(lambda x: np.reshape(x, (32, 32, 1))),
        transforms.ToTensor()
    ])

    return transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to the dataset'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='path to the images'
    )
    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='batch size'
    )

    args = parser.parse_args()

    # Create output folder.
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    # Create SIFT.
    SIFT = kornia.feature.SIFTDescriptor(patch_size=32, rootsift=False)

    transforms = get_transforms()

    # Prediction loop.
    for sequence in tqdm.tqdm(list(sorted(os.listdir(args.dataset_path)))):
        # Create output sub-folder.
        current_output_path = os.path.join(args.output_path, sequence)
        if not os.path.exists(current_output_path):
            os.mkdir(current_output_path)
        
        # for pair in itertools.product(['e', 'h', 't'], map(str, range(1, 6))):
        #     # Read patches from disk.
        #     patches_path = os.path.join(args.dataset_path, sequence, pair[0] + pair[1] + '.png')
        #     patches = cv2.imread(patches_path)
        #     patches = np.reshape(patches, (-1, 65, 65, 3))
            
        #     # Extract descriptors.
        #     descriptors = np.zeros((len(patches), 128), dtype=np.float32)
        #     for i in range(0, len(patches), args.batch_size):
        #         data_a = patches[i : i + args.batch_size]
        #         data_a = torch.stack(
        #             [transforms(patch) for patch in data_a]
        #         )
        #         # Predict
        #         out_a = SIFT(data_a)
        #         descriptors[i : i + args.batch_size] = out_a.cpu().detach().numpy()
            
        #     # Save descriptors to disk.
        #     with open(os.path.join(current_output_path, pair[0] + pair[1] + '.csv'), 'w') as f:
        #         for idx in range(descriptors.shape[0]):
        #             f.write(';'.join(map(str, descriptors[idx])) + '\n')

        # Read patches from disk.
        patches_path = os.path.join(args.dataset_path, sequence, 'ref.png')
        patches = cv2.imread(patches_path)
        patches = np.reshape(patches, (-1, 65, 65, 3))
        
        # Extract descriptors.
        descriptors = np.zeros((len(patches), 128), dtype=np.float32)
        for i in range(0, len(patches), args.batch_size):
            data_a = patches[i : i + args.batch_size]
            data_a = torch.stack(
                [transforms(patch) for patch in data_a]
            )
            # Predict
            out_a = SIFT(data_a)
            descriptors[i : i + args.batch_size] = out_a.cpu().detach().numpy()
        
        # Save descriptors to disk.
        with open(os.path.join(current_output_path, 'ref.csv'), 'w') as f:
            for idx in range(descriptors.shape[0]):
                f.write(';'.join(map(str, descriptors[idx])) + '\n')