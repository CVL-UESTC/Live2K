import os
from os import path as osp

from utils import scandir


def paired_paths_from_new_folder(root_folder):
    """Generate paths for the Live2K folder layout.

    Expected sample layout:
        sample/
            gt.png
            ref.png
            lq_sequence/*.png

    Returns:
        list[dict]: Each item contains gt_path, ref_path, and lq_paths.
    """
    assert osp.isdir(root_folder), f'Root folder {root_folder} does not exist.'

    paths = []
    subfolders = sorted([
        folder for folder in os.listdir(root_folder)
        if osp.isdir(osp.join(root_folder, folder))
    ])

    for folder in subfolders:
        folder_path = osp.join(root_folder, folder)
        lq_seq_folder = osp.join(folder_path, 'lq_sequence')
        gt_path = osp.join(folder_path, 'gt.png')
        ref_path = osp.join(folder_path, 'ref.png')

        if not osp.exists(gt_path):
            print(f'Warning: GT image not found at {gt_path}. '
                  f'Skipping folder {folder}.')
            continue
        if not osp.exists(ref_path):
            print(f'Warning: REF image not found at {ref_path}. '
                  f'Skipping folder {folder}.')
            continue
        if not osp.isdir(lq_seq_folder):
            print(f'Warning: lq_sequence folder not found at '
                  f'{lq_seq_folder}. Skipping folder {folder}.')
            continue

        lq_paths = [
            osp.join(lq_seq_folder, lq_file)
            for lq_file in sorted(scandir(lq_seq_folder, suffix='.png'))
        ]

        if len(lq_paths) != 9:
            print(f'Warning: LQ sequence in {lq_seq_folder} does not have '
                  f'9 frames. Found {len(lq_paths)}. Skipping.')
            continue

        paths.append({
            'lq_paths': lq_paths,
            'gt_path': gt_path,
            'ref_path': ref_path,
        })

    return paths
