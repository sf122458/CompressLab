import os
import sys
import wget
import argparse
import zipfile
import tarfile
import py7zr
from typing import Union

DATASET_INFO = {
    "Vimeo90k_septuplet": {
        "url": "http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip",
    },
    "Vimeo90k_triplet": {
        "url": "http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip",
    },
    "DIV2K_val": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    },
    "CLIC2020_professional_train": {
        "url": "https://storage.googleapis.com/clic_datasets/clic2020_professional_train.zip",
    },
    "CLIC2020_mobile_train": {
        "url": "https://storage.googleapis.com/clic_datasets/clic2020_mobile_train.zip",
    },
    "CLIC2020_professional_val": {
        "url": "https://storage.googleapis.com/clic_datasets/clic2020_professional_valid.zip",
    },
    "CLIC2020_mobile_val": {
        "url": "https://storage.googleapis.com/clic_datasets/clic2020_mobile_valid.zip",
    },
    "CLIC2020_professional_test": {
        "url": "https://storage.googleapis.com/clic_datasets/clic2020_professional_test.zip",
    },
    "CLIC2020_mobile_test": {
        "url": "https://storage.googleapis.com/clic_datasets/clic2020_mobile_test.zip",
    },
    "Tecnick-100": {
        "url": "https://sourceforge.net/projects/testimages/files/OLD/OLD_SAMPLING/testimages.zip",
        "prefix": "RGB_OR_1200x1200",
    },
    "Tecnick-40": {
        "url": "https://sourceforge.net/projects/testimages/files/SAMPLING/8BIT/RGB/SAMPLING_8BIT_RGB_1200x1200.tar.bz2",
        "prefix": "C00C00",
    },
    "UVG": {
        "url": [
            "https://ultravideo.fi/video/Beauty_1920x1080_120fps_420_8bit_YUV_RAW.7z",
            "https://ultravideo.fi/video/Bosphorus_1920x1080_120fps_420_8bit_YUV_RAW.7z",
            "https://ultravideo.fi/video/HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW.7z",
            "https://ultravideo.fi/video/Jockey_1920x1080_120fps_420_8bit_YUV_RAW.7z",
            "https://ultravideo.fi/video/ReadySetGo_1920x1080_120fps_420_8bit_YUV_RAW.7z", 
            "https://ultravideo.fi/video/ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW.7z", 
            "https://ultravideo.fi/video/YachtRide_1920x1080_120fps_420_8bit_YUV_RAW.7z",
        ],
    }
}

archieve_methods = {
    zipfile.ZipFile: {
        'namelist': lambda ref: ref.namelist(),
        'getmembers': lambda ref: ref.infolist(),
        'getname': lambda member: member.filename,
        'setname': lambda member, name: setattr(member, 'filename', name)
    },
    tarfile.TarFile: {
        'namelist': lambda ref: ref.getnames(),
        'getmembers': lambda ref: ref.getmembers(),
        'getname': lambda member: member.name,
        'setname': lambda member, name: setattr(member, 'name', name)
    },
}



def dataset_preparation(dataset: str, remove: bool = False):
    os.makedirs(dataset, exist_ok=True)
    
    url = DATASET_INFO[dataset]["url"]

    if isinstance(url, list):   # temporary solution for UVG dataset
        for u in url:
            if not os.path.exists(u.split('/')[-1]):
                wget.download(u)
        
            compressed_filename = u.split('/')[-1]
            with py7zr.SevenZipFile(compressed_filename, mode='r') as ref:
                all_files = ref.getnames()
        
                yuv_files = [
                    file for file in all_files 
                    if file.lower().endswith('.yuv')
                ]
                ref.extract(dataset, yuv_files)
            if remove:
                os.remove(compressed_filename)
        return
    
    
    # Single URL case                
    compressed_filename = url.split('/')[-1]
    prefix = DATASET_INFO[dataset].get("prefix", None)

    if not os.path.exists(compressed_filename):
        print(f"Downloading {compressed_filename}...")
        wget.download(url)
        print("Download completed.")
    
    if len(os.listdir(dataset)) > 0:
        print(f"Directory {dataset} is not empty. Skipping extraction.")
        return
    
    context = zipfile.ZipFile(compressed_filename, "r") if compressed_filename.endswith('.zip') \
        else tarfile.open(compressed_filename, 'r:bz2')
    with context as ref:
        methods = archieve_methods[type(ref)]
        file_list = methods['namelist'](ref)
        
        root_dirs = set()
        for file_path in file_list:
            parts = file_path.split('/')
            if len(parts) > 1:
                root_dirs.add(parts[0])
        
        
        def process(member: Union[zipfile.ZipInfo, tarfile.TarInfo]) -> bool:
            filename = methods['getname'](member)
            if prefix is not None:
                if prefix in filename and filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    new_name = os.path.basename(filename)
                    methods['setname'](member, new_name)
                    return True
            else:
                if len(root_dirs) == 1:
                    root_dir = list(root_dirs)[0]
                    if filename.startswith(root_dir + '/'):
                        # Remove the root directory from the path
                        new_name = filename[len(root_dir) + 1:]
                        if new_name:
                            methods['setname'](member, new_name)
                            return True
                else:
                    return True
            return False
        
        for member in methods['getmembers'](ref):
            if process(member):
                ref.extract(member, dataset)
        
    print(f"Extraction {compressed_filename} completed.")
    
    if remove:
        print(f"Removing {compressed_filename}...")
        os.remove(compressed_filename)
        print(f"Removal {compressed_filename} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--list", action="store_true",
        help="List available datasets.",
    )
    parser.add_argument(
        "-r", "--remove", action="store_true",
        help="Remove the zip file after extraction.",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, nargs="+", 
        choices=list(DATASET_INFO.keys()),
        help="Specify which dataset(s) to download.",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available datasets:")
        for name in DATASET_INFO.keys():
            print(f"- {name}")
        exit(0)
        
    os.chdir(os.path.dirname(sys.argv[0]) or './')
    
    # Download the file
    for dataset in args.dataset:
        dataset_preparation(dataset, args.remove)