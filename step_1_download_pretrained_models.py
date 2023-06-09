import gdown
import argparse
import shutil
import os
import tarfile

def download_models(download_type):
    if download_type == 'best_models':
        url = 'https://drive.google.com/uc?id=1gRWin0BLPRDAAHjVI7h6DYOdSp2H8D2l'
    elif download_type == 'saved_models':
        url = 'https://drive.google.com/uc?id=1r-Xk80Na4EZ50XJJ9tzjqyUospRXwhbV'
    
    if os.path.exists(download_type):
        shutil.rmtree(download_type)

    gdown.download(url, output=download_type + '.tar', quiet=False)
    with tarfile.open(download_type + '.tar') as tar:
        tar.extractall()
    os.remove(download_type + '.tar')

####### DEFINE MAIN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download pretrained models')
    parser.add_argument('--download_the_best_models', default = "no", type=str, metavar='yes/no',
                        help="downloading the best models (default: no)")
    parser.add_argument('--download_all_saved_models', default = "no", type=str, metavar='yes/no',
                        help="downloading all saved models (default: no)")
    args = parser.parse_args()

    if args.download_the_best_models == 'yes':
        download_models('best_models')
    
    if args.download_all_saved_models == 'yes':
        download_models('saved_models')
