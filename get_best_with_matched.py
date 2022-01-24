# ### built-in deps
import json
from operator import index
import pathlib
from PIL import Image
from shutil import copyfile

# ### external deps
from skimage.exposure import match_histograms
from skimage.io import imsave
import numpy as np
import pandas as pd

# ### local deps
from lib.utils import normalize

def search_in_file(folder):
    # ### find all txt files saved during training
    files = sorted(list(pathlib.Path(folder).rglob("*.txt")))
    losses_dict = {}
    dirs = sorted(pathlib.Path(f'{folder}').glob('sample*'))
    files_dict = pathlib.Path(str(dirs[0])).rglob('*.txt')

    # ### create dict of images
    for fl in sorted(files_dict):
        img = str(fl.stem).split('IMG_')
        img_number = img[-1].split('.')[0]
        if img_number.split('_')[-1] == '2':
            continue
        losses_dict[img_number] = []
        

    for txt_file in files:

        img = str(txt_file.stem).split('IMG_')
        img_number = img[-1].split('.')[0]
        if img_number.split('_')[-1] == '2':
            continue
        with open(txt_file, 'r') as fp:
            data = fp.readlines()

        last_loss = data[-1].split(' ')

        style_loss = float(last_loss[2])
        content_loss = float(last_loss[5].split('\n')[0])

        # ### find the best losses
        if len(losses_dict[img_number]) == 0:
            losses_dict[img_number].append(str(txt_file))
            losses_dict[img_number].append( [style_loss, content_loss] )
        else:
            if style_loss < losses_dict[img_number][1][0]:
                if content_loss < losses_dict[img_number][1][1]:
                    losses_dict[img_number][0] = str(txt_file)
                    losses_dict[img_number][1][0] = style_loss
                    losses_dict[img_number][1][1] = content_loss

    with open(f'{folder}/best_losses.json', 'w') as fp:
        json.dump(losses_dict, fp, indent=2)
    
def copy_samples(folder):
    # ### copy the best samples based on loss
    with open(f'{folder}/search/best_losses.json', 'r') as fp:
        samples = json.load(fp)

    for k, v in samples.items():
        img_path = v[0].split('/')[0:-1]
        img_name = f'output_IMG_{k}.tif'
        img_path.append('results')
        img_path.append(img_name)

        img_full_path = '/'.join(img_path)
        dest_folder = f'{folder}/optim_result/NST'
        pathlib.Path(dest_folder).mkdir(parents=True, exist_ok=True)

        # ### copying files
        print(img_full_path)
        copyfile(img_full_path, f'{dest_folder}/{img_name}')

def load_img(image_path, normalization=False):
    # ### load image
    image = Image.open(image_path) 
    image = np.array(image)

    if normalization:  
        pix = normalize(image)
        pix_rgb = np.stack((pix,)*3, axis=-1).transpose((0, 1, 2))

        image = Image.fromarray(np.uint8(pix_rgb)).convert('RGB')

    return image

def save_CSV(path):
    with open(f'{path}/best_losses.json', 'r') as fp:
        samples = json.load(fp)

    # ### save json file info in comma-separated values (CSV)
    result_dict = {}
    for k, v in samples.items():
        iter_lr = v[0].split('/')[0:-1][-1]
        iter_lr_info = iter_lr.split('_')
        img_name = f'IMG_{k}'
        iter = iter_lr_info[-2]
        lr = iter_lr_info[-1]
        s_loss = v[1][0]
        c_loss = v[1][1]

        train_info = [iter, lr, s_loss, c_loss]

        result_dict[img_name] = train_info

    df_columns = ['Iteration', 'Learning Rate', 
                      'Style Loss', 'Content Loss']

    df = pd.DataFrame.from_dict(result_dict, orient='index', columns=df_columns)
    df.to_csv('results.csv')

def get_matched_samples(folder, original_path, nst_path):
    original_imgs = sorted(list(pathlib.Path(original_path).glob('*.tif')))
    nst_imgs = sorted(list(pathlib.Path(nst_path).glob('*.tif')))

    # ### make batches
    batches_original = [original_imgs[i:i+5] for i in range(0, len(original_imgs), 4)]
    batches_nst = [nst_imgs[i:i+5] for i in range(0, len(nst_imgs), 4)]

    print("generating matched images...")
    for i_, batch in enumerate(batches_original):
        for j, img_ in enumerate(batch):

            image = load_img(batches_nst[i_][j])            
            reference = load_img(img_)

            matched = match_histograms(image, reference, multichannel=True)
            pathlib.Path(f'{folder}/optim_result/matched').mkdir(parents=True, exist_ok=True)
            imsave(f'{folder}/optim_result/matched/matched_{img_.stem}.png', matched)

if __name__ in '__main__':
    folder = './experiments'
    search_in_file(f'{folder}/search')
    copy_samples(folder)
    get_matched_samples(folder, f'{folder}/search/original', f'{folder}/optim_result/NST')
    save_CSV(f'{folder}/search')
    print("Done!")