# ### built-in deps
import os
import sys
import json
from operator import index
import pathlib
from PIL import Image
import shutil
from shutil import copyfile

# ### external deps
from skimage.exposure import match_histograms
from skimage.io import imsave
import numpy as np
import pandas as pd


# Configurações
# =============
IS_ORIG = False


def search_in_file(folder):
    """
    Versão nova (atual) da busca pela melhor imagem.
    Considera a soma do Style Loss e do Content Loss (em conjunto).
    """
    # Dicionários de perdas (losses). 
    #   Cada elemente armazena uma lista com quatro elementos:
    #   [file+path, style_loss, content_loss, loss]
    losses_dict = {}

    # All folders started with sample
    dirs = sorted(pathlib.Path(f'{folder}').glob('sample*'))
    # All txt files inside the first folder in 'dirs' list
    files_dict = pathlib.Path(str(dirs[0])).rglob('*.txt')

    # ### create dict of images
    for fl in sorted(files_dict):
        img = str(fl.stem).split('IMG_')
        img_number = img[-1].split('.')[0]
        if img_number.split('_')[-1] == '2':
            continue
        losses_dict[img_number] = []
    
    # ### find all txt files saved during training
    files = sorted(list(pathlib.Path(folder).rglob("*.txt")))

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

        # >>>>>
        # O loss a ser minimizado é a soma do style e do content loss.
        loss = style_loss + content_loss

        # DEBUG
        ### print(losses_dict[img_number])

        # ### find the best losses
        if len(losses_dict[img_number]) == 0:
            losses_dict[img_number].append(str(txt_file)) # 0
            ### losses_dict[img_number].append( [style_loss, content_loss] )
            losses_dict[img_number].append(style_loss)    # 1
            losses_dict[img_number].append(content_loss)  # 2
            losses_dict[img_number].append(loss)          # 3
        else:
            if loss < losses_dict[img_number][3]:
                losses_dict[img_number][0] = str(txt_file)
                losses_dict[img_number][1] = style_loss
                losses_dict[img_number][2] = content_loss
                losses_dict[img_number][3] = loss

    with open(f'{folder}{os.path.sep}best_losses.json', 'w') as fp:
        json.dump(losses_dict, fp, indent=2)


def copy_samples(folder, dest_folder):
    """
    """
    # ### copy the best samples based on loss
    with open(f'{folder}{os.path.sep}search{os.path.sep}best_losses.json', 'r') as fp:
        samples = json.load(fp)

    pathlib.Path(dest_folder).mkdir(parents=True, exist_ok=True)

    for k, v in samples.items():
        img_path = v[0].split(os.path.sep)[0:-1]
        img_name = f'output_IMG_{k}.tif'
        img_path.append('results')
        img_path.append(img_name)

        img_full_path = os.path.sep.join(img_path)
        ### dest_folder = f'{folder}{os.path.sep}optim_result{os.path.sep}NST'
        ### pathlib.Path(dest_folder).mkdir(parents=True, exist_ok=True)

        # ### copying files
        print(f'Copying {img_full_path} to {dest_folder}{os.path.sep}{img_name}')
        copyfile(img_full_path, f'{dest_folder}{os.path.sep}{img_name}')

        if k.split('_')[1] == '1':
            # Copy the channel 2 (green). It is not in 'best_losses.json' file. 
            k_ = k[:-1] + '2'

            img_path = v[0].split(os.path.sep)[0:-1]
            img_name = f'output_IMG_{k_}.tif'
            img_path.append('results')
            img_path.append(img_name)

            img_full_path = os.path.sep.join(img_path)
            ### dest_folder = f'{folder}{os.path.sep}optim_result{os.path.sep}NST'
            ### pathlib.Path(dest_folder).mkdir(parents=True, exist_ok=True)

            # ### copying files
            print(f'Copying {img_full_path} to {dest_folder}{os.path.sep}{img_name} (*)')
            copyfile(img_full_path, f'{dest_folder}{os.path.sep}{img_name}')


def load_img(image_path):
    """
    """
    # ### load image
    image = Image.open(image_path) 
    image = np.array(image)

    return image


def save_CSV(path, dest_path):
    """
    """
    with open(f'{path}{os.path.sep}best_losses.json', 'r') as fp:
        samples = json.load(fp)

    # ### save json file info in comma-separated values (CSV)
    result_dict = {}
    for k, v in samples.items():
        iter_lr = v[0].split(os.path.sep)[0:-1][-1]
        iter_lr_info = iter_lr.split('_')
        img_name = f'IMG_{k}'
        iter = iter_lr_info[-2]
        lr = iter_lr_info[-1]
        s_loss = v[1]
        c_loss = v[2]
        l_loss = v[3]

        train_info = [iter, lr, s_loss, c_loss, l_loss]

        result_dict[img_name] = train_info

    df_columns = ['Iteration', 'Learning Rate', 'Style Loss', 'Content Loss', 'Loss']

    df = pd.DataFrame.from_dict(result_dict, orient='index', columns=df_columns)
    ### df.to_csv('results.csv')
    ### df.to_csv(f'{folder}{os.path.sep}results_orig.csv')
    df.to_csv(f'{dest_path}')

    
def get_matched_samples_2(original_path, nst_path, folder):
    """
    """
    original_imgs = sorted(list(pathlib.Path(original_path).glob('*.tif')))
    nst_imgs = sorted(list(pathlib.Path(nst_path).glob('*.tif')))

    # ### make batches
    batches_original = [original_imgs[i:i+5] for i in range(0, len(original_imgs), 5)]
    batches_nst = [nst_imgs[i:i+5] for i in range(0, len(nst_imgs), 5)]

    # Cria a pasta de destino, se ela ainda não existe.
    pathlib.Path(f'{folder}').mkdir(parents=True, exist_ok=True)
    
    print("Generating matched images...")
    for i, (batch_orig, batch_nst) in enumerate(zip(batches_original, batches_nst)):
        print(f'\nBatch {i}:')
        print(batch_orig)
        print('----')
        print(batch_nst)

        for j, (path_orig, path_nst) in enumerate(zip(batch_orig, batch_nst)):

            img_orig = load_img(path_orig)            
            img_nst = load_img(path_nst)

            if j == 1:
                print(f'Processing imagens \'{path_orig}\' and \'{path_nst}\'... (*)')

                # O canal Green não é submetido ao histogram match.
                imsave(f'{folder}{os.path.sep}matched_{path_orig.stem}.png', img_orig)

            else:
                print(f'Processing imagens \'{path_orig}\' and \'{path_nst}\'...')

                matched = match_histograms(img_nst, img_orig, multichannel=True)

                imsave(f'{folder}{os.path.sep}matched_{path_orig.stem}.png', matched)


def get_matched_samples(original_path, nst_path, folder):
    """
    """
    original_imgs = sorted(list(pathlib.Path(original_path).glob('*.tif')))
    nst_imgs = sorted(list(pathlib.Path(nst_path).glob('*.tif')))

    # ### make batches
    batches_original = [original_imgs[i:i+5] for i in range(0, len(original_imgs), 5)]
    batches_nst = [nst_imgs[i:i+5] for i in range(0, len(nst_imgs), 5)]

    # Cria a pasta de destino, se ela ainda não existe.
    pathlib.Path(f'{folder}{os.path.sep}matched').mkdir(parents=True, exist_ok=True)
    
    print("generating matched images...")
    for i_, batch in enumerate(batches_original):
        print(f'\nBatch {i_}:')
        print(batch)
        for j, img_ in enumerate(batch):

            print(f'Processing imagens \'{batches_nst[i_][j]}\' and \'{img_}\'...')

            image = load_img(batches_nst[i_][j])            
            reference = load_img(img_)

            matched = match_histograms(image, reference, multichannel=True)

            ### print(matched.dtype, matched.min(), matched.max(), matched.mean())
            matched_2 = matched.astype(np.uint8)
            ### print(matched_2.dtype, matched_2.min(), matched_2.max(), matched_2.mean())
            
            imsave(f'{folder}{os.path.sep}matched{os.path.sep}matched_{img_.stem}.png', matched)


if __name__ in '__main__':


    if len(sys.argv) < 2:
        print(f'ERROR: Inform the path to the experiment folder! Exiting...')
        print('Done!')
        sys.exit()

    folder = sys.argv[1]

    print('\n\n' + ('=' * 40))
    print('(match_ok.py -- Starting new experiment...')

    if os.path.exists(f'{folder}{os.path.sep}search'):
        print('\nSearching and matching...')
        # Se dentro da pasta 'folder' existe uma pasta 'search'.
        # Localizar as melhores imagens de acordo com o Loss
        search_in_file(f'{folder}{os.path.sep}search')
        copy_samples(folder, f'{folder}{os.path.sep}optim_result{os.path.sep}nst')

        get_matched_samples_2(f'{folder}{os.path.sep}search{os.path.sep}original', 
                                f'{folder}{os.path.sep}optim_result{os.path.sep}nst',
                                f'{folder}{os.path.sep}optim_result{os.path.sep}matched')
        
        save_CSV(f'{folder}{os.path.sep}search', f'{folder}{os.path.sep}search{os.path.sep}results.csv')

    else:
        print('Matching...')
        # Se não tiver uma pasta chamad 'search' dentro de folder, então é o resultado de 
        # treino.
        # --> Apenas aplicar histogram matching
        get_matched_samples_2(f'{folder}{os.path.sep}original{os.path.sep}results',
                              f'{folder}{os.path.sep}nst{os.path.sep}results', 
                              f'{folder}{os.path.sep}matched')
        
    if os.path.exists('./nohup.out'):
        shutil.move('./nohup.out', f'{folder}/nohup_match.out') 
    
    print("Done!")