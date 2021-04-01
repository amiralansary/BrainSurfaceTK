import os
from subprocess import call


# pip3 install --upgrade --force-reinstall torch torchvision
# pip install torch-scatter --upgrade --force-reinstall -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
# pip install torch-sparse --upgrade --force-reinstall -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
# pip install torch-cluster --upgrade --force-reinstall -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
# pip install torch-spline-conv --upgrade --force-reinstall -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
# pip install --upgrade --force-reinstall torch-geometric

###############################################################################
import glob, re


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def tryint(s):
    try:
        return int(s)
    except:
        return s


def listFiles(dirpath, dirnames):
    curpath = os.getcwd()
    os.chdir(dirpath)
    f = glob.glob(dirnames)
    f.sort(key=alphanum_key)
    os.chdir(curpath)
    return f


###############################################################################

def make_dirs(path):
    """make a new directory if path does not exist"""
    if os.path.exists(path):
        return
    os.makedirs(path)


###############################################################################

# age prediction
surf_dir = '/vol/biomedic3/aa16914/dhcp_brain_emma/data/age_prediction/data/ico6_white_surfaces'
feat_dir = '/vol/biomedic3/aa16914/dhcp_brain_emma/data/age_prediction/data/merged'
save_dir = '/vol/biomedic3/aa16914/dhcp_brain_emma/data/age_prediction/data/surf_feat_vtp'

# # parcellation/segmentation
# surf_dir = '/vol/biomedic3/aa16914/dhcp_brain_emma/data/parcellation/ico6_white_surfaces'
# feat_dir = '/vol/biomedic3/aa16914/dhcp_brain_emma/data/parcellation/features_merged'
# label_dir = '/vol/biomedic3/aa16914/dhcp_brain_emma/data/parcellation/labels'
# save_dir = '/vol/biomedic3/aa16914/dhcp_brain_emma/data/parcellation/surf_feat_label_vtp'

###############################################################################

make_dirs(save_dir)
surf_files = listFiles(surf_dir, '*.gii')
feat_files = listFiles(feat_dir, '*.gii')
# label_files = listFiles(label_dir, '*.gii')
ext = 'vtp'

print('number of files ', len(surf_files))

for idx, _ in enumerate(surf_files):
    filename = surf_files[idx]

    names = filename.split('_')
    subject_id = names[0] + '_' + names[1] + '_' + names[2][0].capitalize()

    # print(filename)

    found = 0
    feat_filename = None
    for feat_file in feat_files:
        if subject_id in feat_file:
            found = 1
            feat_filename = feat_file
            break

    if found != 0:
        pass
    else:
        print('file not found', subject_id)
        continue



    surf_file = os.path.join(surf_dir, filename)
    feat_file = os.path.join(feat_dir, feat_filename)


    # break


    # out_file = os.path.join(save_dir, surf_files[idx][:-3] + 'shape.'+ ext)
    # add label for segmentation
    out_file = os.path.join(save_dir, surf_files[idx][:-3] + 'shape.label.' + ext)

    cmd = ['mirtk', 'copy-pointset-attributes', feat_file, surf_file, out_file]
    print(5 * '-', idx + 1, surf_files[idx][:-3], 5 * '-')
    print(cmd)
    call(cmd)

    # # for parcellation add labels too
    # label_file = os.path.join(label_dir, label_files[idx])
    # cmd = ['mirtk', 'copy-pointset-attributes', label_file, out_file, out_file]
    # print(cmd)
    # call(cmd)
