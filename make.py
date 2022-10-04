import os

def create_val_img_folder():
    '''
    This method is responsible for separating validation images into separate sub folders
    '''
    data_dir = "../../"
    dataset = "tiny-imagenet-200"
    dataset_dir = os.path.join(data_dir, dataset)
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

def main():
    create_val_img_folder()
    import pdb; pdb.set_trace()  # XXX BREAKPOINT


if __name__ == "__main__":
    main()
