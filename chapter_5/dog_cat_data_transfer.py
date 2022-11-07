import os, shutil

dataset_path = '/home/zlh/data/dogs_vs_cats'
source_train_path = os.path.join(dataset_path, 'train')
source_test_path = os.path.join(dataset_path, 'test1')

# data division
part_data_path = os.path.join(dataset_path, "part_data_path")
os.mkdir(part_data_path)
dog_cat = ['dog', 'cat']
for animal in dog_cat:
    train_dir = os.path.join(part_data_path, 'train', animal)
    verify_dir = os.path.join(part_data_path, 'verify', animal)
    test_dir = os.path.join(part_data_path, 'test', animal)
    os.system("mkdir -p {}".format(train_dir))
    os.system("mkdir -p {}".format(verify_dir))
    os.system("mkdir -p {}".format(test_dir))
    # training
    frame = ["{}.{}.jpg".format(animal, i) for i in range(1000)]
    for f in frame:
        srcf = os.path.join(source_train_path, f)
        tarf = os.path.join(train_dir, f)
        shutil.copyfile(srcf, tarf)
    # validation
    frame = ["{}.{}.jpg".format(animal, i) for i in range(1000, 1500)]
    for f in frame:
        srcf = os.path.join(source_train_path, f)
        tarf = os.path.join(verify_dir, f)
        shutil.copyfile(srcf, tarf)
    # test
    frame = ["{}.{}.jpg".format(animal, i) for i in range(1500, 2000)]
    for f in frame:
        srcf = os.path.join(source_train_path, f)
        tarf = os.path.join(test_dir, f)
        shutil.copyfile(srcf, tarf)
