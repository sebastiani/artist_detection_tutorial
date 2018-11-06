import os
import sys
sys.path.append('../../artist_detection_tutorial/')   # hack to add upper visibility to this file
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as transforms
import pickle


SEED = 23
RESIZE = False

IMG_WIDTH = 256
IMG_HEIGHT = 256
PIXEL_DEPTH = 255.0
ARTISTS = 11

CHANNELS = 3
DATA_PATH = '/data/raw/wikiart/images/'
PREPROCESSED_PATH = '../data/intermediate/wikiart/'
NUM_PER_CLASS = 50

TRAIN_TEST_SPLIT = 0.1
TEST_VAL_SPLIT = 0.5

MAKE_TEST = True
MAKE_DATASET = False

def loadArtist(path):

    images = []

    image_files = os.listdir(path)
    for image_file in image_files:
        file_path = os.path.join(path, image_file)
        with Image.open(file_path) as img:
            im = np.array(img)
            if len(im.shape) != 3:
                continue
            else:
                images.append(np.expand_dims(im, axis=0))

    images = np.vstack(images)
    mean = np.mean(images)
    std = np.std(images)
    print("Artist dataset shape: ", images.shape)
    print("Artist dataset mean: ", mean)
    print("Artist dataset std: ", std)
    return images, mean, std

def resizeImages(datasetPath):
    artists = os.listdir(datasetPath)
    for artist in artists:
        years = os.listdir(os.path.join(datasetPath, artist))
        print("Resizing Images for %s" %(artist))
        for year in years:
            images = os.listdir(os.path.join(datasetPath, artist, year))
            if not os.path.exists(os.path.join(PREPROCESSED_PATH, artist)):
                os.makedirs(os.path.join(PREPROCESSED_PATH, artist))
            for image in images:
                image_path = os.path.join(datasetPath, artist, year, image)
                with Image.open(image_path) as img:
                    resizedImage = transforms.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    filename = os.path.join(PREPROCESSED_PATH, artist, image)
                    try:
                        resizedImage.save(filename)
                    except OSError:
                        os.remove(image_path)
                        print('dropped: ', image_path)
    print('Done!')

def scale_pixels(img):
    return (img - PIXEL_DEPTH/2)/PIXEL_DEPTH

def randomize(dataset, labels, axis=0):
    indexes = np.random.permutation(dataset.shape[axis])
    randomizedDataset = dataset[indexes, :, :, :]
    randomizedLabels = labels[indexes, :]
    return randomizedDataset, randomizedLabels

def makeAgumentedTest(test, test_labels):
    num_image = test.shape[0]
    translatedImages = []
    brightenedImages = []
    darkenedImages = []
    hContrasted = []
    lContrasted = []
    flippedImages = []
    invertedImages = []

    for i in range(num_image):
        img  = test[i, :, :, :]

        img = Image.fromarray(img)
        # Do Translation
        translatedIMG = transforms.affine(img, angle=0, translate=(50, 50), scale=1, shear=0)
        # Do Brigthened
        brightenedIMG = transforms.adjust_brightness(img, 1.5)
        # Do Darkened
        darkenedIMG = transforms.adjust_brightness(img, 0.75)
        # Do High Contrast
        highContrastIMG = transforms.adjust_contrast(img, 1.5)
        # Do Low Contrast
        lowContrastIMG  = transforms.adjust_contrast(img, 0.75)
        # Flipped Upside Down
        flippedIMG = transforms.hflip(img)
        # Colors Inverted
        invertedIMG = ImageOps.invert(img)

        translatedImages.append(np.expand_dims(np.array(translatedIMG), axis=0))
        brightenedImages.append(np.expand_dims(np.array(brightenedIMG), axis=0))
        darkenedImages.append(np.expand_dims(np.array(darkenedIMG), axis=0))
        hContrasted.append(np.expand_dims(np.array(highContrastIMG), axis=0))
        lContrasted.append(np.expand_dims(np.array(lowContrastIMG), axis=0))
        flippedImages.append(np.expand_dims(np.array(flippedIMG), axis=0))
        invertedImages.append(np.expand_dims(np.array(invertedIMG), axis=0))


    translatedImages = np.vstack(translatedImages)
    brightenedImages = np.vstack(brightenedImages)
    darkenedImages = np.vstack(darkenedImages)
    hContrasted = np.vstack(hContrasted)
    lContrasted = np.vstack(lContrasted)
    flippedImages = np.vstack(flippedImages)
    invertedImages = np.vstack(invertedImages)


    augmentedTest = {
        'translated': translatedImages,
        'brightened': brightenedImages,
        'darkened': darkenedImages,
        'high_contrast': hContrasted,
        'low_contrast': lContrasted,
        'flipped': flippedImages,
        'inverted': invertedImages,
        'labels': test_labels
    }

    with open('augmented_test.pkl', 'wb') as outfile:
        pickle.dump(augmentedTest, outfile)

    print('Done!')

def makeAgumentedSet(train, test, train_labels, test_labels):
    pass

def make_dataset():
    if RESIZE:
        print("Resize images")
        resizeImages(DATA_PATH)

    artists =  os.listdir(PREPROCESSED_PATH)
    labelDictionary = {}
    train_labels = []
    test_labels = []
    val_labels = []

    train_dataset = []
    test_dataset = []
    val_dataset = []
    for label, artist in enumerate(artists):
        print("processing: ", artist)
        labelDictionary[label-1] = artist
        artist_path = os.path.join(PREPROCESSED_PATH, artist)
        images, _, _ = loadArtist(artist_path)
        image_num = images.shape[0]

        labelVectors = np.zeros((image_num, ARTISTS))
        labelVectors[:, label-1] = 1

        artist_train, artist_test, train_label, test_label = train_test_split(images, labelVectors, test_size=TRAIN_TEST_SPLIT, random_state=SEED)
        artist_val, artist_test, val_label, test_label = train_test_split(artist_test, test_label, test_size=TEST_VAL_SPLIT, random_state=SEED)

        artist_train, train_label = randomize(artist_train, train_label)
        if artist_train.shape[0] > 50:
            print("%i found Capping to 50" % (artist_train.shape[0]))
            image_num = 50
        train_dataset.append(artist_train[:image_num, :, :, :])
        test_dataset.append(artist_test)
        val_dataset.append(artist_val)

        train_labels.append(train_label[:image_num, :])
        test_labels.append(test_label)
        val_labels.append(val_label)

    train_dataset = np.vstack(train_dataset)
    test_dataset = np.vstack(test_dataset)
    val_dataset = np.vstack(val_dataset)

    train_labels = np.vstack(train_labels)
    test_labels = np.vstack(test_labels)
    val_labels = np.vstack(val_labels)

    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    val_dataset, val_labels = randomize(val_dataset, val_labels)


    with open('art_dataset.pkl', 'wb')  as outfile:
        dataset = {'train': scale_pixels(train_dataset), 'test': scale_pixels(test_dataset), 'validation': scale_pixels(val_dataset),
                  'train_labels': train_labels, 'test_labels': test_labels, 'validation_labels':val_labels, 'label_names': labelDictionary}
        pickle.dump(dataset, outfile)

    print("making augmented test")
    makeAgumentedTest(test_dataset, test_labels)
    print('DONE!')


if __name__ == '__main__':

    make_dataset()