import glob

from PIL import Image
import torchvision.transforms as T


def main():
    transforms = T.Compose([
        T.RandomVerticalFlip(),
        T.RandomHorizontalFlip(),
    ])

    directory = glob.glob('../Calibration_frames/*')
    for img_file_name in directory:
        img = Image.open(img_file_name)
        img = transforms(img)

        save_extension = img_file_name[-4:]
        img_file_name = img_file_name[:-4]
        img_file_name += 'flip' + save_extension

        img.save(img_file_name)


if __name__ == '__main__':
    main()
