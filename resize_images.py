from PIL import Image
import os

def resize_aspect_fit(input_dir, output_dir, resize_ratio=0.5):
    dirs = os.listdir(input_dir)
    for item in dirs:
        if item == '.DS_Store':
            continue
        if os.path.isfile(input_dir+"/"+item):
            print("Resizing " + item)
            image = Image.open(input_dir+"/"+item)
            file_path, extension = os.path.splitext(input_dir+"/"+item)

            new_image_height = int(image.size[0] / (1/resize_ratio))
            new_image_length = int(image.size[1] / (1/resize_ratio))

            image = image.resize((new_image_height, new_image_length), Image.ANTIALIAS)
            image.save(output_dir + "/" + item, 'JPEG', quality=90)
            # image.save(file_path + "_small" + extension, 'JPEG', quality=90)
