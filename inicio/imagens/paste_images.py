from PIL import Image


def paste(images_paths, name_fig='auto'):
    if name_fig == 'auto':
        from datetime import datetime
        name_fig = datetime.now().strftime('%Y%m%d_%H%M%S')
    imgs = [Image.open(img) for img in images_paths]
    total_width = sum([x.size[0] for x in imgs])
    height = imgs[0].size[1]
    blank_img = Image.new("RGB", (total_width, height))
    pre_width = 0
    for img in imgs:
        blank_img.paste(img, (pre_width, 0))
        pre_width += img.size[0]
    if name_fig:
        blank_img.save(name_fig, format='png')
    return blank_img
