#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import operator
import argparse
import datetime
import glob

import fitz
import xmltodict
from PIL import Image
from tqdm import tqdm

if not tuple(map(int, fitz.version[0].split('.'))) >= (1, 18, 18):
    raise SystemExit('require PyMuPDF v1.18.18+')

# This key is the item name when writing to markdown (can be set arbitrarily),
# and value must match the name of the key saved in xml (need to be adjusted according to the prompt to GPT).
XML_DICT_KEYS = {'要旨': 'summary',
                 '入出力': 'input_output',
                 '新規性・手法のキモ': 'method',
                 '検証方法': 'validation',
                 '議論': 'discussion'}


def recover_pix(doc, item):
    xref = item[0]  # xref of PDF image
    s_mask = item[1]  # xref of its /SMask

    # Special case: /SMask or /Mask exists
    if s_mask > 0:
        pix0 = fitz.Pixmap(doc.extract_image(xref)['image'])
        if pix0.alpha:  # Catch irregular situation
            pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
        mask = fitz.Pixmap(doc.extract_image(s_mask)['image'])

        try:
            pix = fitz.Pixmap(pix0, mask)
        except:
            # Fallback to original base image in case of problems
            pix = fitz.Pixmap(doc.extract_image(xref)['image'])

        if pix0.n > 3:
            ext = 'pam'
        else:
            ext = 'png'

        # Create dictionary expected by caller
        return {
            'ext': ext,
            'colorspace': pix.colorspace.n,
            'image': pix.tobytes(ext),
        }

    # Special case: /ColorSpace definition exists
    # To be sure, we convert these cases to RGB PNG images
    if '/ColorSpace' in doc.xref_object(xref, compressed=True):
        pix = fitz.Pixmap(doc, xref)
        pix = fitz.Pixmap(fitz.csRGB, pix)

        # Create dictionary expected by caller
        return {
            'ext': 'png',
            'colorspace': 3,
            'image': pix.tobytes('png'),
        }
    return doc.extract_image(xref)


def extract_images_from_pdf(f_name, imgdir='./output', min_width=400, min_height=400, max_img_file_size=2048,
                            max_ratio=8, max_num=5):
    os.makedirs(imgdir, exist_ok=True)

    doc = fitz.open(f_name)

    xref_list = []
    img_list = []
    images = []
    for pno in range(doc.page_count):
        if len(images) >= max_num:
            break
        il = doc.get_page_images(pno)
        img_list.extend([x[0] for x in il])
        for img in il:
            xref = img[0]
            if xref in xref_list:
                continue
            width = img[2]
            height = img[3]
            if width < min_width and height < min_height:
                continue
            image = recover_pix(doc, img)
            n = image['colorspace']
            imgdata = image['image']

            if len(imgdata) <= max_img_file_size:
                continue

            if width / height > max_ratio or height / width > max_ratio:
                continue

            img_name = 'img%02d_%05i.%s' % (pno + 1, xref, image['ext'])
            images.append((img_name, pno + 1, width, height))
            img_file = os.path.join(imgdir, img_name)
            with open(img_file, 'wb') as f:
                f.write(imgdata)
            xref_list.append(xref)

    img_list = list(set(img_list))
    return xref_list, img_list, images


def get_half(fname):
    # Open the PDF file
    pdf_file = fitz.open(fname)
    # Get the first page
    page = pdf_file[0]
    # Get the page as a whole image
    mat = fitz.Matrix(2, 2)  # zoom factor 2 in each direction
    pix = page.get_pixmap(matrix=mat)
    # Convert to a PIL Image
    im = Image.open(io.BytesIO(pix.tobytes()))
    # Get the dimensions of the image
    width, height = im.size
    # Define the box for the upper half (left, upper, right, lower)
    box = (0, height // 20, width, (height // 2) + (height // 20))

    # Crop the image to this box
    im_cropped = im.crop(box)
    return im_cropped


def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    min_height = min(im.height for im in im_list)
    im_list_resize = [im.resize((int(im.width * min_height / im.height), min_height), resample=resample)
                      for im in im_list]
    total_width = sum(im.width for im in im_list_resize)
    dst = Image.new('RGB', (total_width, min_height))
    pos_x = 0
    for im in im_list_resize:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst


def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
    min_width = min(im.width for im in im_list)
    im_list_resize = [im.resize((min_width, int(im.height * min_width / im.width)), resample=resample)
                      for im in im_list]
    total_height = sum(im.height for im in im_list_resize)
    dst = Image.new('RGB', (min_width, total_height))
    pos_y = 0
    for im in im_list_resize:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst


def get_concat_tile_resize(im_list_2d, resample=Image.BICUBIC):
    im_list_v = [get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
    return get_concat_v_multi_resize(im_list_v, resample=resample)



def get_all_page(fname, max_w=1600, max_h=900):
    pdf_file = fitz.open(fname)
    page_imgs = []
    for page in pdf_file:
        try:
            mat = fitz.Matrix(2, 2)  # zoom factor 2 in each direction
            # Get the page as a whole image
            pix = page.get_pixmap(matrix=mat)
            # Convert to a PIL Image
            im = Image.open(io.BytesIO(pix.tobytes()))
            page_imgs.append(im)
        except Exception as e:
            print(e)
            continue

    if len(page_imgs) == 0:
        return None
    elif len(page_imgs) == 1:
        return page_imgs[0]

    if pdf_file.page_count <= 8:
        n = int(pdf_file.page_count / 2)
        top_imgs = page_imgs[:n]
        bottom_imgs = page_imgs[n:]
        tiled_img = get_concat_tile_resize([top_imgs, bottom_imgs])
    else:
        n = int(pdf_file.page_count / 3)
        top_imgs = page_imgs[:n]
        middle_imgs = page_imgs[n:2 * n]
        bottom_imgs = page_imgs[2 * n:]
        tiled_img = get_concat_tile_resize([top_imgs, middle_imgs, bottom_imgs])

    w, h = tiled_img.size
    if w < h:
        tiled_img = tiled_img.resize((max_w, int(max_w * h / w)))
    else:
        tiled_img = tiled_img.resize((int(max_h * w / h), max_h))

    return tiled_img


def make_md(f, dirname, filename, keywords=[]):
    path = f'{dirname}/{filename}'
    with open(path, 'r') as fin:
        xml = fin.read()
        xml_lower = xml.lower()
        if (keywords is not None) and not (any([k.lower() in xml_lower for k in keywords])):
            return
    paper_info = xmltodict.parse(xml)['paper']

    # Get half top image of pdf
    pdf_name = f'{dirname}/paper.pdf'
    img_cropped = get_half(pdf_name)
    img_cropped.save(f'{dirname}/half.png', 'PNG')

    f.write('\n---\n')
    f.write('<!-- _class: info -->\n')
    f.write(f'![width:1400]({dirname}/half.png)\n')

    # Get tiled image of pdf
    tiled_pages = get_all_page(pdf_name)
    if tiled_pages is not None:
        tiled_pages.save(f'{dirname}/tiled_pages.jpg', 'JPEG')

        f.write('\n---\n')
        f.write('<!-- _class: info -->\n')
        f.write(f'![height:650]({dirname}/tiled_pages.jpg)\n')

    # Get all images from the pdf
    _, _, image_list = extract_images_from_pdf(pdf_name, imgdir=dirname)
    images = [{'src': imgname, 'pno': pno, 'width': width, 'height': height} for
              imgname, pno, width, height in image_list]

    # Get 1st & 2nd images in the pdf
    images = sorted(images, key=operator.itemgetter('pno'))[:2]

    # Write the images
    for img in images:
        src = img['src']
        width = int(img['width'])
        height = int(img['height'])
        x_ratio = (1600.0 * 0.7) / (float)(width)
        y_ratio = (900.0 * 0.7) / (float)(height)
        ratio = min(x_ratio, y_ratio)

        f.write('\n---\n')
        f.write('<!-- _class: info -->\n')
        f.write(f'![width:{(int)(ratio * width)}]({dirname}/{src})\n')

    # Write summaries
    f.write('\n---\n')
    f.write('<!-- _class: title -->\n')
    f.write(f"## {paper_info['title']}\n")

    f.write(f"[{paper_info['date']}] {paper_info['entry_id']}\n")
    try:
        f.write(f"__Keywords__ {paper_info['keywords']}\n")
    except:
        pass

    for i, (k, v) in enumerate(XML_DICT_KEYS.items()):
        try:
            # New page when "i" is odd to make it easier to see as slides
            if i % 2 == 1:
                f.write('\n---\n')

            f.write(f"\n__{k}__\n{paper_info[v]}\n")
        except KeyError as e:
            print(e)
            f.write(f"__{v}__\nNone\n")


def get_files(xml_dir):
    xml_files = glob.glob(f'{xml_dir}/*/*.xml')

    # Ascending order by publish date
    xml_files.sort()
    return xml_files


def main(input_dir='./xmrs', output='./out.md', keywords=[]):
    xml_files = get_files(xml_dir=input_dir)
    dt_now = datetime.datetime.now()
    date = dt_now.strftime('%Y年%m月%d日')
    with open(output, 'w') as f:
        f.write('---\n')
        f.write('marp: true\n')
        f.write('theme: default\n')
        f.write('size: 16:9\n')
        f.write('paginate: true\n')
        f.write('_class: ["cool-theme"]\n')
        f.write('\n---\n')
        f.write(f'# {keywords} on arXiv\n')
        f.write('Generated by GPT-4\n')
        f.write(f'{date}\n')
        f.write('\n')

        for file in tqdm(xml_files):
            dirname, filename = os.path.split(file)
            tqdm.write(file)
            make_md(f, dirname, filename, keywords=keywords)
    print('Output file: ', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, help='xml dir', default='./xmls')
    parser.add_argument('--output', '-o', type=str, default='output.md', help='output markdown file')
    parser.add_argument('positional_args', nargs='?', help='query keywords')
    args = parser.parse_args()

    keywords = args.positional_args
    if type(keywords) == str:
        keywords = [keywords]

    print(args, keywords)

    main(input_dir=args.dir, output=args.output, keywords=keywords)
