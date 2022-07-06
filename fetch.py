from xml.dom import minidom
import os.path
import urllib.request
import zipfile
import OpenEXR
import numpy as np
import cv2
from PIL import Image, ImageFile


OUTPUT_PATH = 'output'
OUTPUT_IMAGES = 'earth_{}_{}.png'
OUTPUT_TERRAIN = 'terrain.png'
OUTPUT_TERRAIN_FP = 'terrain.exr'
OUTPUT_CLOUDS = 'clouds.png'
OUTPUT_NIGHT = 'night.png'

CACHE_PATH = 'images'
IMG_NAMES = 'img_{}_{{}}{{}}.png'
BATH_NAMES = 'bath_{}{}.tif'
TOPO_NAMES = 'topo_{}{}.tif'
CLOUD_NAMES = 'clouds_{}.png'
NIGHT_PREFIX = 'night'
NIGHT_LAYER = 5  # ToDo: Now it works only for 5


#######################################################################
def download(from_url, to_file):
    global CACHE_PATH
    print(f"Downloading {to_file}...")
    to_file = os.path.join(CACHE_PATH, to_file)
    if not os.path.exists(to_file):
        urllib.request.urlretrieve(from_url, to_file)
    else:
        print("\tAlready cached.")
    return to_file


#######################################################################
# 1) Download Bathymetry & Topography images
def download_collection(base_name, new_name, num_x, num_y):
    tiles = []
    for y in range(1, num_y + 1):
        tiles.append([])
        for x in range(0, num_x):
            xn = chr(ord('A') + x)
            tiles[-1].append(download(base_name.format(xn, y), new_name.format(xn, y)))
    return tiles


bathymetry = download_collection('https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73963/gebco_08_rev_bath_{}{}_grey_geo.tif', BATH_NAMES, 4, 2)
topography = download_collection('https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_{}{}_grey_geo.tif', TOPO_NAMES, 4, 2)


#######################################################################
# 2) Download cloud images
def download_ew_collection(base_name, new_name):
    tiles = []
    for letter in ('E', 'W'):
        tiles.append(download(base_name.format(letter), new_name.format(letter)))
    return [tiles]


clouds = download_ew_collection('https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/cloud.{}.2001210.21600x21600.png', CLOUD_NAMES)


#######################################################################
# 3) Download colored images
bmng_urls_shaded = [
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73580/world.topo.bathy.200401.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73605/world.topo.bathy.200402.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73630/world.topo.bathy.200403.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73655/world.topo.bathy.200404.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73701/world.topo.bathy.200405.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73726/world.topo.bathy.200406.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73751/world.topo.bathy.200407.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73776/world.topo.bathy.200408.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73801/world.topo.bathy.200409.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73826/world.topo.bathy.200410.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73884/world.topo.bathy.200411.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x21600x21600.{0}{1}.png",
]
bmng_urls_flat = [
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73938/world.200401.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73967/world.200402.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73992/world.200403.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74017/world.200404.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74042/world.200405.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/76000/76487/world.200406.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74092/world.200407.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74117/world.200408.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74142/world.200409.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74167/world.200410.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74192/world.200411.3x21600x21600.{0}{1}.png",
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/74000/74218/world.200412.3x21600x21600.{0}{1}.png"
]

images = []
for urls, name in zip((bmng_urls_shaded, bmng_urls_flat), ("shaded", "flat")):
    for month_num, month_url in enumerate(urls):
        images.append(download_collection(month_url, IMG_NAMES.format(str(month_num + 1).zfill(2)), 4, 2))


#######################################################################
# 4) Download night cities images
def download_kml(kml_url, cache_prefix, layer):
    kml_file = cache_prefix + '.kml'
    kml_file = download(kml_url, kml_file)
    doc = minidom.parse(kml_file)
    tiles = {}
    for parent_node in doc.getElementsByTagName('NetworkLink'):
        url = parent_node.getElementsByTagName('href')[0].childNodes[0].data
        if not url.endswith('.kmz'):
            continue

        parts = url.rsplit('/', maxsplit=1)
        new_path = download(url, cache_prefix + '_' + parts[1])
        with zipfile.ZipFile(new_path, 'r') as zipf:
            doc_text = zipf.read(parts[1].replace('.kmz', '.kml')).decode('utf-8')
            doc2 = minidom.parseString(doc_text)
            for node in doc2.getElementsByTagName('GroundOverlay'):
                if int(node.getElementsByTagName('drawOrder')[0].childNodes[0].data) != layer:
                    continue

                url = parts[0] + '/' + node.getElementsByTagName('href')[0].childNodes[0].data
                file = cache_prefix + '_' + node.getElementsByTagName('name')[0].childNodes[0].data
                north = float(node.getElementsByTagName('north')[0].childNodes[0].data)
                west = float(node.getElementsByTagName('west')[0].childNodes[0].data)
                east = float(node.getElementsByTagName('east')[0].childNodes[0].data)
                file = download(url, file)
                if north in tiles:
                    tiles[north][west] = file, east
                else:
                    tiles[north] = {}

    tiles_list = []
    for y in sorted(tiles.keys(), reverse=True):
        row = tiles[y]
        tiles_list.append([])
        for x in sorted(row.keys()):
            tiles_list[-1].append(row[x][0])
    return tiles_list


night = download_kml('https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79803/black_marble.kml', NIGHT_PREFIX, NIGHT_LAYER)


#######################################################################
# 5) Merge color images
def load_image(img):
    print(f'\tLoading {img}...')
    i = np.array(Image.open(img))
    if len(i.shape) > 2 and i.shape[2] > 3:
        i = i[:, :, :3]
    return i


def kml_vstack_with_pole_correction(imgs):
    # Resize all to the same size
    total_w = imgs[1].shape[1]

    # Move first and last row one cell to the right
    pad_w = total_w // 31
    reduced_w = total_w - pad_w
    for i in (0, -1):
        padding = np.zeros((imgs[i].shape[0], pad_w, 3)).astype(np.uint8)
        imgs[i] = cv2.resize(imgs[i], (reduced_w, imgs[i].shape[0]), interpolation=cv2.INTER_LANCZOS4)
        imgs[i] = np.hstack((padding, imgs[i]))

    return np.vstack(imgs)


def merge_images(tiles, hstack=np.hstack, vstack=np.vstack):
    return vstack([hstack([load_image(col) for col in row]) for row in tiles])


def save_image(img, file):
    print(f'\tSaving to {file}...')
    # Image.fromarray(img).save(file, compress_level=9)  # Slow
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(file, img, [cv2.IMWRITE_PNG_COMPRESSION, 5])


def merge_images_and_save(tiles, output, hstack=np.hstack, vstack=np.vstack):
    print(f'Merging tiles to {output}...')
    if os.path.exists(output):
        print('\tTarget file already exist, skipping...')
        return
    merged = merge_images(tiles, hstack, vstack)
    save_image(merged, output)


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Hacky: should use projection by coordinates instead of this simple shift
merge_images_and_save(night, os.path.join(OUTPUT_PATH, OUTPUT_NIGHT), vstack=kml_vstack_with_pole_correction)
merge_images_and_save(clouds, os.path.join(OUTPUT_PATH, OUTPUT_CLOUDS))
for month, earth in enumerate(images):
    merge_images_and_save(earth, os.path.join(OUTPUT_PATH, OUTPUT_IMAGES.format('flat' if month >= 12 else 'shaded', (month % 12) + 1)))


print('Merging topography & bathymetry...')
terrain_path = os.path.join(OUTPUT_PATH, OUTPUT_TERRAIN)
terrain_path_fp = os.path.join(OUTPUT_PATH, OUTPUT_TERRAIN_FP)
if not os.path.exists(terrain_path) or not os.path.exists(terrain_path_fp):
    topo_img = merge_images(topography)
    bath_img = merge_images(bathymetry)
    none_img = np.zeros(bath_img.shape).astype(np.uint8)
    if not os.path.exists(terrain_path):
        save_image(np.dstack([none_img, topo_img, 255-bath_img]), terrain_path)

    if not os.path.exists(terrain_path_fp):
        print('\tConverting to floating point...')
        fp_topo = (6400.0 * topo_img.astype(float) / 255.0) + (8000.0 * (bath_img.astype(float) - 255.0) / 255.0)
        print(f'\tSaving to {OUTPUT_TERRAIN_FP}...')
        exr = OpenEXR.OutputFile(os.path.join(OUTPUT_PATH, OUTPUT_TERRAIN_FP), OpenEXR.Header(fp_topo.shape[1], fp_topo.shape[0]))
        exr.writePixels({'A': fp_topo})
else:
    print('\tTarget file already exist, skipping...')

