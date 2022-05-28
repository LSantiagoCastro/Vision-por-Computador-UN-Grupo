"""Script to take read image from disk and display it 

Usage:
    2_readImage.py <path>
"""
from docopt import docopt

import funciones 

import cv2

def main(path_image):
    image =  cv2.imread(path_image)
    im = funciones.detect_color(path_image)
    
    conteo = im.Conteo_total()

    
    conteo
    cv2.waitKey(0)


if __name__ == "__main__":
    args = docopt(__doc__)
    path_image = args['<path>']
    main(path_image)