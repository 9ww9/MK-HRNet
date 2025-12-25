import numpy as np
from numpy import ndarray
import PIL
from PIL import ImageDraw, ImageFont
from PIL.Image import Image


point_name = ["1", "2", "3","4",
              "5", "6","7", "8",
              "9", "10","11", "12",
              "13","14","15", "16",
              "17", "18","19","20",
              "21", "22", "23","24"]

point_color = [(255,0,0),(255,128,0),(255, 255, 51),(240, 2, 127),
               (128, 255, 128), (44, 127, 184),
               (0, 255, 0), (0, 0, 255),
               (0, 128, 64), (128, 0, 128),(255,0,0),(255,128,0),(255, 255, 51),(240, 2, 127),
               (128, 255, 128), (44, 127, 184),
               (0, 255, 0), (0, 0, 255),
               (0, 128, 64), (128, 0, 128),(255,0,0),(255,128,0),(255, 255, 51),(240, 2, 127),]


def draw_keypoints(img: Image,
                   keypoints: ndarray,
                   scores: ndarray = None,
                   thresh: float = 0.5,
                   r: int = 2,
                   draw_text: bool = False,
                   font: str = 'arial.ttf',
                   font_size: int = 10,
                   anti_info= None):
    if isinstance(img, ndarray):
        img = PIL.Image.fromarray(img)

    if scores is None:
        scores = np.ones(keypoints.shape[0])

    if draw_text:
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            font = ImageFont.load_default()
    skeleton = anti_info['skeleton']
    coord = np.zeros([24, 3])
    lengths = np.ones(len(skeleton)) * -1
    ratio = 262.88
    draw = ImageDraw.Draw(img)
    for i, (point, score) in enumerate(zip(keypoints, scores)):
        if score > thresh and np.max(point) > 0:
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=point_color[i],
                         outline=(255, 255, 255))
            if draw_text:
                draw.text((point[0] + r, point[1] + r), text=point_name[i], font=font)

            coord[i][0] = 1
            coord[i][1] = point[0]
            coord[i][2] = point[1]

    for i in range(len(skeleton)):
        j = skeleton[i][0]
        k = skeleton[i][1]
        if int(coord[j][0]) == 1 and int(coord[k][0]) == 1:
            s = [coord[j][1], coord[j][2], coord[k][1], coord[k][2]]
            draw.line(s, fill='red', width=10)

            vec1 = np.array([coord[j][1], coord[j][2]])
            vec2 = np.array([coord[k][1], coord[k][2]])
            distance = np.linalg.norm(vec1 - vec2) / ratio
            lengths[i] = distance

    return img,lengths
