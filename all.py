import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import colorsys

class PixelConverter:
    def __init__(self, pixelspace: int, rgb_to_custom, custom_to_rgb):
        self.pixelspace = pixelspace
        self.__to_converter = np.vectorize(rgb_to_custom)
        self.__from_converter = np.vectorize(custom_to_rgb)

    def convert_to(self, channels):
        return self.__to_converter(*channels)

    def convert_from(self, channels):
        return self.__from_converter(*channels)

class ImageColorSeparator:
    def __init__(self, image: str, text: str, converter: PixelConverter):
        self.channels = Image.open(image).split()
        self.channels = [np.array(channel) / 256.0 for channel in self.channels]
        self.text = text
        self.converter = converter

    def show_image(self, ax, image):
        ax.imshow(np.moveaxis(image, 0, -1))

    def process(self, channel_filter):
        ch = self.converter.convert_to(self.channels)

        ones_matrix = np.ones(ch[0].shape, dtype=np.float64)

        ch_separated = [[ch[i] if j == i else ones_matrix * channel_filter[i][j] for j in range(self.converter.pixelspace)] for i in range(self.converter.pixelspace)]

        ch_converted_separated = [np.array(self.converter.convert_from(c), dtype=np.float64) for c in ch_separated]

        fig, axs = plt.subplots(2, 2)  
        (ax1, ax2), (ax3, ax4) = axs 
        ax = [ax1, ax2, ax3]

        for i, ch_conv in enumerate(ch_converted_separated):
            ax[i].axis('off')
            self.show_image(ax[i], ch_conv)
            extent = ax[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig("img"+str(i)+str(self.text)+".png", dpi=1000, transparent=True, bbox_inches=extent)


def draw_rgb():
    rgb_to_rgb = lambda x, y, z: (x, y, z)

    converter = PixelConverter(3, rgb_to_rgb, rgb_to_rgb)
    separator = ImageColorSeparator('Daniil.png', 'RGB', converter)
    separator.process(
        channel_filter = (
            (0, 0, 0),
            (0, 0, 0),
            (0, 0, 0)
        )
    )

def draw_hsv():
    converter = PixelConverter(3, colorsys.rgb_to_hsv, colorsys.hsv_to_rgb)
    separator = ImageColorSeparator('Daniil.png', 'HSV', converter)
    separator.process(
        channel_filter = (
            (1.0, 1.0, 1,0),
            (0.3, 1.0, 0.9),
            (0.3, 0.6, 1.0)
        )
    )

def draw_ycbcr():
    def rgb_to_ycbcr(r, g, b):
        return (
            0 + 0.299 * r + 0.587 * g + 0.114 * b,
            0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b,
            0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
        )

    def ycbcr_to_rgb(y, cb, cr):
        return (
            y + 1.402 * (cr - 0.5),
            y - 0.34414 * (cb - 0.5) - 0.71414 * (cr - 0.5),
            y + 1.1772 * (cb - 0.5)
        )

    converter = PixelConverter(3, rgb_to_ycbcr, ycbcr_to_rgb)
    separator = ImageColorSeparator('Daniil.png', 'YCbCr', converter)
    separator.process(
        channel_filter = (
            (1.0, 0.5, 0.5),
            (0.6, 1.0, 0.5),
            (0.6, 0.5, 1.0)
        )
    )

def draw_grayscale():
    def rgb_to_grayscale(r, g, b):
        return (
            0.333 * r + 0.333 * g + 0.333 * b,
            0
        )

    def grayscale_to_rgb(gr, a):
        return (
            gr,
            gr,
            gr
        )

    converter = PixelConverter(2, rgb_to_grayscale, grayscale_to_rgb)
    separator = ImageColorSeparator('Daniil.png', 'GrayScale', converter)
    separator.process(
        channel_filter = (
            (0.5, 0.5),
            (0.6, 0.5),
        )
    )

# def main():
#     draw_rgb()
#     draw_hsv()
#     draw_ycbcr()
#     draw_grayscale()

# if __name__ == '__main__':
#     main()

draw_rgb()
draw_hsv()
draw_ycbcr()
draw_grayscale()