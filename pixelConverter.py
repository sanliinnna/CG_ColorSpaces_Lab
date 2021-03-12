from imports import np

class PixelConverter:
    def __init__(self, pixelspace: int, rgb_to_custom, custom_to_rgb):
        self.pixelspace = pixelspace
        self.__to_converter = np.vectorize(rgb_to_custom)
        self.__from_converter = np.vectorize(custom_to_rgb)

    def convert_to(self, channels):
        return self.__to_converter(*channels)

    def convert_from(self, channels):
        return self.__from_converter(*channels)