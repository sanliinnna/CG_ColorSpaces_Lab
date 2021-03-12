from imports import PixelConverter, plt, Image, np

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