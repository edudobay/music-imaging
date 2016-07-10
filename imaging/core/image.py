from PIL import Image

def get_image_channels(image: Image.Image):
    from itertools import count
    return dict(zip(image.getbands(), count()))

def get_channel_data(image: Image.Image, *channel_names: str, as_dict: bool = False):
    from numpy import array
    width, height = image.size
    channels = get_image_channels(image)
    channel_data = {}
    for channel_name in channel_names:
        channel_index = channels[channel_name]
        channel_data[channel_name] = array(image.getdata(channel_index)).reshape(height, width)
    if as_dict or len(channel_names) != 1:
        return dict(zip(channel_names, (channel_data[channel_name] for channel_name in channel_names)))
    else:
        return channel_data[channel_names[0]] 

def alpha_blend_over_background(image, background_color='white'):
    background = Image.new('RGBA', image.size, background_color)
    return Image.alpha_composite(background, image.convert('RGBA'))

def offset_image(image, offset, *, image_mode=None, background_color='white'):
    """
    Return a copy of the given image offset by the given displacement.

    offset -- a (x, y) pair. These can be floating and will be converted to
    integers (using int())

    image_mode -- mode to use for the new image. See `PIL.Image.new`.  If not
    given, the same mode of the source image is assumed.

    background_color -- background color to fill the unused pixels. See
    `PIL.Image.new`.
    """
    x, y = offset
    offset = int(x), int(y)

    if image_mode is None:
        image_mode = image.mode

    board = Image.new(image_mode, image.size, background_color)
    board.paste(image, offset)
    return board
