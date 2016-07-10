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
