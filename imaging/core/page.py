def center_on_page(object_size, page_size):
    """
    Compute the (x, y) position where an object's top-left corner must be
    placed so that it is centered on a page.
    """
    page_width, page_height = page_size
    object_width, object_height = object_size

    object_x = (page_width - object_width) / 2
    object_y = (page_height - object_height) / 2
    return object_x, object_y

