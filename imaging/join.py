#!/usr/bin/env python3
import os
import sys
from PIL import Image

from imaging.core.units import Distance

class Joiner:
    def __init__(self, *, page_width, page_height, margin_top, margin_bottom, spacing,
            fill_page_height = False, max_spacing = None):
        self.page_width = int(page_width)
        self.page_height = int(page_height)
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.spacing = spacing
        self.fill_page_height = fill_page_height
        self.max_spacing = max_spacing

        self.override_margin_top = None

        self.current_page = []

    def can_add(self, item):
        return self.get_size(self.current_page + [item]) <= self.page_height

    def _margin_top(self):
        return self.margin_top if self.override_margin_top is None \
                else self.override_margin_top

    def get_size(self, items):
        return self._margin_top() + self.margin_bottom + \
                self.spacing * (len(items) - 1) + \
                sum(item.size[1] for item in items)

    def add(self, item):
        self.current_page.append(item)

    def set(self, *, margin_top):
        self.override_margin_top = margin_top

    def clear(self):
        self.override_margin_top = None
        self.current_page.clear()

    def layout_page(self):
        page = Image.new('RGBA', (self.page_width, self.page_height), 'white')
        background = page.copy()

        top = self._margin_top()
        left = 0

        spacing = self.spacing
        spare_height = self.page_height - self.get_size(self.current_page) 
        if spare_height < 0:
            print('warning: page contents exceeds page height by %d points' % (-spare_height), file=sys.stderr)
        elif self.fill_page_height:
            spacing = spacing + spare_height / (len(self.current_page) - 1)
            if self.max_spacing is not None:
                spacing = min(spacing, self.max_spacing)

        for item in self.current_page:
            _, item_height = item.size
            page.paste(item, (left, int(round(top))))
            top += item_height + spacing

        page = Image.alpha_composite(background, page)
        return page

    def is_empty(self):
        return not self.current_page

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('infile', metavar='FILE', nargs='+', help='input image file')
    parser.add_argument('-O', metavar='DIR', dest='output_dir', help='destination directory')
    parser.add_argument('-f', dest='format', default='collage_%02d.png', help='format for naming output files')
    parser.add_argument('-e', dest='fill_page_height', action='store_true', help='fill all available vertical space')
    parser.add_argument('-S', dest='max_spacing', type=Distance, help='maximum spacing between systems')
    parser.add_argument('-d', '--dpi', type=float, dest='input_dpi', help='set DPI of input image')
    parser.add_argument('-W', '--width', type=Distance, dest='page_width', help='set output page width')
    parser.add_argument('-H', '--height', type=Distance, dest='page_height', help='set output page height')
    args = parser.parse_args()

    dpi = args.input_dpi
    MM = Distance(1, 'mm').get_logical(dpi)
    page_width  = args.page_width.get_logical(dpi)
    page_height = args.page_height.get_logical(dpi)
    max_spacing = args.max_spacing.get_logical(dpi)

    joiner = Joiner(
            page_width = page_width, page_height = page_height,
            margin_top = 8*MM, margin_bottom = 15*MM, spacing = 8*MM,
            fill_page_height = args.fill_page_height,
            max_spacing = max_spacing,
            )

    page_number = 1
    joiner.set(margin_top = 0)

    basename = args.format
    dirname = args.output_dir
    outfile = os.path.join(dirname, basename)

    for infile in args.infile:
        with Image.open(infile) as im:
            if not joiner.can_add(im):
                page = joiner.layout_page()
                page.save(outfile % page_number, dpi = (dpi, dpi)) # (DPI_x, DPI_y)
                page_number += 1
                joiner.clear()

            print(page_number, infile)
            joiner.add(im.copy())

    if not joiner.is_empty():
        page = joiner.layout_page()
        page.save(outfile % page_number)

if __name__ == '__main__':
    main()
