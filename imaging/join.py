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
        elif self.fill_page_height and len(self.current_page) > 1:
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
    parser.add_argument('-s', '--spacing', type=Distance, dest='spacing', default=Distance('8mm'), help='set default spacing between systems')
    parser.add_argument('-S', '--max-spacing', type=Distance, dest='max_spacing', help='set maximum spacing between systems')
    parser.add_argument('-d', '--dpi', type=float, dest='input_dpi', help='set DPI of input image')
    parser.add_argument('-W', '--width', type=Distance, dest='page_width', help='set output page width')
    parser.add_argument('-H', '--height', type=Distance, dest='page_height', help='set output page height')
    parser.add_argument('-mt', '--margin-top', type=Distance, dest='margin_top', default=Distance('8mm'), help='set page top margin')
    parser.add_argument('-mb', '--margin-bottom', type=Distance, dest='margin_bottom', default=Distance('15mm'), help='set page bottom margin')
    args = parser.parse_args()

    dpi = args.input_dpi

    def get_logical(value):
        return 0 if value is None else value.get_logical(dpi=args.input_dpi)

    joiner = Joiner(
            page_width          = get_logical(args.page_width),
            page_height         = get_logical(args.page_height),
            margin_top          = get_logical(args.margin_top),
            margin_bottom       = get_logical(args.margin_bottom),
            spacing             = get_logical(args.spacing),
            max_spacing         = get_logical(args.max_spacing),
            fill_page_height    = args.fill_page_height,
            )

    page_number = 1
    # when overriding the top margin for the first page:
    # joiner.set(margin_top = 0)

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
