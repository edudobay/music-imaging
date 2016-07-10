#!/usr/bin/env python3
import os
import sys
import re
from argparse import *
from PIL import Image

class Distance:
    RE_VALUE = r'(-?(?:\d+(?:\.\d+)?|\.\d+))'
    RE_UNIT = r'(cm|mm|in|px|pt)'
    RE_ALL = RE_VALUE + RE_UNIT

    UNITS = {
        'in': (1, 'physical'),
        'cm': (1/2.54, 'physical'),
        'mm': (1/25.4, 'physical'),
        'pt': (1/72, 'physical'),
        'px': (1, 'logical'),
    }

    def __init__(self, value, unit=None):
        if unit is None:
            self._parse(value)
        else:
            self._set(value, unit)

    def _parse(self, value):
        m = re.match(Distance.RE_ALL, value)
        if m is None:
            raise ValueError("unrecognized value")

        self._set(m.group(1), m.group(2))

    def _set(self, value, unit):
        self.value = float(value)
        self.unit = Distance.UNITS[unit]

    def get_logical(self, dpi=None):
        unit_inches, unit_type = self.unit
        if dpi is None and unit_type != 'logical':
            raise ValueError("dpi not provided for non-logical unit")
        return self.value * unit_inches * dpi

class Joiner:
    def __init__(self, *, page_width, page_height, margin_top, margin_bottom, spacing,
            fill_page_height = False, max_spacing = None):
        self.page_width = page_width
        self.page_height = page_height
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
            page.paste(item, (left, round(top)))
            top += item_height + spacing

        page = Image.alpha_composite(background, page)
        return page

    def is_empty(self):
        return not self.current_page

def main():
    parser = ArgumentParser()
    parser.add_argument('infile', metavar='FILE', nargs='+', help='input image file')
    parser.add_argument('-O', metavar='DIR', dest='output_dir', help='destination directory')
    parser.add_argument('-f', dest='format', default='collage_%02d.png', help='format for naming output files')
    parser.add_argument('-e', dest='fill_page_height', action='store_true', help='fill all available vertical space')
    parser.add_argument('-S', dest='max_spacing', type=Distance, help='maximum spacing between systems')
    args = parser.parse_args()

    DPI = 300
    MM = DPI / 25.4
    page_width  = round(210*MM)
    page_height = round(297*MM)

    joiner = Joiner(
            page_width = page_width, page_height = page_height,
            margin_top = 8*MM, margin_bottom = 15*MM, spacing = 8*MM,
            fill_page_height = args.fill_page_height,
            max_spacing = args.max_spacing.get_logical(dpi = DPI),
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
                page.save(outfile % page_number, dpi = (DPI, DPI)) # (DPI_x, DPI_y)
                page_number += 1
                joiner.clear()

            print(page_number, infile)
            joiner.add(im.copy())

    if not joiner.is_empty():
        page = joiner.layout_page()
        page.save(outfile % page_number)

if __name__ == '__main__':
    main()
