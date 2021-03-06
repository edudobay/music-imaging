import re

CM_TO_INCH = 1 / 2.54

def cm2points(cm, dpi=72):
    return cm * (CM_TO_INCH * dpi)

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
            m = re.match(Distance.RE_VALUE, value)
            if m is None:
                raise ValueError("unrecognized value")
            else:
                self._set(m.group(1), 'px')
        else:
            self._set(m.group(1), m.group(2))

    def _set(self, value, unit):
        self.value = float(value)
        self.unit = Distance.UNITS[unit]
        self.unit_name = unit

    def __repr__(self):
        return '<Distance: {value} {unit}>'.format(value=self.value, unit=self.unit_name)

    def get_logical(self, dpi=None):
        unit_value, unit_type = self.unit
        if dpi is None and unit_type != 'logical':
            raise ValueError("dpi not provided for non-logical unit")
        if unit_type == 'physical':
            unit_value *= dpi
        return self.value * unit_value

