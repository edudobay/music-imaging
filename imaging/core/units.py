CM_TO_INCH = 1 / 2.54

def cm2points(cm, dpi=72):
    return cm * (CM_TO_INCH * dpi)
