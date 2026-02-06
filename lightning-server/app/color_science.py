"""Color space conversions and WID calculation.

CRITICAL: These algorithms MUST match the Next.js implementation exactly.
Any changes here must be synchronized with lib/image-utils.ts and lib/wid-calculator.ts
"""

# D65 standard illuminant white point
D65 = {"x": 95.047, "y": 100.0, "z": 108.883}


def srgb_to_linear(v: int) -> float:
    """Convert sRGB value [0-255] to linear RGB [0-1].

    Args:
        v: sRGB value (0-255)

    Returns:
        Linear RGB value (0-1)
    """
    v_norm = v / 255.0
    if v_norm <= 0.04045:
        return v_norm / 12.92
    return ((v_norm + 0.055) / 1.055) ** 2.4


def linear_rgb_to_xyz(r: float, g: float, b: float) -> dict:
    """Convert linear RGB to CIE XYZ color space.

    Uses sRGB to XYZ transformation matrix (D65 illuminant).

    Args:
        r, g, b: Linear RGB values (0-1)

    Returns:
        dict with x, y, z components (0-100 scale)
    """
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.072175
    z = r * 0.0193339 + g * 0.119192  + b * 0.9503041
    return {"x": x * 100, "y": y * 100, "z": z * 100}


def xyz_to_lab(x: float, y: float, z: float) -> dict:
    """Convert CIE XYZ to CIELAB color space.

    Uses D65 white point as reference.

    Args:
        x, y, z: XYZ values (0-100 scale)

    Returns:
        dict with l, a, b components
    """
    def f(t):
        if t > 0.008856:
            return t ** (1/3)
        return 7.787 * t + 16/116

    fx = f(x / D65["x"])
    fy = f(y / D65["y"])
    fz = f(z / D65["z"])

    return {
        "l": 116 * fy - 16,
        "a": 500 * (fx - fy),
        "b": 200 * (fy - fz)
    }


def rgb_to_lab(r: int, g: int, b: int) -> dict:
    """Convert sRGB to CIELAB color space.

    Complete pipeline: sRGB -> linear RGB -> XYZ -> Lab

    Args:
        r, g, b: sRGB values (0-255)

    Returns:
        dict with l, a, b components
    """
    rl = srgb_to_linear(r)
    gl = srgb_to_linear(g)
    bl = srgb_to_linear(b)

    xyz = linear_rgb_to_xyz(rl, gl, bl)
    return xyz_to_lab(xyz["x"], xyz["y"], xyz["z"])


def compute_wid(l: float, a: float, b: float) -> float:
    """Calculate Whiteness Index for Dentistry (WID).

    Formula: WID = 0.511*L* + (-2.324)*a* + (-1.100)*b*
    Higher WID = whiter teeth

    Args:
        l, a, b: CIELAB color values

    Returns:
        WID value (typically -20 to 60 range for natural teeth)
    """
    return 0.511 * l + (-2.324) * a + (-1.100) * b
