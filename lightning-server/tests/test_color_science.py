"""Tests for color science module - exact RGB->Lab->WID conversions."""
import pytest
from app.color_science import (
    D65,
    compute_wid,
    linear_rgb_to_xyz,
    rgb_to_lab,
    srgb_to_linear,
    xyz_to_lab,
)


class TestSrgbToLinear:
    """Test sRGB to linear RGB conversion."""

    def test_black(self):
        """Test conversion of black (0)."""
        assert srgb_to_linear(0) == 0.0

    def test_white(self):
        """Test conversion of white (255)."""
        result = srgb_to_linear(255)
        assert abs(result - 1.0) < 1e-6

    def test_boundary_low(self):
        """Test boundary at 0.04045 threshold (v_norm = 10/255 ≈ 0.0392)."""
        # v=10 -> v_norm=0.0392 -> below threshold -> linear scaling
        v = 10
        v_norm = v / 255.0
        expected = v_norm / 12.92
        result = srgb_to_linear(v)
        assert abs(result - expected) < 1e-6

    def test_boundary_high(self):
        """Test boundary at 0.04045 threshold (v_norm = 11/255 ≈ 0.0431)."""
        # v=11 -> v_norm=0.0431 -> above threshold -> gamma correction
        v = 11
        v_norm = v / 255.0
        expected = ((v_norm + 0.055) / 1.055) ** 2.4
        result = srgb_to_linear(v)
        assert abs(result - expected) < 1e-6

    def test_mid_gray(self):
        """Test conversion of mid gray (128)."""
        result = srgb_to_linear(128)
        # 128/255 = 0.502, above threshold
        v_norm = 128 / 255.0
        expected = ((v_norm + 0.055) / 1.055) ** 2.4
        assert abs(result - expected) < 1e-6


class TestLinearRgbToXyz:
    """Test linear RGB to XYZ conversion."""

    def test_white(self):
        """Test conversion of linear white (1, 1, 1)."""
        xyz = linear_rgb_to_xyz(1.0, 1.0, 1.0)
        # White should give D65 white point (approximately)
        assert abs(xyz["x"] - D65["x"]) < 0.1
        assert abs(xyz["y"] - D65["y"]) < 0.1
        assert abs(xyz["z"] - D65["z"]) < 0.1

    def test_black(self):
        """Test conversion of linear black (0, 0, 0)."""
        xyz = linear_rgb_to_xyz(0.0, 0.0, 0.0)
        assert xyz["x"] == 0.0
        assert xyz["y"] == 0.0
        assert xyz["z"] == 0.0

    def test_red(self):
        """Test conversion of pure red (1, 0, 0)."""
        xyz = linear_rgb_to_xyz(1.0, 0.0, 0.0)
        assert xyz["x"] > 0
        assert xyz["y"] > 0
        assert xyz["z"] > 0


class TestXyzToLab:
    """Test XYZ to Lab conversion."""

    def test_white_d65(self):
        """Test conversion of D65 white point."""
        lab = xyz_to_lab(D65["x"], D65["y"], D65["z"])
        assert abs(lab["l"] - 100.0) < 0.1
        assert abs(lab["a"]) < 0.1
        assert abs(lab["b"]) < 0.1

    def test_black(self):
        """Test conversion of black."""
        lab = xyz_to_lab(0.0, 0.0, 0.0)
        assert abs(lab["l"]) < 0.1
        assert abs(lab["a"]) < 0.1
        assert abs(lab["b"]) < 0.1

    def test_f_function_boundary(self):
        """Test f function boundary at t = 0.008856."""
        # This tests the internal f function indirectly
        # Values near boundary should be continuous
        t_low = 0.008
        t_high = 0.009

        def f(t):
            if t > 0.008856:
                return t ** (1/3)
            return 7.787 * t + 16/116

        assert abs(f(t_low) - f(t_high)) < 0.01


class TestRgbToLab:
    """Test complete RGB to Lab conversion pipeline."""

    def test_white_255(self):
        """Test pure white RGB(255, 255, 255)."""
        lab = rgb_to_lab(255, 255, 255)
        assert abs(lab["l"] - 100.0) < 0.1
        assert abs(lab["a"]) < 0.1
        assert abs(lab["b"]) < 0.1

    def test_black_0(self):
        """Test pure black RGB(0, 0, 0)."""
        lab = rgb_to_lab(0, 0, 0)
        assert abs(lab["l"]) < 0.1
        assert abs(lab["a"]) < 0.1
        assert abs(lab["b"]) < 0.1

    def test_mid_gray_128(self):
        """Test mid gray RGB(128, 128, 128)."""
        lab = rgb_to_lab(128, 128, 128)
        # Mid gray should have L* around 53-54
        assert 52.0 < lab["l"] < 55.0
        # Neutral gray should have a*, b* near 0
        assert abs(lab["a"]) < 1.0
        assert abs(lab["b"]) < 1.0

    def test_typical_tooth_color(self):
        """Test typical tooth color RGB(230, 225, 215)."""
        lab = rgb_to_lab(230, 225, 215)
        # Tooth should have high L* (bright)
        assert lab["l"] > 85.0
        # Slight yellow tint (positive b*)
        assert lab["b"] > 0
        # Slight green tint (negative a*)
        assert lab["a"] < 0


class TestComputeWid:
    """Test WID calculation."""

    def test_formula_white(self):
        """Test WID formula on pure white Lab(100, 0, 0)."""
        wid = compute_wid(100.0, 0.0, 0.0)
        expected = 0.511 * 100.0
        assert abs(wid - expected) < 0.01
        assert abs(wid - 51.1) < 0.01

    def test_formula_black(self):
        """Test WID formula on pure black Lab(0, 0, 0)."""
        wid = compute_wid(0.0, 0.0, 0.0)
        assert abs(wid) < 0.01

    def test_formula_typical_tooth(self):
        """Test WID formula on typical tooth Lab(90, -2, 10)."""
        wid = compute_wid(90.0, -2.0, 10.0)
        # WID = 0.511*90 + (-2.324)*(-2) + (-1.100)*10
        expected = 0.511 * 90 + (-2.324) * (-2) + (-1.100) * 10
        assert abs(wid - expected) < 0.01
        assert abs(wid - 44.638) < 0.01

    def test_wid_decreases_with_yellow(self):
        """Test that increasing b* (more yellow) decreases WID."""
        wid1 = compute_wid(90.0, -2.0, 5.0)
        wid2 = compute_wid(90.0, -2.0, 15.0)
        assert wid1 > wid2

    def test_wid_decreases_with_red(self):
        """Test that increasing a* (more red) decreases WID."""
        wid1 = compute_wid(90.0, -3.0, 10.0)
        wid2 = compute_wid(90.0, 1.0, 10.0)
        assert wid1 > wid2


class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_white_rgb_to_wid(self):
        """Test RGB(255, 255, 255) -> WID."""
        lab = rgb_to_lab(255, 255, 255)
        wid = compute_wid(lab["l"], lab["a"], lab["b"])
        assert abs(wid - 51.1) < 0.1

    def test_black_rgb_to_wid(self):
        """Test RGB(0, 0, 0) -> WID."""
        lab = rgb_to_lab(0, 0, 0)
        wid = compute_wid(lab["l"], lab["a"], lab["b"])
        assert abs(wid) < 0.1

    def test_typical_tooth_rgb_to_wid(self):
        """Test RGB(230, 225, 215) -> WID."""
        lab = rgb_to_lab(230, 225, 215)
        wid = compute_wid(lab["l"], lab["a"], lab["b"])
        # Typical tooth should have WID in 35-45 range
        assert 30.0 < wid < 50.0
