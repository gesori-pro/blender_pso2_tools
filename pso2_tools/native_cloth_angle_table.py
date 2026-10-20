"""Quantized integer-degree coefficients for FLTD initialization.

cos/sin outputs of live 273A3A0 after the native 16-bit-angle conversion.
Only the finite byte-angle domain consumed by 28EF2D0 is represented.
No client machine code is loaded by this table.
"""

import numpy as np

ANGLE_BITS = (
    (0x3f800000, 0x00000000),  # 0 degrees
    (0x3f7ff606, 0x3c8eef6b),  # 1 degrees
    (0x3f7fd819, 0x3d0ee9d9),  # 2 degrees
    (0x3f7fa63b, 0x3d5650d8),  # 3 degrees
    (0x3f7f6070, 0x3d8ed392),  # 4 degrees
    (0x3f7f06bd, 0x3db27395),  # 5 degrees
    (0x3f7e992a, 0x3dd605b1),  # 6 degrees
    (0x3f7e17be, 0x3df9871d),  # 7 degrees
    (0x3f7d8285, 0x3e0e7a8c),  # 8 degrees
    (0x3f7cd989, 0x3e20266d),  # 9 degrees
    (0x3f7c1cd9, 0x3e31c5d4),  # 10 degrees
    (0x3f7b4c81, 0x3e435760),  # 11 degrees
    (0x3f7a6894, 0x3e54d9b2),  # 12 degrees
    (0x3f797122, 0x3e664b6c),  # 13 degrees
    (0x3f78663f, 0x3e77ab34),  # 14 degrees
    (0x3f774800, 0x3e847bd6),  # 15 degrees
    (0x3f76167a, 0x3e8d17c1),  # 16 degrees
    (0x3f74d1c7, 0x3e95a8aa),  # 17 degrees
    (0x3f7379fe, 0x3e9e2dea),  # 18 degrees
    (0x3f720f3b, 0x3ea6a6d7),  # 19 degrees
    (0x3f70919b, 0x3eaf12c5),  # 20 degrees
    (0x3f6f013a, 0x3eb7710e),  # 21 degrees
    (0x3f6d5e39, 0x3ebfc10c),  # 22 degrees
    (0x3f6ba643, 0x3ec80da8),  # 23 degrees
    (0x3f69de49, 0x3ed03f06),  # 24 degrees
    (0x3f680415, 0x3ed86028),  # 25 degrees
    (0x3f6617cb, 0x3ee0706d),  # 26 degrees
    (0x3f641993, 0x3ee86f35),  # 27 degrees
    (0x3f620993, 0x3ef05bde),  # 28 degrees
    (0x3f5fe7f5, 0x3ef835cc),  # 29 degrees
    (0x3f5db4e3, 0x3efffc60),  # 30 degrees
    (0x3f5b708a, 0x3f03d780),  # 31 degrees
    (0x3f591b16, 0x3f07a68a),  # 32 degrees
    (0x3f56b4b6, 0x3f0b6b01),  # 33 degrees
    (0x3f543d9a, 0x3f0f249b),  # 34 degrees
    (0x3f51b5f3, 0x3f12d30c),  # 35 degrees
    (0x3f4f1df3, 0x3f16760c),  # 36 degrees
    (0x3f4c75cf, 0x3f1a0d51),  # 37 degrees
    (0x3f49bdbc, 0x3f1d9894),  # 38 degrees
    (0x3f46f5f0, 0x3f21178f),  # 39 degrees
    (0x3f441ea1, 0x3f2489fd),  # 40 degrees
    (0x3f413809, 0x3f27ef96),  # 41 degrees
    (0x3f3e4262, 0x3f2b4819),  # 42 degrees
    (0x3f3b3de6, 0x3f2e9342),  # 43 degrees
    (0x3f382ad3, 0x3f31d0d0),  # 44 degrees
    (0x3f3504f3, 0x3f3504f3),  # 45 degrees
    (0x3f31d555, 0x3f382676),  # 46 degrees
    (0x3f2e97da, 0x3f3b399e),  # 47 degrees
    (0x3f2b4cc3, 0x3f3e3e2e),  # 48 degrees
    (0x3f27f453, 0x3f4133ea),  # 49 degrees
    (0x3f248ecc, 0x3f441a98),  # 50 degrees
    (0x3f211c71, 0x3f46f1fc),  # 51 degrees
    (0x3f1d9d88, 0x3f49b9de),  # 52 degrees
    (0x3f1a1255, 0x3f4c7207),  # 53 degrees
    (0x3f167b20, 0x3f4f1a43),  # 54 degrees
    (0x3f12d831, 0x3f51b259),  # 55 degrees
    (0x3f0f29d0, 0x3f543a17),  # 56 degrees
    (0x3f0b7046, 0x3f56b14a),  # 57 degrees
    (0x3f07abdd, 0x3f5917c2),  # 58 degrees
    (0x3f03dce2, 0x3f5b6d4e),  # 59 degrees
    (0x3f0003a0, 0x3f5db1c0),  # 60 degrees
    (0x3ef840c7, 0x3f5fe4e9),  # 61 degrees
    (0x3ef066f5, 0x3f6206a0),  # 62 degrees
    (0x3ee87a66, 0x3f6416b9),  # 63 degrees
    (0x3ee07bb8, 0x3f66150a),  # 64 degrees
    (0x3ed86b8b, 0x3f68016d),  # 65 degrees
    (0x3ed04a80, 0x3f69dbbb),  # 66 degrees
    (0x3ec81939, 0x3f6ba3ce),  # 67 degrees
    (0x3ebfccb0, 0x3f6d5bdf),  # 68 degrees
    (0x3eb77cc8, 0x3f6efefa),  # 69 degrees
    (0x3eaf1e92, 0x3f708f75),  # 70 degrees
    (0x3ea6b2b7, 0x3f720d30),  # 71 degrees
    (0x3e9e39dd, 0x3f73780d),  # 72 degrees
    (0x3e95b4ae, 0x3f74cff1),  # 73 degrees
    (0x3e8d23d4, 0x3f7614bf),  # 74 degrees
    (0x3e8487fa, 0x3f77465f),  # 75 degrees
    (0x3e77c390, 0x3f7864ba),  # 76 degrees
    (0x3e6663e3, 0x3f796fb9),  # 77 degrees
    (0x3e54f241, 0x3f7a6746),  # 78 degrees
    (0x3e437007, 0x3f7b4b4f),  # 79 degrees
    (0x3e31de90, 0x3f7c1bc1),  # 80 degrees
    (0x3e203f3d, 0x3f7cd88e),  # 81 degrees
    (0x3e0e936c, 0x3f7d81a5),  # 82 degrees
    (0x3df9b8fe, 0x3f7e16fa),  # 83 degrees
    (0x3dd637ac, 0x3f7e9881),  # 84 degrees
    (0x3db2a5a7, 0x3f7f0631),  # 85 degrees
    (0x3d8f05b7, 0x3f7f5fff),  # 86 degrees
    (0x3d56b540, 0x3f7fa5e6),  # 87 degrees
    (0x3d0f4e37, 0x3f7fd7e1),  # 88 degrees
    (0x3c8fb845, 0x3f7ff5ea),  # 89 degrees
    (0x00000000, 0x3f800000),  # 90 degrees
)


def coefficients(degrees):
    if not isinstance(degrees, int) or not 0 <= degrees <= 90:
        raise ValueError("Integer angle must be within 0..90 degrees")
    return np.asarray(ANGLE_BITS[degrees], dtype=np.uint32).view(np.float32).copy()
