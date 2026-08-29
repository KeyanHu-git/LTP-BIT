# Website result selection

The website uses unmodified 256 x 256 PNG files copied from the native result archive. Every displayed LTP-BT result ranks first by RGB PSNR against its paired target among the ten translated methods in the archive.

PSNR is calculated from the mean squared error over all RGB pixels at the native 0-255 range. Input and target images are not included as candidate methods.

| Website slot | Source sample | Scene group | LTP-BT PSNR (dB) | Margin over runner-up (dB) |
| --- | --- | --- | ---: | ---: |
| QXS 01 | `17253.png` | Waterfront & Maritime | 15.345 | +0.369 |
| QXS 02 | `17686.png` | Transport & Industry | 18.742 | +1.176 |
| QXS 03 | `19334.png` | Transport & Industry | 14.931 | +2.728 |
| QXS 04 | `14502.png` | Urban Buildings | 12.791 | +0.271 |
| SpaceNet6 01 | `20190804140510_20190804140727_tile_4705_y429_x000.png` | Road Networks | 22.268 | +1.068 |
| SpaceNet6 02 | `20190804122434_20190804122704_tile_6465_y644_x000.png` | Port Areas | 18.270 | +0.648 |
| SpaceNet6 03 | `20190804120805_20190804121023_tile_6210_y000_x644.png` | Industrial Facilities | 15.022 | +1.362 |
| SpaceNet6 04 | `20190804145216_20190804145445_tile_6511_y429_x644.png` | Buildings | 16.684 | +0.033 |
| Chesapeake 01 | `va_m_3807821_ne_17_1_y02560_x01792.png` | Cloud / Water | 18.936 | +0.285 |
| Chesapeake 02 | `de_m_3807504_se_18_1_y01280_x01024.png` | Cloud / Water | 21.927 | +6.517 |
| Chesapeake 03 | `wv_m_3907858_nw_17_1_y02560_x03328.png` | Road Networks | 19.665 | +0.007 |

The incomplete SpaceNet6 sample containing a `.downloading` DiffusionSat file is excluded because its full group ranking cannot be certified.

The interactive viewer also includes optional, scene-aligned outputs from CycleGAN, CFCA-SET, StegoGAN, BBDM, ControlNet, Uni-ControlNet, C-DiffSET, DiffusionSat, and Text2Earth. These baseline layers are copied without modification from the same native result archive and can be added from the viewer's method picker.
