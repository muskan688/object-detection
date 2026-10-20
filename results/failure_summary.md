Threshold: confidence >= 0.25, IoU >= 0.5

**Recall by weather**

| Group | Found | Total | Recall |
|---|---|---|---|
| fog | 237 | 296 | 0.80 |
| rain | 201 | 248 | 0.81 |
| sand | 290 | 370 | 0.78 |
| snow | 290 | 382 | 0.76 |

**Recall by object size**

| Group | Found | Total | Recall |
|---|---|---|---|
| large (>96x96) | 465 | 513 | 0.91 |
| medium | 477 | 633 | 0.75 |
| small (<32x32) | 76 | 150 | 0.51 |

- dark images (mean brightness < 90): 8 images, recall 0.76
- brighter images: 146 images, recall 0.79

**Worst 12 images** (see results/failures/):

| Image | Weather | GT | TP | FP | FN | Brightness |
|---|---|---|---|---|---|---|
| sand_sand_storm_g2_-007.jpg | sand | 49 | 29 | 18 | 20 | 130.1 |
| snow_snow_storm-158.jpg | snow | 53 | 33 | 18 | 20 | 172.5 |
| rain_rain_storm-016.jpg | rain | 37 | 28 | 20 | 9 | 145.9 |
| fog_haze-048.jpg | fog | 30 | 25 | 19 | 5 | 124.5 |
| sand_sand_storm-239.jpg | sand | 48 | 40 | 16 | 8 | 182.4 |
| snow_snow_storm-176.jpg | snow | 32 | 24 | 15 | 8 | 216.7 |
| snow_snow_storm-362.jpg | snow | 40 | 28 | 11 | 12 | 214.3 |
| rain_rain_storm-485.jpg | rain | 33 | 29 | 17 | 4 | 147.2 |
| snow_snow_storm-299.jpg | snow | 35 | 31 | 17 | 4 | 118.3 |
| sand_sand_storm_g2_-075.jpg | sand | 41 | 34 | 10 | 7 | 173.9 |
| snow_snow_storm-119.jpg | snow | 23 | 16 | 7 | 7 | 207.0 |
| fog_mist-189.jpg | fog | 14 | 9 | 8 | 5 | 118.9 |