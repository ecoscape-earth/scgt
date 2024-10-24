# %%
from scgt import *


# %%
gt = GeoTiff.from_file("test_files/test_Psilopogon_rafflesii_300.tif")
gt.width, gt.height

# %%
b = 100
p = 0
it = Reader(gt, b=b, p=p, w=400, h=400)
out_gt = gt.crop_to_new_file(b - p, in_memory=True)
tile_pos = []
for tile in it:
    tile_pos.append((tile.x, tile.y, tile.w, tile.h, tile.b))
    # print("tile:", tile)
    out_gt.set_tile(tile, offset = b - p, verbose=False)
desired_tile_pos = [
    (100, 100, 400, 400, 100), (500, 100, 400, 400, 100), (900, 100, 400, 400, 100), (1300, 100, 220, 400, 100), 
    (100, 500, 400, 400, 100), (500, 500, 400, 400, 100), (900, 500, 400, 400, 100), (1300, 500, 220, 400, 100), 
    (100, 900, 400, 400, 100), (500, 900, 400, 400, 100), (900, 900, 400, 400, 100), (1300, 900, 220, 400, 100), 
    (100, 1300, 400, 196, 100), (500, 1300, 400, 196, 100), (900, 1300, 400, 196, 100), (1300, 1300, 220, 196, 100)]
assert tile_pos == desired_tile_pos

# %%
b = 100
p = 100
it = Reader(gt, b=100, p=100, w=400, h=400)
tit = Reader(gt, b=b, p=p, w=400, h=400)
out_gt = gt.crop_to_new_file(b - p, in_memory=True)
tile_pos = []
for tile in it:
    tile_pos.append((tile.x, tile.y, tile.w, tile.h, tile.b))
    # print("tile:", tile)
    out_gt.set_tile(tile, offset = b - p, verbose=False)
desired_tile_pos = [
    (0, 0, 400, 400, 100), (400, 0, 400, 400, 100), (800, 0, 400, 400, 100), (1200, 0, 400, 400, 100), (1600, 0, 20, 400, 100), 
    (0, 400, 400, 400, 100), (400, 400, 400, 400, 100), (800, 400, 400, 400, 100), (1200, 400, 400, 400, 100), (1600, 400, 20, 400, 100), 
    (0, 800, 400, 400, 100), (400, 800, 400, 400, 100), (800, 800, 400, 400, 100), (1200, 800, 400, 400, 100), (1600, 800, 20, 400, 100), 
    (0, 1200, 400, 396, 100), (400, 1200, 400, 396, 100), (800, 1200, 400, 396, 100), (1200, 1200, 400, 396, 100), (1600, 1200, 20, 396, 100)]
assert tile_pos == desired_tile_pos


# %%



