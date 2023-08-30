# SCGT

## Santa Cruz Geographical Toolkit

This library provides convenience functions to read and write GeoTiff files, 
including iterating over their tiles. 

**Main Authors:**

* Jasmine Tai (cjtai@ucsc.edu)
* Natalie Valett (nvalett@ucsc.edu)

**Other Conributors:**

* Luca de Alfaro (luca@ucsc.edu)


## Basic Usage

```
from scgt import GeoTiff, Tile, Reader

gt_in = GeoTiff("somefile.tif")
reader = gt.get_reader(b=border_size, w=block_size, h=block_size)
with gt_in.clone_shape("outputfile.tif") as gt_out:
    for tile in reader:
        new_tile = process(tile)
        gt_out.set_tile(new_tile)
```

For more details, see [the documentation](Documentation.md).

