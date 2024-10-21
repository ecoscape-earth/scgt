<a id="scgt"></a>

# scgt

<a id="scgt.GeoTiff"></a>

## GeoTiff Objects

```python
class GeoTiff(object)
```

A GeoTiff object provides an interface to reading/writing geotiffs in
a tiled fashion, and to many other additional operations.

<a id="scgt.GeoTiff.__init__"></a>

#### \_\_init\_\_

```python
def __init__(file=None, memory_file=None)
```

Initializes a GeoTiff object.

**Arguments**:

- `file`: A file object for the geotiff.
- `memory_file`: The corresponding opened MemoryFile object if the geotiff is stored in memory.
Properties:
dataset : open geotiff file
size : tuple -> raster width, height
bands : int -> number of bands in geotiff
transform : the dataset's geospatial transform - an affine transformation matrix that maps pixel
            locations in (row, col) coordinates to (x, y) spatial positions
corners : array of lat/long of corners, in order top left, top right, bottom right, bottom left
block_shapes : array with the shape of blocks for all bands
                i.e [(1, (3, 791)), (2, (3, 791))]
profile : Geotiff profile - used for writing metadata to new file
memory_file : the associated memory file object if geotiff is stored in memory

<a id="scgt.GeoTiff.__enter__"></a>

#### \_\_enter\_\_

```python
def __enter__()
```

Context manager entry for use in with statements.

<a id="scgt.GeoTiff.__exit__"></a>

#### \_\_exit\_\_

```python
def __exit__(type, value, traceback)
```

Context manager exit.

<a id="scgt.GeoTiff.from_file"></a>

#### from\_file

```python
@classmethod
def from_file(cls, filename)
```

Creates a GeoTiff object from a filename.

**Arguments**:

- `filename`: filename to open in read mode.

**Returns**:

the GeoTiff object.

<a id="scgt.GeoTiff.create_new_file"></a>

#### create\_new\_file

```python
@classmethod
def create_new_file(cls, filename, profile, no_data_value=None)
```

creates a new file to store empty GeoTiff obj with specified params.

:param filename: file name for the new file. 
:param profile: profile for writing the geotiff.
:param no_data_value (dtype of raster): value to be used as transparent 'nodata' value, otherwise fills 0's
    (note that if geotiff is of unsigned type, like uint8, the no_data value must be  a positive int in the data
    range, which could result in data obstruction. If datatype is signed, we suggest using a negative value)
:return: the GeoTiff object for the new file.


<a id="scgt.GeoTiff.create_memory_file"></a>

#### create\_memory\_file

```python
@classmethod
def create_memory_file(cls, profile, no_data_value=None)
```

creates a temporary file in memory in which to store a GeoTiff obj.

:param profile: profile for writing the geotiff.
:param no_data_value (dtype of raster): value to be used as transparent 'nodata' value, otherwise fills 0's
    (note that if geotiff is of unsigned type, like uint8, the no_data value must be  a positive int in the data range, which could result in data obstruction. If datatype is signed, we suggest using a negative value)
:return: the GeoTiff object for the new file.


<a id="scgt.GeoTiff.clone_shape"></a>

#### clone\_shape

```python
def clone_shape(filename, no_data_value=None, dtype=None)
```

Creates a new geotiff with the indicated filename, cloning the shape of the current one.

:param filename: filename where to create the new clone.
:param no_data_value (dtype of raster): value to be used as transparent 'nodata' value, otherwise fills 0's
    (note that if geotiff is of unsigned type, like uint8, the no_data value must be  a positive int in the data
    range, which could result in data obstruction. If datatype is signed, we suggest using a negative value)
:param dtype (str): datatype of tiff (ref: https://github.com/rasterio/rasterio/blob/master/rasterio/dtypes.py#L21)
:return: the GeoTiff object for the new file.


<a id="scgt.GeoTiff.scale_tiff"></a>

#### scale\_tiff

```python
def scale_tiff(reference_tif=None, scale_x=1, scale_y=1)
```

Scales a tiff.

**Arguments**:

- `reference_tif`: Tiff to be scaled.
- `scale_x`: x scale.
- `scale_y`: y scale.

**Returns**:

Nothing.  The GeoTiff is scaled in place.

<a id="scgt.GeoTiff.get_reader"></a>

#### get\_reader

```python
def get_reader(b=0, p=0, w=None, h=None, pad_value=0)
```

Returns a reader that can iterate over tiles of size w, h , and border b.

The border overlaps, so if you ask for tiles that are (e.g.) 100 x 100 with border 50,
and the geotiff is 1000 x 1000,

**Arguments**:

- `b`: Border (in pixels) around each tile.
- `p`: Padding (in pixels) around the geotiff.
- `w`: width of each tile.
- `h`: height of each tile.
- `pad_value`: padding value to be used around each tile.

**Returns**:

an iterator over tiles. Note that if the geotiff is not of size multiple of w, h,
some tiles may be smaller than others.

<a id="scgt.GeoTiff.set_tile"></a>

#### set\_tile

```python
def set_tile(tile, offset=0, verbose=False)
```

Sets a tile in the geotiff.  

**Arguments**:

- `tile`: the tile to be set.
- `offset`: offset for the tile. 
Each tile has (x, y) coordinates, which define the position of its top left corner in the 
source geotiff. set_tile will place the tile in the destination geotiff, such that the 
top left corner ends up at (x - offset, y - offset). 
Typically, if a tile is obtained via a reader that has border b and padding p, the offset needs
to be set to b - p.            
Note that the function only sets the tile core, not the border.

<a id="scgt.GeoTiff.get_pixel_from_coord"></a>

#### get\_pixel\_from\_coord

```python
def get_pixel_from_coord(coord)
```

Gets pixel on the geotiff that corresponds to the CRS coordinate given

**Arguments**:

- `coord`: coordinate to get (already in geotiff's CRS)

**Returns**:

(x,y) pixel of geotiff which corresponds to coord, or None if out of bounds

<a id="scgt.GeoTiff.get_coord_from_pixel"></a>

#### get\_coord\_from\_pixel

```python
def get_coord_from_pixel(xy, offset='center')
```

Gets CRS coordinate that corresponds to the (x, y) pixel on the geotiff

**Arguments**:

- `xy`: (x,y) pixel of geotiff
- `offset`: defines whether to return one of the corner coordinates or the center
coordinates of the pixel; must be one of 'center', 'ul', 'ur', 'll', 'lr'

**Returns**:

the corresponding CRS coordinate, or None if out of bounds

<a id="scgt.GeoTiff.get_tile_from_coord"></a>

#### get\_tile\_from\_coord

```python
def get_tile_from_coord(coord, tile_scale=4)
```

gets tile of size tile_scale x tile_scale centered at coord from geotiff

**Arguments**:

- `coord`: coordinate to get (already in geotiff's CRS)
- `tile_scale`: size of the tile
(if the pixel resolution is 300m, and tile_scale is 3, the tile will be 900m x 900m centered around coord)

**Returns**:

numpy array representation of the tile, or None if coord is out of range

<a id="scgt.GeoTiff.set_tile_from_coord"></a>

#### set\_tile\_from\_coord

```python
def set_tile_from_coord(coord, value, tile_scale=4)
```

Sets a tile to a (constant) value, given the coordinates. This is useful

to draw on a geotiff.

**Arguments**:

- `coord`: coordinates, in geotiff CRS.
- `value`: value to be set.
- `tile_scale`: size of the tile.
(if the pixel resolution is 300m, and tile_scale is 3, the tile will be 900m x 900m centered around coord)

**Returns**:

Nothing.

<a id="scgt.GeoTiff.get_tile"></a>

#### get\_tile

```python
def get_tile(w, h, b, x, y)
```

Returns a tile at a certain position with border b

reads this window from all bands in Geotiff storing in 3D np array
of dimensions (2b+w, 2b+h, bands)

**Arguments**:

- `w`: width of tile.
- `h`: height of tile.
- `b`: border of tile.
- `x`: x of tile in pixel grid.
- `y`: y of tile in pixel grid.

**Returns**:

The tile.

<a id="scgt.GeoTiff.get_all_as_tile"></a>

#### get\_all\_as\_tile

```python
def get_all_as_tile(b=0)
```

Gets the entire content of the geotiff as a tile with specified border.

**Returns**:

A tile representing the entire geotiff.

<a id="scgt.GeoTiff.get_tile_from_window"></a>

#### get\_tile\_from\_window

```python
def get_tile_from_window(w, border)
```

Gets a tile from a window description.

**Arguments**:

- `w`: A window, as in rasterio.windows.Window
- `border`: Border around the window.

**Returns**:

The tile.

<a id="scgt.GeoTiff.get_rectangle"></a>

#### get\_rectangle

```python
def get_rectangle(range_width, range_height, band=None)
```

reads any square of geotiff x1:x2, y1:y2 into a numpy matrix.

If band is None, returns a 3D numpy matrix, with dimensions (x1 - x0, y1 - y0, c),
where c is the number of channels (bands) of the geotiff.
If band is specified, then the result is a 2D matrix of shape (x1 - x0, y1 - y0).

**Arguments**:

- `range_width`: pair (x0, x1) defining the x-range.
- `range_height`: pair (y0, y1) defining the y-range.
- `band`: which band to get.

**Returns**:

2D Numpy array for the values, if band is specified, otherwise 3D array.

<a id="scgt.GeoTiff.file_write"></a>

#### file\_write

```python
def file_write(filename)
```

Writes itself to a filename, using inherent blocking of geotiff to write

**Arguments**:

- `filename`: Filename to use for writing.

**Returns**:

Nothing.

<a id="scgt.GeoTiff.getAttributeTable"></a>

#### getAttributeTable

```python
def getAttributeTable()
```

Opens geotiff with GDAL, creates attribute table containing tiff's unique values and counts.
To be called in __exit__ as to update the RAT with any new values.

<a id="scgt.GeoTiff.draw_geotiff"></a>

#### draw\_geotiff

```python
def draw_geotiff(width=5, height=5, band=1, title=None)
```

Plots tile's numpy array representation at given band.

**Arguments**:

- `width`: width of what to plot.
- `height`: height of the plot.
- `band`: band to show.

<a id="scgt.GeoTiff.crop_to_new_file"></a>

#### crop\_to\_new\_file

```python
def crop_to_new_file(border, data_type=None, filename=None, in_memory=False)
```

Create a new geotiff by cropping the current one and writing to a new file.

**Arguments**:

- `border`: number of pixels to crop from each side of the geotiff.
- `filename`: path where to write result.
- `in_memory`: whether to create the file in memory only. filename is ignored if True.

**Returns**:

the GeoTiff object for the new file.

<a id="scgt.GeoTiff.crop_to_polygon"></a>

#### crop\_to\_polygon

```python
def crop_to_polygon(polygon, filename=None, in_memory=False)
```

Crop the GeoTIFF using a specified polygon and save to a new file or return in-memory file.

**Arguments**:

- `polygon`: A Shapely polygon specifying the area to crop. Should be in same coordinate system.
- `filename`: The path where the cropped GeoTIFF should be saved.
- `in_memory`: whether to create the file in memory only. filename is ignored if True.

**Returns**:

the GeoTiff object for the new file.

<a id="scgt.GeoTiff.reproject_from_crs"></a>

#### reproject\_from\_crs

```python
def reproject_from_crs(output_path, dest_crs, res=None, resampling="nearest")
```

Reprojects geotiff to a new CRS. A specific resolution given as a tuple (xSize, ySize) can be specified.

See https://rasterio.readthedocs.io/en/latest/topics/reproject.html.

**Arguments**:

- `output_path`: Output path where to write the result.
- `dest_crs`: Destination CRS.
- `res`: desired resolution.
- `resampling`: resampling to use, as in rasterio.warp.Resampling.

<a id="scgt.GeoTiff.reproject_from_tiff"></a>

#### reproject\_from\_tiff

```python
def reproject_from_tiff(output_path, ref_tiff, resampling="near")
```

Reprojects geotiff to the CRS/resolution of a reference tiff. In addition, this ensures that the tiffs are aligned exactly with each other.

See https://gdal.org/programs/gdalwarp.html and https://gdal.org/api/python/osgeo.gdal.html.

**Arguments**:

- `output_path`: output path for reporojection.
- `ref_tiff`: reference geotiff.
- `resampling`: what resampling to do, if any.

<a id="scgt.GeoTiff.get_pixel_value"></a>

#### get\_pixel\_value

```python
def get_pixel_value(x, y)
```

Gets the value of the GeoTiff at a given pixel.

**Arguments**:

- `x`: x-coord of the pixel.
- `y`: y-coord of the pixel.

**Returns**:

The value at the pixel.

<a id="scgt.GeoTiff.get_slow_pixel_value"></a>

#### get\_slow\_pixel\_value

```python
def get_slow_pixel_value(x, y)
```

Gets the value of the GeoTiff at a given pixel.

**Arguments**:

- `x`: x-coord of the pixel.
- `y`: y-coord of the pixel.

**Returns**:

The value at the pixel.
FIXME: this is inefficient.  Redo using Window.

<a id="scgt.GeoTiff.get_average_around_pixel"></a>

#### get\_average\_around\_pixel

```python
def get_average_around_pixel(x, y, size)
```

Gets the average around a pixel.

**Arguments**:

- `x`: x-coord of pixel.
- `y`: y-coord of pixel.
- `size`: size of which to get the average.

**Returns**:

average value of geotiff around point.

<a id="scgt.GeoTiff.close_memory_file"></a>

#### close\_memory\_file

```python
def close_memory_file()
```

Closes the memory file if the GeoTiff object is created in memory only.
This is called automatically if the object is used in a with statement.
Otherwise, it should be called when the GeoTiff object is not needed anymore
so that the memory file is deleted.

<a id="scgt.Reader"></a>

## Reader Objects

```python
class Reader(object)
```

A reader iterates through the tiles of a geotiff.

<a id="scgt.Reader.__init__"></a>

#### \_\_init\_\_

```python
def __init__(geo, b=0, p=0, w=None, h=None, pad_value=0)
```

Initializes a reader.

**Arguments**:

- `geo`: geotiff to read.
- `b`: border of tiles.
- `p`: padding of tiles.
- `w`: width of tiles.
- `h`: height of tiles.
- `pad_value`: value to pad the tiles with.
If the padding is 0, if the region has size m x n, then only the internal portion of size 
(m - 2b) x (n - 2b) will be part of a tile "core", and so processed. 
If the padding is equal to the border, then the entire region of size m x n will be part 
of the tile core.

<a id="scgt.Reader.__iter__"></a>

#### \_\_iter\_\_

```python
def __iter__()
```

Tile iterator.

**Returns**:

the tiles.

<a id="scgt.Reader.__next__"></a>

#### \_\_next\_\_

```python
def __next__()
```

Tile iterator implementation.

**Returns**:

a tile, each time.

<a id="scgt.Tile"></a>

## Tile Objects

```python
class Tile(object)
```

This is a tile.  A tile records where it came from in the GeoTiff (x, y), its size (w, h),
its border b, and its matrix of values m.

<a id="scgt.Tile.__init__"></a>

#### \_\_init\_\_

```python
def __init__(w, h, b, c, x, y, m)
```

Creates a tile.

**Arguments**:

- `w`: Width of inner region only.
- `h`: Height of inner region only.
- `b`: Border width.
- `c`: Channels.
- `x`: x location in raster of core region (excluding border).
- `y`: y location in raster of core region (excluding border).
- `m`: array.

<a id="scgt.Tile.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

String representation of the tile.

**Returns**:

the string representation.

<a id="scgt.Tile.clone_shape"></a>

#### clone\_shape

```python
def clone_shape()
```

Clones the shape of the tile.

**Returns**:

the cloned tile.

<a id="scgt.Tile.get_window"></a>

#### get\_window

```python
def get_window(includes_border=True)
```

gets Window representation of Tile.

By default, Window includes the border.

**Arguments**:

- `includes_border`: Whether to include the border.

**Returns**:

the Window object for the tile.

<a id="scgt.Tile.draw_tile"></a>

#### draw\_tile

```python
def draw_tile(width=5, height=5, band=0, title=None)
```

Plots a tile using matplotlib.

**Arguments**:

- `width`: of figure.
- `height`: of figure.
- `band`: Which band to show.

**Returns**:

Nothing.

<a id="scgt.Tile.fit_to_bounds"></a>

#### fit\_to\_bounds

```python
def fit_to_bounds(width, height)
```

If tile dimensions are not within width/height bounds, alter them.

**Arguments**:

- `width`: desired width.
- `height`: desired height.

**Returns**:

The adjusted tile.

