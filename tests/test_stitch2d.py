import os
import hashlib
import shlex

import numpy as np
import pytest

from stitch2d import Mosaic, StructuredMosaic, OpenCVTile, ScikitImageTile, build_grid
from stitch2d.__main__ import main


TILE_PATH = os.path.abspath(os.path.join(__file__, "..", "tiles"))


def hash_image(im):
    """Hashes an image array

    Parameters
    ----------
    im : np.ndarray
        an array containing image data

    Returns
    -------
    str
        MD5 hash of array
    """
    imdata = [f"{n:.3f}" for n in im.flat]
    return hashlib.md5(str(imdata).encode("utf-8")).hexdigest()


def map_coords_to_filename(params):
    """Maps coordinates from params to filenames

    Parameters
    ----------
    params : dict
        mosaic params

    Returns
    -------
    dict
        dict linking filenames to coordinates
    """
    return {params["filenames"][i]: c for i, c in params["coords"].items()}


@pytest.fixture(scope="session")
def output_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("output")


@pytest.fixture(scope="session")
def cv_tiles():
    mosaic = Mosaic(TILE_PATH, tile_class=OpenCVTile)
    mosaic.detect_and_extract()
    return mosaic.tiles


@pytest.fixture(scope="session")
def cv_aligned_mosaic(cv_tiles):
    mosaic = Mosaic(cv_tiles.copy())
    mosaic.align()
    return mosaic


@pytest.fixture(scope="session")
def cv_aligned_structured_mosaic(cv_tiles):
    mosaic = StructuredMosaic(cv_tiles.copy())
    mosaic.align(limit=5)
    return mosaic


@pytest.fixture(scope="session")
def sk_tiles():
    mosaic = Mosaic(TILE_PATH, tile_class=ScikitImageTile)
    mosaic.detect_and_extract()
    return mosaic.tiles


@pytest.fixture(scope="session")
def sk_aligned_mosaic(sk_tiles):
    mosaic = Mosaic(sk_tiles.copy())
    mosaic.align()
    return mosaic


@pytest.fixture(scope="session")
def sk_aligned_structured_mosaic(sk_tiles):
    mosaic = StructuredMosaic(sk_tiles.copy())
    mosaic.align(limit=4)
    return mosaic


@pytest.fixture()
def params():
    return {
        "metadata": {"shape": [2, 3], "size": 6, "tile_shape": [442, 512]},
        "coords": {
            "0": [0.0, 9.726760864257812],
            "1": [395.91877841949463, 4.91668701171875],
            "2": [791.5322999954224, 0.0],
            "3": [2.2812623977661133, 389.53586196899414],
            "4": [398.6409387588501, 384.7066230773926],
            "5": [794.4158906936646, 379.5469093322754],
        },
        "filenames": {
            "0": "stitch2d_ex[Grid@0 0].jpg",
            "1": "stitch2d_ex[Grid@0 1].jpg",
            "2": "stitch2d_ex[Grid@0 2].jpg",
            "3": "stitch2d_ex[Grid@1 0].jpg",
            "4": "stitch2d_ex[Grid@1 1].jpg",
            "5": "stitch2d_ex[Grid@1 2].jpg",
        },
    }


@pytest.mark.parametrize(
    "fixture_name", ["cv_aligned_structured_mosaic", "sk_aligned_structured_mosaic"]
)
def test_copy(fixture_name, params, request):
    mosaic = request.getfixturevalue(fixture_name).copy()
    for tile in mosaic.tiles:
        assert tile.grid == mosaic.grid


@pytest.mark.parametrize("fixture_name", ["cv_aligned_mosaic", "sk_aligned_mosaic"])
def test_mosaic(fixture_name, params, request, output_dir):
    mosaic = request.getfixturevalue(fixture_name).copy()
    assert mosaic.placed

    mosaic.save(str(output_dir / f"test_mosaic_{fixture_name}.jpg"))

    p1 = map_coords_to_filename(mosaic.params)
    p2 = map_coords_to_filename(params)
    for fn, (y1, x1) in p1.items():
        y2, x2 = p2[fn]
        assert abs(y1 - y2) < 10
        assert abs(x1 - x2) < 10


@pytest.mark.parametrize(
    "fixture_name", ["cv_aligned_structured_mosaic", "sk_aligned_structured_mosaic"]
)
def test_structured_mosaic(fixture_name, params, request, output_dir):
    mosaic = request.getfixturevalue(fixture_name).copy()
    assert mosaic.placed

    mosaic.save(str(output_dir / f"test_structured_mosaic_{fixture_name}.jpg"))

    p1 = map_coords_to_filename(mosaic.params)
    p2 = map_coords_to_filename(params)
    for fn, (y1, x1) in p1.items():
        y2, x2 = p2[fn]

        assert abs(y1 - y2) < 10
        assert abs(x1 - x2) < 10


def test_mosaic_from_downsampled(cv_tiles, params, output_dir):
    mosaic = StructuredMosaic(cv_tiles.copy())
    mosaic.downsample(0.3)
    mosaic.align(limit=5)
    mosaic.reset_tiles()
    mosaic.build_out()
    assert mosaic.placed

    mosaic.save(str(output_dir / f"test_mosaic_from_downsampled.jpg"))

    p1 = map_coords_to_filename(mosaic.params)
    p2 = map_coords_to_filename(params)
    for fn, (y1, x1) in p1.items():
        y2, x2 = p2[fn]
        assert abs(y1 - y2) < 10
        assert abs(x1 - x2) < 10


def test_mosaic_with_varying_tile_dims(params, output_dir):
    mosaic = StructuredMosaic(TILE_PATH, tile_class=OpenCVTile)

    # Trim rows from first grid row
    mosaic.grid[0][0].imdata = np.delete(mosaic.grid[0][0].imdata, range(100), axis=0)
    mosaic.grid[0][1].imdata = np.delete(mosaic.grid[0][1].imdata, range(100), axis=0)

    # Trim columns from first grid column
    mosaic.grid[0][0].imdata = np.delete(mosaic.grid[0][0].imdata, range(100), axis=1)
    mosaic.grid[1][0].imdata = np.delete(mosaic.grid[1][0].imdata, range(100), axis=1)
    mosaic.grid[2][0].imdata = np.delete(mosaic.grid[2][0].imdata, range(100), axis=1)

    mosaic = StructuredMosaic(mosaic.grid)
    mosaic.save(str(output_dir / f"test_mosaic_with_varying_tile_dims.jpg"))

    assert mosaic.stitch().shape == (1226, 924)


def test_ragged_mosaic(cv_tiles, output_dir):
    mosaic = StructuredMosaic(cv_tiles.copy()[:-1])
    mosaic.align()

    # FIXME: Tile indexes are different than params, so placement check fails
    mosaic.save(str(output_dir / f"test_ragged_mosaic.jpg"))


@pytest.mark.parametrize(
    "fixture_name,expected", [("cv_aligned_mosaic", "1a51f1c3ec3d066996301cf4e39d2cac")]
)
def test_smoothing_mosaic(fixture_name, expected, params, request, output_dir):
    mosaic = request.getfixturevalue(fixture_name).copy()
    mosaic.load_params(params)
    mosaic.smooth_seams()

    mosaic.save(str(output_dir / f"test_smoothing_mosaic_{fixture_name}.jpg"))

    assert hash_image(mosaic.stitch()) == expected


@pytest.mark.parametrize("fixture_name", ["cv_aligned_structured_mosaic"])
def test_build_out_from_placed(fixture_name, params, request, output_dir):
    mosaic = request.getfixturevalue(fixture_name).copy()
    mosaic.build_out(from_placed=True)
    assert mosaic.placed

    mosaic.save(str(output_dir / f"test_build_out_from_placed_{fixture_name}.jpg"))

    p1 = map_coords_to_filename(mosaic.params)
    p2 = map_coords_to_filename(params)
    for fn, (y1, x1) in p1.items():
        y2, x2 = p2[fn]
        assert abs(y1 - y2) < 10
        assert abs(x1 - x2) < 10


@pytest.mark.parametrize("fixture_name", ["cv_aligned_structured_mosaic"])
def test_build_out_from_scratch(fixture_name, params, request, output_dir):
    mosaic = request.getfixturevalue(fixture_name).copy()
    mosaic.build_out(from_placed=False)
    assert mosaic.placed

    mosaic.save(str(output_dir / f"test_build_out_from_scratch_{fixture_name}.jpg"))

    p1 = map_coords_to_filename(mosaic.params)
    p2 = map_coords_to_filename(params)
    for fn, (y1, x1) in p1.items():
        y2, x2 = p2[fn]
        assert abs(y1 - y2) < 10
        assert abs(x1 - x2) < 10


def test_building_mosaic_from_paths(cv_tiles):
    msc1 = Mosaic(cv_tiles)
    msc2 = Mosaic([t.source for t in cv_tiles])
    for t1, t2 in zip(msc1.tiles, msc2.tiles):
        assert hash_image(t1.imdata) == hash_image(t2.imdata)
        assert t1.id != t2.id
        assert t1.row == t2.row
        assert t1.col == t2.col


def test_building_mosaic_from_grid(cv_tiles):
    msc1 = Mosaic(cv_tiles)
    msc2 = Mosaic(msc1.copy().grid)
    for t1, t2 in zip(msc1.tiles, msc2.tiles):
        assert hash_image(t1.imdata) == hash_image(t2.imdata)
        assert t1.id != t2.id
        assert t1.row == t2.row
        assert t1.col == t2.col


def test_building_mosaic_from_sem_params(cv_tiles):
    msc1 = StructuredMosaic(cv_tiles)  # grabs params from filenames
    msc2 = StructuredMosaic(cv_tiles, dim=(3, 4), direction="vertical")
    for t1, t2 in zip(msc1.tiles, msc2.tiles):
        assert hash_image(t1.imdata) == hash_image(t2.imdata)
        assert t1.id != t2.id
        assert t1.row == t2.row
        assert t1.col == t2.col


def test_mosaic_resize_and_reset(cv_aligned_mosaic):
    mosaic = cv_aligned_mosaic.copy()
    tile = mosaic.tiles[0]
    assert tile.shape == (442, 512)
    mosaic.resize((221, 256))
    assert tile.shape == (221, 256)
    mosaic.reset_tiles()
    assert tile.shape == (442, 512)


def test_stitch_channel_order(cv_aligned_mosaic):
    mosaic = cv_aligned_mosaic.copy()
    bgr_mosaic = mosaic.stitch()
    rgb_mosaic = mosaic.stitch("rgb")
    assert np.array_equal(bgr_mosaic[..., 0], rgb_mosaic[..., 2])
    assert np.array_equal(bgr_mosaic[..., 1], rgb_mosaic[..., 1])
    assert np.array_equal(bgr_mosaic[..., 2], rgb_mosaic[..., 0])


def test_load_params(cv_aligned_mosaic, output_dir):
    mosaic = cv_aligned_mosaic.copy()
    path = str(output_dir / "test_load_params.json")

    mosaic.save_params(path)

    for tile in mosaic.tiles:
        tile.y = None
        tile.x = None
    mosaic.load_params(path)

    assert mosaic.params == cv_aligned_mosaic.params


def test_missing_params_file(cv_aligned_mosaic):
    with pytest.raises(FileNotFoundError) as e:
        cv_aligned_mosaic.load_params("fake.json")


def test_build_grid_ul_horizontal_raster():
    items = list(range(1, 6))
    expected = [[1, 2, 3], [4, 5, None]]
    assert build_grid(items, 3, "upper left", "horizontal", "raster") == expected


def test_build_grid_ul_horizontal_snake():
    items = list(range(1, 6))
    expected = [[1, 2, 3], [None, 5, 4]]
    assert build_grid(items, 3, "upper left", "horizontal", "snake") == expected


def test_build_grid_ul_vertical_raster():
    items = list(range(1, 6))
    expected = [[1, 3, 5], [2, 4, None]]
    assert build_grid(items, 2, "upper left", "vertical", "raster") == expected


def test_build_grid_ul_vertical_snake():
    items = list(range(1, 6))
    expected = [[1, 4, 5], [2, 3, None]]
    assert build_grid(items, 2, "upper left", "vertical", "snake") == expected


def test_build_grid_ur_horizontal_raster():
    items = list(range(1, 6))
    expected = [[3, 2, 1], [None, 5, 4]]
    assert build_grid(items, 3, "upper right", "horizontal", "raster") == expected


def test_build_grid_ur_horizontal_snake():
    items = list(range(1, 6))
    expected = [[3, 2, 1], [4, 5, None]]
    assert build_grid(items, 3, "upper right", "horizontal", "snake") == expected


def test_build_grid_ur_vertical_raster():
    items = list(range(1, 6))
    expected = [[5, 3, 1], [None, 4, 2]]
    assert build_grid(items, 2, "upper right", "vertical", "raster") == expected


def test_build_grid_ur_vertical_snake():
    items = list(range(1, 6))
    expected = [[5, 4, 1], [None, 3, 2]]
    assert build_grid(items, 2, "upper right", "vertical", "snake") == expected


def test_build_grid_ll_horizontal_raster():
    items = list(range(1, 6))
    expected = [[4, 5, None], [1, 2, 3]]
    assert build_grid(items, 3, "lower left", "horizontal", "raster") == expected


def test_build_grid_ll_horizontal_snake():
    items = list(range(1, 6))
    expected = [[None, 5, 4], [1, 2, 3]]
    assert build_grid(items, 3, "lower left", "horizontal", "snake") == expected


def test_build_grid_ll_vertical_raster():
    items = list(range(1, 6))
    expected = [[2, 4, None], [1, 3, 5]]
    assert build_grid(items, 2, "lower left", "vertical", "raster") == expected


def test_build_grid_ll_vertical_snake():
    items = list(range(1, 6))
    expected = [[2, 3, None], [1, 4, 5]]
    assert build_grid(items, 2, "lower left", "vertical", "snake") == expected


def test_build_grid_lr_horizontal_raster():
    items = list(range(1, 6))
    expected = [[None, 5, 4], [3, 2, 1]]
    assert build_grid(items, 3, "lower right", "horizontal", "raster") == expected


def test_build_grid_lr_horizontal_snake():
    items = list(range(1, 6))
    expected = [[4, 5, None], [3, 2, 1]]
    assert build_grid(items, 3, "lower right", "horizontal", "snake") == expected


def test_build_grid_lr_vertical_raster():
    items = list(range(1, 6))
    expected = [[None, 4, 2], [5, 3, 1]]
    assert build_grid(items, 2, "lower right", "vertical", "raster") == expected


def test_build_grid_lr_vertical_snake():
    items = list(range(1, 6))
    expected = [[None, 3, 2], [5, 4, 1]]
    assert build_grid(items, 2, "lower right", "vertical", "snake") == expected


@pytest.mark.parametrize(
    "kwargs", [{"origin": "invalid"}, {"direction": "invalid"}, {"pattern": "invalid"}]
)
def test_build_grid_invalid_kwarg(kwargs):
    with pytest.raises(ValueError) as e:
        build_grid([], 3, **kwargs)


def test_command(output_dir):
    jpg_path = str(output_dir / "test_command.jpg")
    json_path = str(output_dir / "test_command.json")

    cmd = (
        f'stitch2d "{TILE_PATH}" -dim 3 -mp 0.3 -limit 5 -origin ul'
        f' -direction vertical -pattern raster -param_file "{json_path}"'
        f' --smooth --build_out -output "{jpg_path}"'
    )

    main(shlex.split(cmd.split(" ", 1)[1]))


def test_command_with_error(output_dir):
    jpg_path = str(output_dir / "test_command_with_error.jpg")
    json_path = str(output_dir / "test_command_with_error.json")

    cmd = (
        f'stitch2d "{TILE_PATH}" -dim 3 -mp 0.3 -limit 5 -origin ul'
        f' -direction vertical -pattern ratser -param_file "{json_path}"'
        f' --smooth --build_out -output "{jpg_path}"'
    )

    with pytest.raises(SystemExit) as e:
        main(shlex.split(cmd.split(" ", 1)[1]))
    assert e.value.code == 2
