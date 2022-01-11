"""Command line tool for stitching tilesets"""
import argparse
import sys

import stitch2d


class ParserWithError(argparse.ArgumentParser):
    """Modifies parser to return help if command contains an error"""

    def error(self, message):
        """Show help text if command contains an error

        From http://stackoverflow.com/questions/4042452

        Parameters
        ----------
        message : str
           the message to write to stderr
        """
        sys.stderr.write(f"error: {message}\n")
        self.print_help()
        sys.exit(2)


def _mosaic_callback(args):
    """Stitches a mosaic using the supplied arguments

    stitch2d mosaic path

    Parameters:
    -----------
    args : Namespace.argspace
        Run `stitch2d --help` for available parameters
    """

    tile_classes = {"opencv": stitch2d.OpenCVTile, "skimage": stitch2d.ScikitImageTile}
    tile_class = tile_classes[args.backend]

    path = args.path[0] if len(args.path) == 1 else args.path

    print(f"Creating a mosaic from {path}")
    mosaic = stitch2d.create_mosaic(
        path,
        tile_class=tile_class,
        dim=args.dim,
        origin=args.origin,
        direction=args.direction,
        pattern=args.pattern,
    )

    try:
        if args.param_file:
            print(f"Loading parameters from {args.param_file}")
            mosaic.load_params(args.param_file)
        else:
            raise FileNotFoundError("No param file provided")

    except FileNotFoundError:
        if args.mp:
            print(f"Downsampling tiles to {args.mp} MP")
            mosaic.downsample(args.mp)

        print("Aligning tiles")
        mosaic.align(limit=args.limit)

        if args.param_file:
            print(f"Saving parameters to {args.param_file}")
            mosaic.save_params(args.param_file)

        if args.mp:
            mosaic.reset_tiles()

    if args.build_out and isinstance(mosaic, stitch2d.StructuredMosaic):
        print("Building out from placed tiles")
        mosaic.build_out(from_placed=True)

    if args.smooth:
        print("Smoothing seams between tiles")
        mosaic.smooth_seams()

    if args.output:
        print(f"Saving mosaic to {args.output}")
        mosaic.save(args.output)
    else:
        print("Showing mosaic")
        mosaic.show()


def main(args=None):
    """Runs the command to stitch a mosaic"""

    parser = ParserWithError(
        description=("Stitches tiles in a regular grid into a mosaic")
    )
    parser.set_defaults(func=_mosaic_callback)

    parser.add_argument(
        dest="path",
        nargs="+",
        metavar="path",
        type=str,
        help=("either a list of images or a directory" " containing images to stitch"),
    )

    parser.add_argument(
        "-backend",
        dest="backend",
        type=str,
        choices=["opencv", "skimage"],
        default="opencv",
        help="specifies backend to use",
    )

    parser.add_argument(
        "-dim",
        dest="dim",
        type=int,
        help=(
            "number of items in the direction being traversed first, that"
            " is, the number of columns (if horizontal) or number of rows"
            " (if vertical)"
        ),
    )

    parser.add_argument(
        "-origin",
        dest="origin",
        type=str,
        choices=["ul", "ur", "ll", "lr"],
        default="ul",
        help="position of the first tile in the mosaic",
    )

    parser.add_argument(
        "-direction",
        dest="direction",
        type=str,
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="direction to traverse first when building the mosaic",
    )

    parser.add_argument(
        "-pattern",
        dest="pattern",
        type=str,
        choices=["raster", "snake"],
        default="raster",
        help="whether the grid is rastered or snaked",
    )

    parser.add_argument(
        "-mp",
        dest="mp",
        type=float,
        help="size in megapixels of working images used to align the mosaic",
    )

    parser.add_argument(
        "-limit",
        dest="limit",
        type=int,
        help="number of images to place when building the mosaic",
    )

    parser.add_argument(
        "-param_file", dest="param_file", type=str, help="path to a parameter file"
    )

    parser.add_argument(
        "--build_out",
        action="store_true",
        help=(
            "whether to include tiles that could not be placed using feature"
            " matching. Uses an average offset based on placed tiles."
        ),
    )

    parser.add_argument(
        "--smooth", action="store_true", help=("whether to smooth seams between tiles")
    )

    parser.add_argument(
        "-output",
        dest="output",
        type=str,
        help=(
            "the path to which to save the finished mosaic. If omitted,"
            " shows the mosaic instead"
        ),
    )

    if args is None:
        args = sys.argv[1:]

    args = parser.parse_args(args)
    args.func(args)


if __name__ == "main":
    main()
