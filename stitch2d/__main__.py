"""Command line tools for preparing and stitching tilesets"""

import argparse
import sys

import stitch2d




def main(args=None):


    class MyParser(argparse.ArgumentParser):


        def error(self, message):
            """Return help text on error with command

               From http://stackoverflow.com/questions/4042452"""
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)




    def _mosaic_callback(args):
        """Calls mosey function from mosaic.py

        Args:
            args['path'] (str): path to tiles
            args['create_jpeg']: specifies whether to create
                JPEG derivative
            args['manual']: force manual stitch
            args['scalar']: amount to scale images before feature
                matching. Does not affect tiles used in final mosaic.
            args['threshold']: threshold for Lowe test. Autostich
                only
            args['homography']: use homography alogithm. If not specified,
                uses a simple clustering algorithm.
            args['equalize_histogram']: use equalize histogram to
                increase contrast in source tiles
        """
        args = vars(args)
        path = args.pop('path')
        jpeg = args.pop('create_jpeg')
        opencv = not args.pop('manual')
        stitch2d.mosey(path, create_jpeg=jpeg, opencv=opencv, **args)




    def _organize_callback(args):
        """Calls organizer function from organize.py

        Args:
            args['source'] (str): path to folder containing unsorted tiles
            args['destination'] (str): path to folder to which to copy
                the sorted tiles
        """
        args = vars(args)
        stitch2d.organize(args['source'], args['destination'])




    def _select_callback(args):
        """Calls select function from selector.py

        Args:
            args['path'] (str): path to folder containing tiles

        """
        args = vars(args)
        selector = stitch2d.Selector(args['path'])
        params = selector.get_job_settings()
        selector.select(*params)




    def _composite_callback(args):
        """Calls composite function from composite.py

        All the args except path are color=element mappings.

        Args:
            args['path'] (str): path to folder containing the images to
                composite
        """
        args = vars(args)
        for color in stitch2d.COLORS:
            if args[color] is not None:
                break
        else:
            parser.error('Must specify at least one color')
        path = args.pop('path')
        label = args.pop('label')
        jpeg = args.pop('create_jpeg')
        for arg in args.keys():
            if args[arg] is None:
                del args[arg]
        del args['func']
        stitch2d.composite(path, label, jpeg, **args)




    def I(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError('{} must be between'
                                             ' 0 and 1'.format(x))
        return x




    if args is None:
        args = sys.argv[1:]

    parser = MyParser(
        description=('Provides access to stitching functions from Stitch2d')
    )
    subparsers = parser.add_subparsers(help='sub-command help')

    # OpenCV mosaic subcommand
    mosaic_parser = subparsers.add_parser(
        'mosaic',
        help='Stitch tiles into a mosaic')
    mosaic_parser.add_argument(
        '-label',
        dest='label',
        type=str,
        help='the name of the mosaic (typically the sample name)')
    mosaic_parser.add_argument(
        '-path',
        dest='path',
        type=str,
        help='the path to the mosaics directory')
    mosaic_parser.add_argument(
        '-numcols',
        dest='num_cols',
        type=int,
        help='number of columns in the mosaic')
    mosaic_parser.add_argument(
        '-matcher',
        dest='matcher',
        type=str,
        # FIXME: The flann matcher is unreliable in Ubuntu and OpenCV 3.1.0.
        # Removed for the time being, but should revisit sometime.
        #choices=['brute-force', 'flann'],
        choices=['brute-force'],
        default='brute-force',
        help='specifies algorithm to use for matching. Only the '
             ' brute-force algorithm is currently supported.')
    mosaic_parser.add_argument(
        '-scalar',
        dest='scalar',
        type=I,
        default=0.5,
        help=('amount to scale images before matching. Smaller'
              ' images are faster but less accurate.'))
    mosaic_parser.add_argument(
        '-threshold',
        dest='threshold',
        type=I,
        default=0.7,
        help=('threshold to use for ratio test. Lower values give'
              ' fewer but better matches.'))
    mosaic_parser.add_argument(
        '--homography',
        action='store_true',
        help=('specifies whether to use the OpenCV homography function'
              ' to identifiy high-quality matches. A simple clustering'
              ' algorithm is used if this parameter is not specified.'))
    mosaic_parser.add_argument(
        '--equalize_histogram',
        action='store_true',
        help=('specifies whether to equalize histograms. In general,'
              'this improves the quality of matches but takes longer.'))
    mosaic_parser.add_argument(
        '--create_jpeg',
        action='store_true',
        help='specifies whether to create a JPEG derivative of the'
             ' final mosaic')
    mosaic_parser.add_argument(
        '--manual',
        action='store_true',
        help='force manual matching. Otherwise, OpenCV will be used'
             ' if it is installed.')
    mosaic_parser.add_argument(
        '--skipped',
        action='store_true',
        help='the path to a text file containg the indexes of'
             ' skipped files generated by selector')
    # Snake keyword argument group
    group = mosaic_parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--raster',
        dest='snake',
        action='store_const',
        const=False,
        default=None,
        help=('specifies whether tiles are rastered. The snake '
              ' and raster arguments are mutually exclusive.'))
    group.add_argument(
        '--snake',
        dest='snake',
        action='store_const',
        const=True,
        default=None,
        help=('specifies whether tiles are snaked. The snake'
              ' and raster arguments are mutually exclusive.'))
    mosaic_parser.set_defaults(func=_mosaic_callback)

    # Organize subcommand
    organize_parser = subparsers.add_parser(
        'organize',
        help='Organize element maps into folders')
    organize_parser.add_argument(
        dest='source',
        type=str,
        nargs='?',
        help='the path to the directory containing the element maps')
    organize_parser.add_argument(
        dest='destination',
        type=str,
        nargs='?',
        help='the path to the directory in which to store the organized maps')
    organize_parser.set_defaults(func=_organize_callback)

    # Select subcommand
    select_parser = subparsers.add_parser(
        'select',
        help='Select tiles to exclude from an SEM map')
    select_parser.add_argument(
        '-path',
        dest='path',
        type=str,
        help='the path to the mosaics directory')
    select_parser.set_defaults(func=_select_callback)

    # Composite subcommand
    composite_parser = subparsers.add_parser(
        'composite',
        help='Create a composite of multiple SEM element maps')
    composite_parser.add_argument(
        '-path',
        dest='path',
        type=str,
        help='the path containing the images to composite')
    composite_parser.add_argument(
        '-label',
        dest='label',
        type=str,
        help='the name of the composite (typically the sample name)')
    composite_parser.add_argument(
        '--create_jpeg',
        action='store_const',
        const=True,
        default=False,
        help='specifies whether to create a JPEG 2000 derivative'
             ' the composite')
    for color in sorted(stitch2d.COLORS):
        composite_parser.add_argument(
            '-{}'.format(color),
            dest=color,
            type=str,
            help='the element to be colored {} in the composite'.format(color))
    composite_parser.set_defaults(func=_composite_callback)

    args = parser.parse_args(args)
    args.func(args)




if __name__ == 'main':
    main()
