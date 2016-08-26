from setuptools import setup

# Read long_description from file
try:
    long_description = open('README.rst', 'rb').read()
except:
    long_description = ('Please see https://github.com/adamancer/stitch2d.git'
                        ' for more information about Stitch2D.')

setup(name='stitch2d',
      version='0.31',
      description='Stitch a planar set of tiles into a mosaic',
      long_description=long_description,
      classifiers = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Multimedia :: Graphics'
      ],
      url='https://github.com/adamancer/stitch2d.git',
      author='adamancer',
      author_email='mansura@si.edu',
      license='MIT',
      packages=['stitch2d'],
      install_requires = [
          'numpy',
          'pillow',
          'pyglet'
      ],
      include_package_data=True,
      entry_points = {
          'console_scripts' : [
              'stitch2d = stitch2d.__main__:main'
          ]
      },
      zip_safe=False)
