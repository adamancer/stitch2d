import glob
import os

path = r'C:\MinSci\Workflows\Mosaics\8362_Al'
n = 19  # first number in grid
m = 22  # second number in grid
i = 0
while i <= n:
    j = 0
    while j <= m:
        fn = '8362[Grid@{} {}]_Counts_Al_K_map.tif'.format(i, j)
        fp = os.path.join(path, fn)
        try:
            open(fp, 'rb')
        except:
            print '{} does not exist!'.format(os.path.basename(fp))
        j += 1
    i += 1
