import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


@pytest.fixture(scope="module")
def minimal_2d_dataset(tmp_path_factory):

    path = tmp_path_factory.mktemp("minimal_2d")

    # A simple example we can produce. This is 2D and has two tiles.

    wcs = WCS(naxis=2)

    index_hdu = fits.PrimaryHDU()
    index_hdu.data = np.zeros((0, 0)).astype(np.float32)
    index_hdu.header.update(wcs.to_header())
    index_hdu.header["ZFVER"] = 1
    index_hdu.header["ZFNAXIS1"] = 20
    index_hdu.header["ZFNAXIS2"] = 30
    index_hdu.header["ZFTILE1"] = 20
    index_hdu.header["ZFTILE2"] = 15
    index_hdu.header["ZFDIR"] = "tiles"

    index_hdu.writeto(path / "index.fits")

    # TODO write tiles

    return path
