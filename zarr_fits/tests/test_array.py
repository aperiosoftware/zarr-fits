from math import prod

import numpy as np
import pytest
from astropy.io import fits
from astropy.io.fits import Header
from numpy.testing import assert_equal
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd

from ..array import create_array, open_array

# The following are tests that specifically check the zarr-FITS format

HEADER_SPECTRAL_CUBE = """
WCSAXES = 3
CRPIX1  = 10.
CRPIX2  = 20.
CRPIX3  = 30.
CDELT1  = -0.02
CDELT2  = 0.02
CDELT3  = 0.1
CUNIT1  = 'deg'
CUNIT2  = 'deg'
CUNIT3  = 'GHz'
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CTYPE3  = 'FREQ'
CRVAL1  = 50.
CRVAL2  = 70.
CRVAL3  = 230.
SPECSYS = 'LSRK'
""".lstrip()

EXPECTED_OUTPUT_HEADER = (
    """
SIMPLE  =                    T
BITPIX  =                  -32 / BITPIX of the array and its tiles
NAXIS   =                    3 / Number of array axes
NAXIS1  =                    0 / No data in this file, see ZFNAXIS1
NAXIS2  =                    0 / No data in this file, see ZFNAXIS2
NAXIS3  =                    0 / No data in this file, see ZFNAXIS3
ZFVER   =                    1 / Version of the zarr-FITS convention
ZFNAXIS1=                   20 / Array length along axis 1
ZFNAXIS2=                   40 / Array length along axis 2
ZFNAXIS3=                   30 / Array length along axis 3
ZFTILE1 =                   10 / Tile length along axis 1
ZFTILE2 =                   20 / Tile length along axis 2
ZFTILE3 =                   15 / Tile length along axis 3
ZFDIR   = 'tiles   '           / Directory holding the tiles
""".lstrip()
    + HEADER_SPECTRAL_CUBE
    + "END"
)


def assert_files_expected(directory, expected):
    actual = sorted(
        [
            f.relative_to(directory).as_posix()
            for f in directory.rglob("*")
            if f.is_file()
        ]
    )
    assert actual == expected


class TestZarrFITSStore:
    def _generate_spectral_cube_zarr(self, tmp_path, *, shape, chunks, **kwargs):

        data = np.arange(prod(shape)).reshape(shape)

        header = Header.fromstring(HEADER_SPECTRAL_CUBE, sep="\n")

        array = create_array(
            tmp_path, shape=shape, chunks=chunks, dtype="f4", header=header, **kwargs
        )

        array[:] = data

        header["NAXIS"] = len(shape)
        for i, n in enumerate(shape[::-1]):
            header[f"NAXIS{i + 1}"] = n

        return data, header

    def test_expected_files(self, tmp_path):

        # This test checks that all the files we expect to be there are indeed
        # there for the spectral cube WCS

        self._generate_spectral_cube_zarr(
            tmp_path,
            shape=(30, 40, 20),
            chunks=(15, 20, 10),
        )

        assert_files_expected(
            tmp_path,
            [
                "index.fits",
                "tiles/0/0/0.fits",
                "tiles/0/0/1.fits",
                "tiles/0/1/0.fits",
                "tiles/0/1/1.fits",
                "tiles/1/0/0.fits",
                "tiles/1/0/1.fits",
                "tiles/1/1/0.fits",
                "tiles/1/1/1.fits",
            ],
        )

    def test_simple_roundtrip(self, tmp_path):

        # This test checks that the data we read back in is identical to the
        # data used to generate the zarr-FITS data

        original_data, _ = self._generate_spectral_cube_zarr(
            tmp_path,
            shape=(30, 40, 20),
            chunks=(15, 20, 10),
        )

        # Make sure the data itself round-trips

        array = open_array(tmp_path)
        assert_equal(array[:], original_data)

        # Check the header in the index file matches the expected one

        header = fits.getheader(tmp_path / "index.fits")

        assert header == Header.fromstring(EXPECTED_OUTPUT_HEADER, sep="\n")

    @pytest.mark.parametrize("fill_value", [np.nan, -1])
    def test_fill_padding(self, tmp_path, fill_value):

        # This test checks that things work properly if the data shape is not
        # a multiple of the chunk size - in this case, the arrays get padded with
        # the fill value on disk

        original_data, _ = self._generate_spectral_cube_zarr(
            tmp_path,
            shape=(28, 34, 11),
            chunks=(15, 20, 10),
            fill_value=fill_value,
        )

        # Make sure the data itself round-trips

        array = open_array(tmp_path)
        assert_equal(array[:], original_data)

        assert array.shape == (28, 34, 11)

        # Check the incomplete corner tile on disk

        corner_tile = fits.getdata(tmp_path / "tiles" / "1" / "1" / "1.fits")

        if np.isnan(fill_value):
            assert not np.any(np.isnan(corner_tile[:13, :14, :1]))
            assert np.all(np.isnan(corner_tile[13:]))
            assert np.all(np.isnan(corner_tile[:, 14:]))
            assert np.all(np.isnan(corner_tile[:, :, 1:]))
        else:
            assert not np.any(corner_tile[:13, :14, :1] == fill_value)
            assert np.all(corner_tile[13:] == fill_value)
            assert np.all(corner_tile[:, 14:] == fill_value)
            assert np.all(corner_tile[:, :, 1:] == fill_value)

    def test_reproject(self, tmp_path):

        original_data, header = self._generate_spectral_cube_zarr(
            tmp_path,
            shape=(28, 34, 11),
            chunks=(15, 20, 10),
            fill_value=np.nan,
        )

        array, footprint = reproject_and_coadd(
            list(tmp_path.glob("**/*.fits")),
            header,
            reproject_function=reproject_interp,
            order=0,
        )

        assert_equal(array, original_data)
        assert_equal(footprint, 1)
