import json

from astropy.io import fits

from ..store import index_fits_to_zarr_json, zarr_json_to_index_fits


def deterministic_json(d):
    return json.dumps(d, sort_keys=True, indent=2)


EXPECTED_ZARR_JSON_MINIMAL_2D = """
{
  "chunk_grid": {
    "configuration": {
      "chunk_shape": [
        15,
        20
      ]
    },
    "name": "regular"
  },
  "chunk_key_encoding": {
    "configuration": {
      "directory": "tiles",
      "offset": [
        0,
        0
      ]
    },
    "name": "fits"
  },
  "codecs": [
    {
      "configuration": {
        "endian": "big"
      },
      "name": "bytes"
    },
    {
      "configuration": {
        "header": {
          "WCSAXES": 2,
          "CDELT1": 1.0,
          "CDELT2": 1.0,
          "CRPIX1": 0.0,
          "CRPIX2": 0.0,
          "CRVAL1": 0.0,
          "CRVAL2": 0.0,
          "LATPOLE": 90.0,
          "MJDREF": 0.0
        }
      },
      "name": "fits_header"
    }
  ],
  "data_type": "float32",
  "fill_value": "NaN",
  "node_type": "array",
  "shape": [
    30,
    20
  ],
  "zarr_format": 3
}
"""


class TestIndexJSONConversion:
    def test_index_to_json(self, minimal_2d_dataset):

        index_header = fits.getheader(minimal_2d_dataset / "index.fits")

        zarr_json = index_fits_to_zarr_json(index_header)

        assert (
            deterministic_json(json.loads(zarr_json)) == EXPECTED_ZARR_JSON_MINIMAL_2D
        )

    def test_json_to_index(self, minimal_2d_dataset):

        expected_header = fits.getheader(minimal_2d_dataset / "index.fits")

        json.loads(EXPECTED_ZARR_JSON_MINIMAL_2D.encode("ascii"))

        actual_header = zarr_json_to_index_fits(
            EXPECTED_ZARR_JSON_MINIMAL_2D.encode("ascii")
        )

        assert expected_header == actual_header
