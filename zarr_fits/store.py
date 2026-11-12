import io
import json
from typing import Any

import numpy as np
from astropy.io import fits
from zarr.abc.buffer import Buffer
from zarr.abc.store import ByteRequest
from zarr.core.buffer import BufferPrototype
from zarr.storage import WrapperStore

__all__ = ["ZarrFITSStore"]


class ZarrFITSStore(WrapperStore):
    def __repr__(self):
        return f"ZarrFITSStore({self._store!r})"

    async def get(
        self,
        key: str,
        prototype: BufferPrototype,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        if key == "zarr.json":
            index_fits = await self._store.get("index.fits", prototype, byte_range)
            if index_fits is None:
                raise FileNotFoundError(
                    "No index.fits key found, not a valid fits-zarr"
                )
            hdul = fits.open(io.BytesIO(index_fits.to_bytes()))
            json_bytes = index_fits_to_zarr_json(hdul[0].header)
            return prototype.buffer.from_bytes(json_bytes)
        return await self._store.get(key, prototype, byte_range)

    async def set(self, key: str, value: Buffer) -> None:
        if key == "zarr.json":
            hdr = zarr_json_to_index_fits(value.to_bytes())
            bytesio = io.BytesIO()
            fits.writeto(bytesio, data=np.zeros([0] * hdr["NAXIS"]), header=hdr)
            value = bytesio.read()
        await self._store.set(key, value)


def _get_keys_sorted(key_pattern: str, hdr: fits.Header) -> dict[str, Any]:
    head = hdr.get(key_pattern)
    if not head:
        return {}
    return dict(sorted(dict(head).items()))


def index_fits_to_zarr_json(hdr: fits.Header) -> bytes:
    # FITS order
    naxis = hdr.get("NAXIS")
    zfnaxisn: tuple[int] = tuple(_get_keys_sorted("ZFNAXIS?", hdr).values())
    if not isinstance(naxis, int) and len(zfnaxisn) != naxis:
        raise ValueError("Incorrect type or inconsistent ZFNAXIS keys with NAXIS")
    chunk_shape: tuple[int] = tuple(_get_keys_sorted("ZFTILE?", hdr).values())
    if zftoffn := _get_keys_sorted("ZFTOFF?", hdr):
        zftoffn = tuple(zftoffn)
    else:
        zftoffn = tuple([0] * hdr.get("NAXIS"))
    zarr_json = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": zfnaxisn[::-1],
        "data_type": hdr.get("BITPIX"),
        "chunk_grid": {
            "name": "regular",
            "configuration": {
                "chunk_shape": chunk_shape[::-1],
            },
        },
        "chunk_key_encoding": {
            "configuration": {
                "directory": hdr.get("ZFDIR"),
                "offset": zftoffn[::-1],
            },
        },
        "fill_value": hdr.get("ZFFILL"),
        "codecs": [
            {
                "name": "fits_header",
                "configuration": dict(hdr),
            },
            {
                "name": "bytes",
                "configuration": {
                    "endian": "big",
                },
            },
        ],
    }
    return json.dumps(zarr_json).encode("utf-8")


def zarr_json_to_index_fits(zarr_bytes: bytes) -> fits.Header:
    zarr: dict[str, Any] = json.loads(zarr_bytes)
    zfnaxisn = zarr["shape"]
    naxis = len(zfnaxisn)
    bitpix = zarr["data_type"]
    chunk_shape = zarr["chunk_grid"]["configuration"]["chunk_shape"]
    zfdir = zarr["chunk_key_encoding"]["configuration"]["directory"]
    zftoffn = zarr["chunk_key_encoding"]["configuration"]["offset"]
    header = zarr["codecs"][0]["configuration"]

    hdr = fits.Header(header)
    hdr["NAXIS"] = naxis
    for i, n in zip(range(naxis), range(naxis + 1, 1, -1)):
        hdr[f"NAXIS{n}"] = 0
        hdr[f"ZFNAXIS{n}"] = zfnaxisn[i]
        hdr["BITPIX"] = bitpix
        hdr[f"ZFTILE{n}"] = chunk_shape[i]
        hdr[f"ZFTOFF{n}"] = zftoffn[i]
        hdr["ZFDIR"] = zfdir

    return hdr
