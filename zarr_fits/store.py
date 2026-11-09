from zarr.storage import WrapperStore

__all__ = ['ZarrFITSStore']


class ZarrFITSStore(WrapperStore):

    def __repr__(self):
        return f"ZarrFITSStore({self._store!r})"
