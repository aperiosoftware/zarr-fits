import pytest
from zarr.core.buffer import cpu, default_buffer_prototype
from zarr.storage import LocalStore
from zarr.testing.store import StoreTests

from ..store import ZarrFITSStore


class TestZarrFITSStore(StoreTests):
    store_cls = ZarrFITSStore
    buffer_cls = cpu.Buffer

    @pytest.fixture
    def store_kwargs(self, tmp_path):
        self.backing = LocalStore(str(tmp_path))
        return {"store": self.backing}

    @pytest.fixture
    def open_kwargs(self, store_kwargs):
        return {"store_cls": LocalStore, "root": str(self.backing.root)}

    # The tests call methods of ZarrFITSStore and check the results against
    # the storage it wraps, for example writing data with ZarrFITSStore.set
    # and checking that LocalStore.get returns the same data. The two following
    # helpers (self.set and self.get) are that independent check, so they use
    # the wrapped LocalStore directly.

    async def set(self, store, key, value):
        await self.backing.set(key, value)

    async def get(self, store, key):
        return await self.backing.get(key, prototype=default_buffer_prototype())

    def test_store_repr(self, store):
        assert (
            repr(store)
            == f"ZarrFITSStore(LocalStore('file://{self.backing.root.as_posix()}'))"
        )

    def test_store_supports_writes(self, store):
        assert store.supports_writes

    def test_store_supports_listing(self, store):
        assert store.supports_listing
