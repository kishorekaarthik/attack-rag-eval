import pytest

from copilot.sysmem import LowMemoryError, free_memory_gb, require_memory


def test_free_memory_is_a_positive_number_on_this_machine():
    free = free_memory_gb()
    assert free is None or free > 0


def test_require_memory_passes_when_enough_is_free():
    assert require_memory(1.5, free=lambda: 4.0) == 4.0


def test_require_memory_stops_when_below_the_floor():
    with pytest.raises(LowMemoryError, match=r"1\.2 GB free.*1\.5 GB"):
        require_memory(1.5, free=lambda: 1.2)


def test_require_memory_does_nothing_where_it_cannot_measure():
    assert require_memory(1.5, free=lambda: None) is None
