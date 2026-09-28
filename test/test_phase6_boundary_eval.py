"""随机配方必须合法，且不会被影子判定误记为 GPU 证据。"""

from bench.boundary_eval import random_peer, shadow_changed
from pact.mutation import standard_recipes


def test_random_peer_is_reproducible_and_valid():
    source = next(item for item in standard_recipes() if item.case == "D")
    first = random_peer(source, 17)
    assert first.fingerprint == random_peer(source, 17).fingerprint
    assert tuple(name for name, _ in first.inputs) == ("X", "Y")
    assert all(spec.storage_size >= spec.numel for _, spec in first.inputs)
