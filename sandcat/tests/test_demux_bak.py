from __future__ import annotations

from sandcat.pipeline.demux_traj import _derive_checkpoint_path


# --- _derive_checkpoint_path ---

def test_derive_checkpoint_path_plain_nc_returns_none():
    assert _derive_checkpoint_path('swish_x_output.nc') is None


def test_derive_checkpoint_path_bak_derives_correctly():
    result = _derive_checkpoint_path('swish_x_output.nc.bak')
    assert result == 'swish_x_output_checkpoint.nc.bak'


def test_derive_checkpoint_path_bak_with_directory():
    result = _derive_checkpoint_path('/files/swish_x_output.nc.bak')
    assert result == '/files/swish_x_output_checkpoint.nc.bak'


def test_derive_checkpoint_path_other_extension_returns_none():
    assert _derive_checkpoint_path('output.h5') is None
