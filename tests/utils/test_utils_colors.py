import pytest

from dgp.utils.colors import get_unique_colors


@pytest.mark.parametrize(
    "num_colors,in_bgr,cmap,expected_map", [
        (5, False, 'tab20', [
            (31, 119, 180),
            (174, 199, 232),
            (255, 127, 14),
            (255, 187, 120),
            (44, 160, 44),
        ]),
        (3, True, 'tab20', [(180, 119, 31), (232, 199, 174), (14, 127, 255)]),
        (1, True, 'tab20', [(180, 119, 31)]),
        (3, False, 'Paired', [(166, 206, 227), (31, 120, 180), (178, 223, 138)]),
    ]
)
def test_get_unique_colors(num_colors, in_bgr, cmap, expected_map):  # pylint: disable=missing-any-param-doc
    '''
    Uses parametrized testing to run multiple cases for colorbar retreival
    '''
    assert get_unique_colors(num_colors=num_colors, in_bgr=in_bgr, cmap=cmap) == expected_map
