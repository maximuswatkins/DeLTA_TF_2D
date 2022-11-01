import pytest

import delta


def nb_params_unet(filters: [int]) -> int:
    nb_params = 0
    for filters_a, filters_b in zip([1] + filters, filters):
        nb_params += (1 + 3**2 * filters_a) * filters_b
        nb_params += (1 + 3**2 * filters_b) * filters_b
    for filters_a, filters_b in zip(filters[::-1], filters[::-1][1:]):
        nb_params += (1 + 2**2 * filters_a) * filters_b
        nb_params += (1 + 3**2 * filters_a) * filters_b
        nb_params += (1 + 3**2 * filters_b) * filters_b
    nb_params += 1 + filters[0]
    return nb_params


@pytest.mark.parametrize(
    "levels, nb_params",
    [
        (1, nb_params_unet([64])),
        (2, nb_params_unet([64, 128])),
        (3, nb_params_unet([64, 128, 256])),
        (4, nb_params_unet([64, 128, 256, 512])),
        (5, nb_params_unet([64, 128, 256, 512, 1024])),
    ],
)
def test_nb_params_unet(levels: int, nb_params: int):
    model = delta.model.unet(levels=levels)
    assert model.count_params() == nb_params
