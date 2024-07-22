import torch
import torch.nn as nn


def test_compute_output_size():
    from clinicadl.nn.utils import compute_output_size

    input_2d = torch.randn(3, 2, 100, 100)
    input_3d = torch.randn(3, 1, 100, 100, 100)
    indices_2d = torch.randint(0, 100, size=(3, 2, 100, 100))

    conv3d = nn.Conv3d(
        in_channels=1,
        out_channels=1,
        kernel_size=7,
        stride=2,
        padding=(1, 2, 3),
        dilation=3,
    )
    max_pool3d = nn.MaxPool3d(kernel_size=(9, 8, 7), stride=1, padding=3, dilation=2)
    conv_transpose2d = nn.ConvTranspose2d(
        in_channels=2,
        out_channels=1,
        kernel_size=7,
        stride=(4, 3),
        padding=0,
        dilation=(2, 1),
        output_padding=1,
    )
    max_unpool2d = nn.MaxUnpool2d(kernel_size=7, stride=(2, 1), padding=(1, 1))
    sequential = nn.Sequential(
        conv3d, nn.Dropout(p=0.5), nn.BatchNorm3d(num_features=1), max_pool3d
    )

    assert compute_output_size(input_3d.shape[1:], conv3d) == tuple(
        conv3d(input_3d).shape[1:]
    )
    assert compute_output_size(input_3d.shape[1:], max_pool3d) == tuple(
        max_pool3d(input_3d).shape[1:]
    )
    assert compute_output_size(input_2d.shape[1:], conv_transpose2d) == tuple(
        conv_transpose2d(input_2d).shape[1:]
    )
    assert compute_output_size(input_2d.shape[1:], max_unpool2d) == tuple(
        max_unpool2d(input_2d, indices_2d).shape[1:]
    )
    assert compute_output_size(tuple(input_3d.shape[1:]), sequential) == tuple(
        sequential(input_3d).shape[1:]
    )
