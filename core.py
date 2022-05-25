import torch


def assert_small_error(error, th, message, input_data):
    cond = torch.all(error.abs() < th)
    if not cond:
        max_err = error.abs().max()
        print("condition not met with max error: {}".format(max_err))
        max_err_data = input_data[error.abs().argmax()]
        print("condition not met with max error on data: {}".format(max_err_data))
    assert cond, message
