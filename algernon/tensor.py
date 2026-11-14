def _flatten(data):
    if isinstance(data, (int, float)):
        return [data]
    flat_list = []
    for item in data:
        flat_list.extend(_flatten(item))
    return flat_list


def _get_shape(data):
    if isinstance(data, (int, float)):
        return ()

    if not isinstance(data, (list, tuple)):
        raise TypeError(
            f"Tensor data must be a number or a (nested) list/tuple of numbers, but found {type(data).__name__}: {data!r}"
        )

    if len(data) == 0:
        return (0,)

    first_shape = _get_shape(data[0])
    for item in data:
        current_shape = _get_shape(item)
        if current_shape != first_shape:
            raise ValueError(
                f"Inconsistent shape in tensor data: expected every element to have shape {first_shape}, but found {current_shape}"
            )
    return (len(data),) + first_shape


def _get_strides(shape):
    strides = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    strides.reverse()
    return tuple(strides)


def _unflatten(storage, shape, strides, offset):
    ''' Rebuild the list in the nested shape from the tensor '''
    if len(shape) == 0:
        return storage[offset]

    result = []
    for i in range(shape[0]):
        result.append(
            _unflatten(storage, shape[1:], strides[1:], offset + i * strides[0])
        )

    return result


class Tensor:
    def __init__(self, data):
        self.shape = _get_shape(data)
        self.storage = _flatten(data)
        self.strides = _get_strides(self.shape)
        self.offset = 0

    def tolist(self):
        return _unflatten(self.storage, self.shape, self.strides, self.offset)
