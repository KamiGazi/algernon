def _flatten(data):
    if isinstance(data, (int, float)):
        return [data]
    flat_list = []
    for item in data:
        flat_list.extend(_flatten(item))
    return flat_list

def _get_shape(data):
    if isinstance(data , (int, float)):
        return ()
    
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Tensor data must be a number or a (nested) list/tuple of numbers, but found {type(data).__name__}: {data!r}")
         
    if len(data) == 0:
        return (0,)
    
    previous_shape = _get_shape(data[0])
    for item in data:
        current_shape = _get_shape(item)
        if current_shape != previous_shape:
            raise ValueError("Inconsistent shape found in tensor data")
    return (len(data),) + current_shape
        
class Tensor:
    def __init__(self, data):
        self.data = data
        self.shape = _get_shape(data)
        self.storage = _flatten(data)

    def __getitem__(self, index):
        return self.data[index]