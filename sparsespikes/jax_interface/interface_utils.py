import ctypes

def create_pycapsule(so_file, fn_name):
    my_functions = ctypes.CDLL(so_file)
    fn_prt = getattr(my_functions, fn_name)

    PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor)
    capsule = PyCapsule_New(fn_prt, b"xla._CUSTOM_CALL_TARGET", PyCapsule_Destructor(0))
    return capsule