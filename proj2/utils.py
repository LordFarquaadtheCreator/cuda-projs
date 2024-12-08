def haversine(x1, y1, x2, y2):
    import math
    R = 6371000.0

    phi_1 = x1 * (math.pi / 180)
    phi_2 = x2 * (math.pi / 180)
    delta_phi = (x1 - x2) * (math.pi / 180)
    delta_lambda = (y1 - y2) * (math.pi / 180)

    a = pow(math.sin(delta_phi / 2.0), 2.0) + math.cos(phi_1) * math.cos(phi_2) * pow(math.sin(delta_lambda / 2.0), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = (R * c)
    kilometers = meters / 1000.0
    return kilometers

def timer(func): 
    from time import time 
    def wrap_func(*args, **kwargs): 
        t1 = time() 
        result = func(*args, **kwargs) 
        t2 = time() 
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s') 
        return result 
    return wrap_func 