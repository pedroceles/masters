def single_runner(args):
    f_args = args[3:]
    instance, method_name, q = args[:3]
    method = getattr(instance, method_name)
    q.put(method(*f_args))
