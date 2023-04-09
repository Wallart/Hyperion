TEXT_SEPARATOR = b'----TEXT-END----\n'
CHUNK_SEPARATOR = b'----CHUNK-END----\n'


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
