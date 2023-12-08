from multiprocessing import Lock

MANAGER_TOKEN = '5402c9b6-2e4f-4fbd-a638-394b97bfd8dc'


class ConcurrentDict:
    def __init__(self):
        self.lock = Lock()
        self.d = {}

    def __contains__(self, key):
        with self.lock:
            return key in self.d

    def __getitem__(self, key):
        with self.lock:
            return self.d[key]

    def __setitem__(self, key, value):
        with self.lock:
            self.d[key] = value

    def __delitem__(self, key):
        with self.lock:
            del self.d[key]

# def connect_to_manager(manager):
#     while True:
#         try:
#             manager.connect()
#             print("Connexion établie !")
#             return
#         except ConnectionError:
#             print("Connexion perdue. Tentative de reconnexion dans 5 secondes...")
#             sleep(5)