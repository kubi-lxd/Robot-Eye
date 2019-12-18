class MyException(Exception):
    def __init__(self, code, message, args):
        self.args = args
        self.message = message
        self.code = code
