class Domain:
    def __init__(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass
        
    def restart(self):
        pass

    def percept(self):
        return self.e

    def action(self,a):
        self.a = a

    def tick(self):
        pass