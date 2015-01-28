import threading
import time


def _sum(somas):
    time.sleep(3)
    return somas.append(sum(range(11)))


class MyThread(threading.Thread):

    def run(self):
        print "started {}".format(self.name)
        super(MyThread, self).run()
        print "finished {}".format(self.name)


def teste():
    antes = threading.active_count()
    somas = []
    a = time.time()
    for i in range(1000000):
        thread = MyThread(target=_sum, name=str(i), args=(somas,))
        thread.start()
        # somas.append(thread.run(_sum, True))

    while threading.active_count() > antes:
        pass
    print somas
    print time.time() - a
