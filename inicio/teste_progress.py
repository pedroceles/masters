import time
import sys

def do_task():
    time.sleep(0.1)

def example_1(n):
    steps = n/10
    for i in range(n):
        do_task()
        if i%steps == 0:
            print '\r.',
            sys.stdout.flush()
    print ' Done!'

print 'Starting ',
sys.stdout.flush()
example_1(100)
