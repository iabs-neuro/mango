import sys


from manager import Manager


if __name__ == '__main__':
    task = sys.argv[1] if len(sys.argv) > 1 else 'densenet_out'
    man = Manager('densenet_out')
    man.run()
