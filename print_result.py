import argparse
import cPickle
import os

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input') 
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise IOError("Error: file non-existent: %s" % args.input)

    with open(args.input, 'r') as f:
        count = 0
        while True:
            try:
                dic = cPickle.load(f)
                count += 1
                print "================ Result #%d ================" % count
                for key in dic:
                    print "{0}: {1}".format(key, dic[key])
                print '\n'
            except EOFError:
                break

if __name__ == "__main__":
    main()
