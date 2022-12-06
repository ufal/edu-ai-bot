import glob
import sys

def read_file(fd):
    tokens = [line.split()[0] for line in fd]
    return ' '.join(tokens).replace('<s>', '').replace('<g/>', '').replace('</s>', '').replace('  ?', '?')


if __name__ == '__main__':
    indir = sys.argv[1]
    out = sys.argv[2]
    with open(out, 'wt') as ofd:
        for subdir in glob.glob(f'{indir}/*'):
            with open(f'{subdir}/01question.vert', 'rt') as fd:
                quest = read_file(fd).strip()
                print(f'qawiki\t{quest}', file=ofd)
