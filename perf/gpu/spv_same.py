# spv_same.py <a.spv> <b.spv>: whether two SPIR-V modules are the same code up to the numbering of their IDs - the
# same instructions (opcode, word count) in the same order, and the words where they differ a consistent one-to-one
# mapping (an ID of a always the same ID of b and back) - e.g. intra_rec's first module before and after a change
# that only its second module (CCP_LFNST_MODULE) should see: compile both with glslang's -O (GpuLib's CMakeLists.txt).
# Prints the verdict and the first inconsistency.
import struct, sys


def words(path):
    data = open(path, 'rb').read()
    return struct.unpack('<%dI' % (len(data) // 4), data)


def main():
    a, b = words(sys.argv[1]), words(sys.argv[2])
    if len(a) != len(b):
        print('different lengths: %d / %d words' % (len(a), len(b)))
        return 1
    forward, backward = {}, {}
    pos, differing = 5, 0   # (after the header: magic, version, generator, ID bound, schema)
    while pos < len(a):
        count, opcode = a[pos] >> 16, a[pos] & 0xffff
        if a[pos] != b[pos] or count == 0:
            print('instruction %d differs: opcode %d / %d' % (pos, opcode, b[pos] & 0xffff))
            return 1
        for i in range(pos + 1, pos + count):
            if a[i] != b[i]:
                differing += 1
                if forward.setdefault(a[i], b[i]) != b[i] or backward.setdefault(b[i], a[i]) != a[i]:
                    print('word %d of opcode %d: %d -> %d, inconsistent with an earlier mapping'
                          % (i, opcode, a[i], b[i]))
                    return 1
        pos += count
    print('same code up to ID numbering (%d words differ, %d IDs renamed)' % (differing, len(forward)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
