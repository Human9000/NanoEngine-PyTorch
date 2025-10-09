from qnn.quantizer import q_packed_bits,q_scale_k
import numpy as np


def test_4():
    a = np.ones((10, 10), dtype='int8') * -2
    a[:, -1] = 7
    b = q_packed_bits(a, 4)
    print(b)
    for i in b:
        # 取前4字节，取后4字节
        mask = (np.ones(1, dtype="uint8") << 4) - 1  # 00001111
        ai = np.frombuffer((i & mask << 4).tobytes(), dtype='int8') >> 4
        bi = np.frombuffer(((i & mask) << 4).tobytes(), dtype='int8') >> 4
        print(ai.dtype, mask.dtype)
        print(f"{i:08b} {ai.tobytes()[0]:08b}, {bi.tobytes()[0]:08b}, {ai}, {bi}")


def test_2():
    a = np.ones((10, 10), dtype='int8') * -2
    scale_k = 2**np.random.randint(-127, 128, (10, 1))

    a = q_scale_k(a, 8, scale_k)
    b = q_packed_bits(a, 2)
    print(b)
    for i in b:
        # 取前4字节，取后4字节
        mask = (np.ones(1, dtype="uint8") << 2) - 1  # 00001111
        ai = np.frombuffer((i & mask << 6).tobytes(), dtype='int8') >> 6
        bi = np.frombuffer(((i & mask << 4) << 2).tobytes(), dtype='int8') >> 6
        ci = np.frombuffer(((i & mask << 2) << 4).tobytes(), dtype='int8') >> 6
        di = np.frombuffer(((i & mask) << 6).tobytes(), dtype='int8') >> 6
        print(f"{i:08b}  {ai}, {bi}, {ci}, {di}")


if __name__ == '__main__':
    # test_4()
    test_2()
