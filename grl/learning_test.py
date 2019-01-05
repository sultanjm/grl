import unittest
import grl

class StorageTestCase(unittest.TestCase):

    def test_store_and_non_persist(self):
        a = grl.Storage(dimensions=3, persist=False)
        a[1][2][3] = 4
        self.assertEqual(a[1][2][3], 4)

# a = grl.learning.Storage(dimensions=3, persist=False)
# a[1][2][3]=4 # {1: {2: {3: 4}}}
# a[0][2][3]
# #a[1][2]=4 # {1: {2: 4}}

# print(a)
# print(a[3][1][2] != a[3][1][2])
# print(a[2][1][1] != a[2][1][1])
# a[1][3][4] = 1.5
# print(a[1][3][4] == 1.5)
# print(a[1][3].max == 1.5)
# print(a[1][3].argmax == 4)
# print(len(a) == 1)

# b = grl.learning.Storage(persist=True)
# print(b[2][(1,2)] == b[2][(1,2)])
# b[1][3] = 2
# b[1][4] = 4
# print(b[1][3] == 2)
# print(b[1].expectation() == 3.0)
# print(b[1].expectation({3: 1.0, 4:0.0}) == 2.0)
# print(b[1].max() == 4)
# print(b[1].argmax() == 4)
# print(len(b) == 2)
# print(b[5].argmax())
# print(b[7].argmin())

