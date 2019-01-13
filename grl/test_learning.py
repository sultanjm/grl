import unittest
import grl

class StorageTestCase(unittest.TestCase):

    def test_non_persist_storage(self):
        a = grl.Storage(dimensions=3, persist=False)
        a[1][2][3] = 4
        self.assertEqual(a[1][2][3], 4)
    
    def test_non_persist_access(self):
        a = grl.Storage(dimensions=2, persist=False, default=(0,1))
        self.assertNotEqual(a[1][2], a[1][2])

    def test_non_persist_max(self):
        a = grl.Storage(dimensions=3, persist=False, leaf_keys=range(4), default=(0,1))
        a[1][2][3]
        a[1][2][1] = 5
        a[1][2][2]
        self.assertEqual(a[1][2].max(), 5)
        self.assertEqual(a[1][2].argmax(), 1)

    def test_non_persist_min(self):
        a = grl.Storage(dimensions=3, persist=False, leaf_keys=range(4), default=(0,1))
        a[1][2][3]
        a[1][2][1] = 0
        a[1][2][2]
        self.assertEqual(a[1][2].min(), 0)
        self.assertEqual(a[1][2].argmin(), 1)

    def test_non_persist_len(self):
        a = grl.Storage(dimensions=3, persist=False)
        a[1][2][3]
        a[1][2][1] = 5
        a[1][2][2]
        self.assertEqual(len(a[1][2]), 1)

    def test_persist_storage(self):
        a = grl.Storage(dimensions=3, persist=True)
        a[1][2][3] = 4
        self.assertEqual(a[1][2][3], 4)
    
    def test_persist_access(self):
        a = grl.Storage(dimensions=2, persist=True, default=(0,1))
        self.assertEqual(a[1][2], a[1][2])

    def test_persist_max(self):
        a = grl.Storage(dimensions=3, persist=True, leaf_keys=range(4), default=(0,1))
        a[1][2][3]
        a[1][2][1] = 5
        a[1][2][2]
        self.assertEqual(a[1][2].max(), 5)
        self.assertEqual(a[1][2].argmax(), 1)

    def test_persist_min(self):
        a = grl.Storage(dimensions=3, persist=True, leaf_keys=range(4), default=(0,1))
        a[1][2][3]
        a[1][2][1] = -1
        a[1][2][2]
        self.assertEqual(a[1][2].min(), -1)
        self.assertEqual(a[1][2].argmin(), 1)

    def test_persist_len(self):
        a = grl.Storage(dimensions=3, persist=True, default=(0,1))
        a[1][2][3]
        a[1][2][1] = 5
        a[1][2][2]
        self.assertEqual(len(a[1][2]), 3)