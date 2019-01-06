import unittest
import grl

class StorageTestCase(unittest.TestCase):

    def test_non_persist_storage(self):
        a = grl.Storage(dimensions=3, persist=False)
        a[1][2][3] = 4
        self.assertEqual(a[1][2][3], 4)
    
    def test_non_persist_access(self):
        a = grl.Storage(dimensions=2, persist=False)
        self.assertNotEqual(a[1][2], a[1][2])

    def test_non_persist_max(self):
        a = grl.Storage(dimensions=3, persist=False, default_arguments=range(4))
        a[1][2][3]
        a[1][2][1] = 5
        a[1][2][2]
        self.assertEqual(max(a[1][2]), (5, 1))

    def test_non_persist_min(self):
        a = grl.Storage(dimensions=3, persist=False, default_arguments=range(4))
        a[1][2][3]
        a[1][2][1] = 0
        a[1][2][2]
        self.assertEqual(min(a[1][2]), (0, 1))

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
        a = grl.Storage(dimensions=2, persist=True)
        self.assertEqual(a[1][2], a[1][2])

    def test_non_persist_max(self):
        a = grl.Storage(dimensions=3, persist=True, default_arguments=range(4))
        a[1][2][3]
        a[1][2][1] = 5
        a[1][2][2]
        self.assertEqual(max(a[1][2]), (5, 1))

    def test_persist_min(self):
        a = grl.Storage(dimensions=3, persist=True, default_arguments=range(4))
        a[1][2][3]
        a[1][2][1] = 0
        a[1][2][2]
        self.assertEqual(min(a[1][2]), (0, 1))

    def test_persist_len(self):
        a = grl.Storage(dimensions=3, persist=True)
        a[1][2][3]
        a[1][2][1] = 5
        a[1][2][2]
        self.assertEqual(len(a[1][2]), 3)