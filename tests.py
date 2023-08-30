from json_np import Serializable, loads, dumps
import numpy as np
import unittest
import datetime

class C(Serializable):

    def __init__(self, a=7):
        self.a = a
        self.b = np.array([5, 6])

class D(object):

    def __init__(self, a=0):
        self.a = a
        self._b = 3

class E(object):

    def __init__(self, a):
        self.a = a
        self._b = 3


class TestSerializable(unittest.TestCase):

    def test_simple(self):
        a = Serializable()
        a.x = 1
        a.y = 'test'
        a.z = 3.14
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_datetime(self):
        a = Serializable()
        a.x = datetime.datetime(2015,1,3)
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_recursive(self):
        a = Serializable()
        a.x = Serializable()
        a.x.y = 'test'
        b = Serializable.from_json(a.to_json())
        self.assertEqual(a, b)

    def test_np(self):
        a = Serializable()
        a.x = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
        b = Serializable.from_json(a.to_json(pack_ndarray=True))
        self.assertEqual(np.sum(np.abs(a.x - b.x)), 0)

    def test_np_twice(self):
        a = Serializable()
        a.x = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
        b = Serializable.from_json(a.to_json(pack_ndarray=True))
        self.assertEqual(np.sum(np.abs(a.x - b.x)), 0)
        c = Serializable.from_json(b.to_json(pack_ndarray=True))
        self.assertEqual(np.sum(np.abs(a.x - c.x)), 0)

    def test_np_direct(self):
        a = np.array([[1,2,3],[4,5,6]], dtype=np.int32)
        s = Serializable.dumps(a, pack_ndarray=True)
        c = Serializable.from_json(s)
        self.assertEqual(np.sum(np.abs(a - c)), 0)

    def test_float(self):
        x = np.float16(3.5)
        y = Serializable.from_json(Serializable.dumps(x))
        self.assertAlmostEqual(y, x, 2)

    def test_set(self):
        s = set(['a', 'b', 'c'])
        x = Serializable.dumps(s)
        t = Serializable.loads(x)
        self.assertEqual(s, t)

    def test_multiple_dicts(self):
        d = dict(cane=4, gatto=4, uccello=2)
        d1 = Serializable.loads(Serializable.dumps(d))
        d2 = Serializable.loads(Serializable.dumps(d1))
        for k in d.keys():
            self.assertEqual(d.get(k), d2.get(k))
        for k in d2.keys():
            self.assertEqual(d.get(k), d2.get(k))

    def test_modifiable(self):
        a = np.zeros((10,10))
        b = loads(dumps(a))
        a[2:4, 5:6] = 1
        b[2:4, 5:6] = 1
        self.assertEqual(np.sum(np.abs(a - b)), 0)

    def test_bytes(self):
        a = b"owefijeowf"
        aa = loads(dumps(a))
        self.assertEqual(type(aa), bytes)
        self.assertEqual(a, aa)

    def test_custom_class(self):
        c = C(3)
        s = dumps(c)
        cc = loads(s)
        self.assertEqual(cc.a, 3)
        self.assertEqual(type(cc.b), np.ndarray)
        self.assertEqual(cc.__class__, C)

    def test_arbitrary_class(self):
        d = D(3)
        s = dumps(d)
        dd = loads(s)
        self.assertEqual(dd.a, 3)

    def test_class_with_required_args(self):
        e = E(5)
        ee = loads(dumps(e))
        self.assertTrue(isinstance(ee, Serializable))
        self.assertFalse(isinstance(ee, E))

if __name__ == '__main__':
    unittest.main()
