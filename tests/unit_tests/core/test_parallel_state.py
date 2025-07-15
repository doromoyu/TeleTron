from unittest import TestCase
from unit_tests.test_utils import spawn
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,
format='%(asctime)s - %(levelname)s - %(message)s')

def success(rank, world_size, q):
    logging.info(f"hello rank{rank}")
    q.put("True")


def fail(rank, world_size, q):
    q.put("False")


class testMPTesting(TestCase):

    def testSuccess(self):
        size = 4
        q = spawn(size, success)

        cnt = 0
        while not q.empty():
            res = q.get()
            self.assertEqual(res, "True")
            cnt += 1 
        self.assertEqual(cnt, size)
    
    def testFail(self):
        size = 4
        q = spawn(size, fail)

        cnt = 0
        while not q.empty():
            res = q.get()
            self.assertEqual(res, "False")
            cnt += 1 
        self.assertEqual(cnt, size)
