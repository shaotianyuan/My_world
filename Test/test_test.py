from unittest import TestCase


class Test(TestCase):
    def test_pr(self):
        self.fail()

a = Test()
a.test_pr(
    'a'
)
