import unittest
from app.verbalizer import analyze

class TestVerbalizer(unittest.TestCase):

    def test_base(self):
        predictions = ["cancel sub", "refund me"]
        labels = ["cancel subscription", "refund request"]
        new_predictions = analyze(predictions, labels)
        self.assertNotEqual(predictions, labels)
        self.assertEqual(labels, new_predictions)

