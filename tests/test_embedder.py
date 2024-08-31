from unittest import TestCase

from app.embeddings import Embedder


class TestEmbedder(TestCase):

    def test_basic(self):
        m = Embedder()
        texts = ["asdasd", "xzczxczx", "xzczxcwewe"]
        m.embed(texts)
        indices, angles = m.top_n("xzczxc", 1)
        print(indices, angles)
        self.assertTrue(len(indices) == 1 and len(angles) == 1)

        indices, angles = m.top_n("xzczxc", 2)
        self.assertTrue(len(indices) == 2 and len(angles) == 2)