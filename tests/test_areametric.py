import unittest
import areametric as am

class TestAreaMetric(unittest.TestCase):
    def test_areame_same(self):
        """
        Test that it can compute the area of two equal datasets
        """
        d1 = [1, 2, 3]
        d2 = [1, 2, 3]
        result = am.areaMe(d1,d2)
        self.assertEqual(result, 0)

    def test_areame_point(self):
        """
        Test that it can compute the area of two degenerate datasets
        """
        d1 = [3]
        d2 = [5]
        result = am.areaMe(d1,d2)
        self.assertEqual(result, 2)

    def test_areame_skinny(self):
        """
        Test on the skinny dataset
        """
        dd = am.skinny()
        d1 = [d[0] for d in dd]
        d2 = [d[1] for d in dd]
        result = am.areaMe(d1,d2)
        self.assertAlmostEqual(result, 0.51, places=3)

    def test_areame_skinny_reverse(self):
        """
        Test on the skinny dataset in reverse order
        """
        dd = am.skinny()
        d1 = [d[0] for d in dd]
        d2 = [d[1] for d in dd]
        result = am.areaMe(d2,d1)
        self.assertAlmostEqual(result, 0.51, places=3)

    def test_areame_puffy(self):
        """
        Test on the puffy dataset
        """
        dd = am.puffy()
        d1 = [d[0] for d in dd]
        d2 = [d[1] for d in dd]
        result = am.areaMe(d1,d2)
        self.assertAlmostEqual(result, 2.828, places=3)

    def test_areame_puffy_reverse(self):
        """
        Test on the puffy dataset in reverse order
        """
        dd = am.puffy()
        d1 = [d[0] for d in dd]
        d2 = [d[1] for d in dd]
        result = am.areaMe(d2,d1)
        self.assertAlmostEqual(result, 2.828, places=3)

    def test_areame_different_size1(self):
        """
        Test on datasets of different sizes
        """
        d1= [1]
        d2= [3,3,3]
        result = am.areaMe(d1,d2)
        self.assertEqual(result, 2)

    # def test_areame_different_size1(self):
    #     """
    #     Test on datasets of different sizes
    #     """
    #     d1= [1,2,3]
    #     d2= [3,3,3,5,6,7]
    #     result = am.areaMe(d1,d2)
    #     self.assertEqual(result, 2)

# class TestDataset(unittest.TestCase):
#     def test_dataset_length(self):
#         d = [1,2,3,4,5,6]
#         l = len(d)
#         dataset = am.Dataset(d)
#         self.assertEqual(l,len(dataset))
#     def test_dataset_add(self):
#         d1= [1,2,3,4,5,6]
#         d2= [7,8]
#         d = d1+d2
#         dataset = am.Dataset(d)
#         dataset1 = am.Dataset(d1)
#         dataset2 = am.Dataset(d2)
#         dataset12 = dataset1+dataset2
#         self.assertEqual(dataset12,dataset)

if __name__ == '__main__':
    unittest.main()
