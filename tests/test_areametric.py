import unittest
import areametric as am
import numpy as np
from scipy.stats import wasserstein_distance

# python3 -m unittest tests/test_areametric.py
# xattr -w com.dropbox.ignored 1 .venv/
# https://help.dropbox.com/files-folders/restore-delete/ignored-files

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

    def test_areame_different_size2(self):
        """
        Test on datasets of different sizes
        """
        xx = am.skinny()
        yy = am.puffy()
        result = am.areaMe(xx,yy)
        self.assertAlmostEqual(result[0], 1.26611, places=5)
        self.assertAlmostEqual(result[1], 1.59278, places=5)

    def test_areame_tabular_2d(self):
        """
        Test on tabular data sets
        """
        X,Y = am.example_2d()
        areas = am.areaMe(X,Y)
        reference_solution = [ 7.00875,  6.38500  , 14.73125,  8.12000   , 11.41875]
        for a,r in zip(areas,reference_solution):
            self.assertAlmostEqual(a, r, places=5)

    def test_areame_vs_scipy_2d(self):
        dim = (3,5)
        X = am.example_random_Nd(n=79,dim=dim)
        Y = am.example_random_Nd(n=79,dim=dim)
        areas = am.areaMe(X,Y)
        J=am.map_index_flat_to_array(dim)
        for i in range(np.prod(dim)):
            j,k = J[i]
            x, y = X[:,j,k], Y[:,j,k]
            a_scipy = wasserstein_distance(x,y)
            self.assertAlmostEqual(areas[j,k], a_scipy, places=5)

    def test_areame_vs_scipy_3d(self):
        dim = (3,7,4)
        X = am.example_random_Nd(n=179,dim=dim)
        Y = am.example_random_Nd(n=179,dim=dim)
        areas = am.areaMe(X,Y)
        J=am.map_index_flat_to_array(dim)
        for i in range(np.prod(dim)):
            j,k,l = J[i]
            x, y = X[:,j,k,l], Y[:,j,k,l]
            a_scipy = wasserstein_distance(x,y)
            self.assertAlmostEqual(areas[j,k,l], a_scipy, places=5)

    def test_areame_mixtures_1d(self):
        dim = (7,)
        X = am.example_random_Nd(n=179,dim=dim)
        Y = am.example_random_Nd(n=179,dim=dim)
        areas = am.areaMe(X,Y)
        XY_m = am.mixture((X,Y))
        area_mix = am.areame_mixture(XY_m)
        for a,a_mix in zip(areas,area_mix):
            self.assertAlmostEqual(a, a_mix, places=5)

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
