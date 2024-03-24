import unittest
import gpucachesim.stats.metrics as metric_funcs


class TestErrorMeasures(unittest.TestCase):
    def test_mape_over_under_estimation_symmetric(self):
        # this symmetry is favourable because for fixed true value,
        # we treat accelsim and ours equally.
        self.assertEqual(
            metric_funcs.mape(true_values=[100], values=[90]),
            metric_funcs.mape(true_values=[100], values=[110]),
        )
        # we dont really need the symmetry here as we dont train a model but only
        # compare two models
        self.assertNotEqual(
            metric_funcs.mape(true_values=[100], values=[150]),
            metric_funcs.mape(true_values=[150], values=[100]),
        )

        # hence: mape > smape for our case
        # see also: https://robjhyndman.com/hyndsight/smape/

    def test_smape_over_under_estimation_symmetric(self):
        self.assertNotEqual(
            metric_funcs.smape(true_values=[100], values=[90]),
            metric_funcs.smape(true_values=[100], values=[110]),
        )
        self.assertEqual(
            metric_funcs.smape(true_values=[100], values=[150]),
            metric_funcs.smape(true_values=[150], values=[100]),
        )

    def test_mase_over_under_estimation_symmetric(self):
        self.assertEqual(
            metric_funcs.mase(true_values=[100], values=[90]),
            metric_funcs.mase(true_values=[100], values=[110]),
        )
        self.assertEqual(
            metric_funcs.mase(true_values=[100], values=[150]),
            metric_funcs.mase(true_values=[150], values=[100]),
        )

    def test_rmspe_zero_values(self):
        self.assertEqual(
            metric_funcs.rmspe(true_values=[0], values=[10]),
            # metric_funcs.mase(true_values=[0], values=[20]),
            4.503599627370496e16,
        )

        self.assertEqual(
            metric_funcs.rmspe(true_values=[10e-9], values=[10e-8]),
            9.0,
            # metric_funcs.mase(true_values=[0], values=[20]),
        )


if __name__ == "__main__":
    unittest.main()
