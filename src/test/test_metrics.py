import unittest
# import os
# import sys
#
# import pandas as pd
# import copy
# import pandas.testing as pdt


class MetricsTester(unittest.TestCase):

    # def test_weighted_variance(self):
    #     file_path = os.path.dirname(__file__)
    #     sample_location = os.path.join(file_path, 'test_data', 'r_out_weighted_variance.csv')
    #     reference_df = pd.read_csv(sample_location)
    #
    #     vdf = ViperFrame(copy.deepcopy(reference_df))
    #
    #     vdf.weighted_variance(output_col='unit_test')
    #
    #     reference = reference_df[['Variable', 'r_out_100']].drop_duplicates(). \
    #         sort_values(by=['Variable'])
    #
    #     new = vdf.as_pandas[['Variable', 'unit_test']].drop_duplicates(). \
    #         sort_values(by=['Variable'])
    #
    #     for ref, n in zip(reference.itertuples(), new.itertuples()):
    #         if ref.Variable != n.Variable:
    #             raise ValueError('Columns mismatch between ref and new')
    #         self.assertAlmostEqual(ref.r_out_100, n.unit_test, delta=0.001 * ref.r_out_100)
    #
    #
    # def test_profile_segment(self):
    #     df = pd.DataFrame({
    #         'person_id': [i for i in range(0, 11)],
    #         'group_id': [1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4],
    #         'cookies': [5, 6, 6, 7, 2, 5, 2, 3, 5, 1, 3],
    #         'y': [6, 2, 6, 7, 1, 7, 3, 4, 6, 2, 623]
    #     }
    #     )
    #
    #     reference_dt_no_dv = {'Variable': ['group_id', 'group_id', 'group_id', 'group_id'],
    #                           'Category': [1, 2, 3, 4],
    #                           'Count': [2, 4, 3, 2],
    #                           'Percentage': [18.181818181818183,
    #                                          36.36363636363637,
    #                                          27.27272727272727,
    #                                          18.181818181818183]}
    #
    #     reference_dt_with_dv = {'Variable': ['group_id', 'group_id', 'group_id', 'group_id'],
    #                             'Category': [1, 2, 3, 4],
    #                             'Count': [2, 4, 3, 2],
    #                             'Percentage': [18.181818181818183,
    #                                            36.36363636363637,
    #                                            27.27272727272727,
    #                                            18.181818181818183],
    #                             'index': [6.5967016491754125,
    #                                       8.65817091454273,
    #                                       7.146426786606696,
    #                                       515.3673163418291]}
    #
    #     vdf = ViperFrame(df)
    #
    #     pdt.assert_frame_equal(vdf.profile_segment(subset_binned_name='group_id'), pd.DataFrame(reference_dt_no_dv))
    #     pdt.assert_frame_equal(vdf.profile_segment(subset_binned_name='group_id',
    #                                                dependent_variable='y'), pd.DataFrame(reference_dt_with_dv))
    #
    #

    def test_dummy(self):
        assert 1 == 1
