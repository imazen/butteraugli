import unittest
from disagreements import unique_choices


class DisagreementsTests(unittest.TestCase):
    def test_repeated_budgets_preserve_count_and_do_not_repeat_pair(self):
        def row(pair, rate, teacher, candidate, target):
            return dict(pair=pair, bpp=rate, codec=pair, target=target, direction='quality',
                        scores={'teacher': {'p3': teacher}, 'candidate': {'p3': candidate}})
        rows = [row('a', 1, 1, 3, 70), row('b', 2, 2, 1, 65), row('c', 3, 4, 4, 50)]
        result = unique_choices(rows, 'candidate', 'p3')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['budget_count'], 2)
        self.assertEqual(result[0]['human_quality_loss'], 5)
        self.assertEqual((result[0]['first_budget'], result[0]['last_budget']), (2, 3))


if __name__ == '__main__':
    unittest.main()
