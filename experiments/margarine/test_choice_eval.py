import math
import unittest
from choice_eval import choices


def row(pair, rate, teacher, student):
    return dict(pair=pair, codec=pair, bpp=rate,
                scores={"teacher": {"p3": teacher}, "box3": {"p3": student}})


class ChoiceTests(unittest.TestCase):
    def test_budget_changes_eligibility_and_uses_teacher_regret(self):
        rows = [row("a", 1, 3, 2), row("b", 2, 2, 3), row("c", 3, 1, 1)]
        result = list(choices(rows, "box3", "p3"))
        self.assertEqual([r["relative_regret"] for r in result], [0, 0.5, 0])
        self.assertEqual([r["eligible"] for r in result], [1, 2, 3])

    def test_fixed_teacher_pooling_is_explicit_and_used_for_selection_and_regret(self):
        rows = [row("a", 1, 1, 1), row("b", 1, 2, 2)]
        rows[0]["scores"]["teacher"]["max"] = 4
        rows[1]["scores"]["teacher"]["max"] = 2
        matching = list(choices(rows, "box3", "p3"))[0]
        fixed = list(choices(rows, "box3", "p3", teacher_norm="max"))[0]
        self.assertEqual(matching["teacher_pair"], "a")
        self.assertEqual(fixed["teacher_pair"], "b")
        self.assertEqual(fixed["candidate_pair"], "a")
        self.assertEqual(fixed["relative_regret"], 1)

    def test_ties_do_not_select_using_teacher_scores(self):
        result = list(choices([row("b", 1, 1, 2), row("a", 1, 3, 2)], "box3", "p3"))[0]
        self.assertEqual(result["candidate_pair"], "a")
        self.assertEqual(result["relative_regret"], 2)

    def test_zero_optimum_is_explicit_and_invalid_input_fails(self):
        result = list(choices([row("a", 1, 0, 2), row("b", 1, 1, 1)], "box3", "p3"))[0]
        self.assertTrue(math.isinf(result["relative_regret"]))
        with self.assertRaises(ValueError):
            list(choices([row("a", float("nan"), 1, 1)], "box3", "p3"))

    def test_human_harm_uses_declared_label_orientation(self):
        rows = [row("a", 1, 1, 2), row("b", 1, 2, 1)]
        for r, target in zip(rows, [70, 65]):
            r.update(target=target, direction="quality")
        self.assertEqual(list(choices(rows, "box3", "p3", True))[0]["human_quality_loss"], 5)
        for r in rows:
            r["direction"] = "distortion"
        self.assertEqual(list(choices(rows, "box3", "p3", True))[0]["human_quality_loss"], -5)
        rows[0]["direction"] = "quality"
        with self.assertRaises(ValueError):
            list(choices(rows, "box3", "p3", True))
        with self.assertRaises(KeyError):
            list(choices([row("a", 1, 1, 1)], "box3", "p3", True))


if __name__ == "__main__":
    unittest.main()
