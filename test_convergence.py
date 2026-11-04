"""Array/plot/AST checks only; no neural network training or forward pass."""
import ast
from pathlib import Path
import tempfile
import unittest
from utils.plot_convergence import convergence_rows, export_convergence


class ConvergenceTests(unittest.TestCase):
    def history(self):
        return [dict(epoch=i, train_loss=.6/i, Loss=.7/i, **{'Mean IoU': .7, 'Dice': .8},
                     lr=.0001, **{split+'_'+key: .1 for split in ('train', 'val')
                     for key in ('ce', 'dice', 'aux', 'weighted_ce', 'weighted_dice', 'weighted_aux')})
                for i in (1, 2)]

    def test_fields_and_legacy(self):
        h = self.history()
        del h[0]['train_ce']
        rows = convergence_rows(h)
        self.assertIsNone(rows[0]['train_ce'])
        self.assertEqual(rows[1]['train_total'], .3)
        self.assertEqual(rows[1]['val_miou'], .7)

    def test_invalid_history(self):
        for history in ([], self.history()[::-1]):
            with self.assertRaises(ValueError):
                convergence_rows(history)
        h = self.history()
        h[0]['Loss'] = float('nan')
        with self.assertRaises(ValueError):
            convergence_rows(h)

    def test_exports(self):
        with tempfile.TemporaryDirectory() as d:
            export_convergence(self.history(), d)
            self.assertTrue((Path(d)/'convergence.csv').is_file())
            for extension in ('png', 'svg', 'pdf'):
                self.assertEqual(len(list(Path(d).glob('*.'+extension))), 5)

    def test_loss_scalar_contract(self):
        # Check the default scalar return remains available and components detach.
        source = Path('model/losses.py').read_text()
        tree = ast.parse(source)
        forward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        self.assertEqual(forward.args.args[-1].arg, 'return_components')
        self.assertIs(forward.args.defaults[-1].value, False)
        self.assertIn('return total', source)
        self.assertIn('ce.detach()', source)


if __name__ == '__main__':
    unittest.main()
