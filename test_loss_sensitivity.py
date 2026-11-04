"""Lightweight sweep checks: no torch import or neural network execution."""
import json
from pathlib import Path
import tempfile
import unittest
import loss_sensitivity as sweep


class SweepTests(unittest.TestCase):
    def test_unique_default(self):
        self.assertEqual(len(sweep.settings()), 9)
        self.assertEqual(sweep.settings().count((1., 0.2)), 1)

    def test_same_epoch_and_tie(self):
        h = [dict(epoch=1, **{'Mean IoU': .8, 'Dice': .85}),
             dict(epoch=2, **{'Mean IoU': .7, 'Dice': .99}),
             dict(epoch=3, **{'Mean IoU': .8, 'Dice': .9})]
        self.assertEqual(sweep.best_row(h, 3)['Dice'], .85)
        with self.assertRaises(ValueError):
            sweep.best_row(h, 4)

    def test_invalid_score(self):
        for value in [float('nan'), None, 2.]:
            with self.assertRaises(ValueError):
                sweep.best_row([{'epoch': 1, 'Mean IoU': value, 'Dice': .8}], 1)

    def test_collect_and_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(sweep.collect(root, {'epochs': 2}), [])
            self.assertFalse((root/'loss_sensitivity.png').exists())
            for d, a in sweep.settings():
                folder = root / sweep.run_name(d, a)
                folder.mkdir()
                sweep.write_json(folder/'completed.json', {'epochs': 2})
                sweep.write_json(folder/'history.json', [
                    {'epoch': 1, 'Mean IoU': .8, 'Dice': .9},
                    {'epoch': 2, 'Mean IoU': .7, 'Dice': .95}])
            rows = sweep.collect(root, {'epochs': 2})
            self.assertEqual(len(rows), 9)
            self.assertTrue(all(row['best_epoch'] == 1 for row in rows))
            for ext in ['png', 'pdf', 'svg']:
                self.assertGreater((root/f'loss_sensitivity.{ext}').stat().st_size, 100)


if __name__ == '__main__':
    unittest.main()
