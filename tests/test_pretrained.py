import argparse
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from train_medical import resolve_initialization
from download_pretrained import download, check_hash


class PretrainedContracts(unittest.TestCase):
    def test_default_is_required_pretraining(self):
        args = argparse.Namespace(backbone_weights=None, from_scratch=False, resume=None)
        with patch('train_medical.Path.is_file', return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, 'download_pretrained'):
                resolve_initialization(args)
        self.assertTrue(args.backbone_weights.endswith('pretrained/resnet50.pth'))
        for scratch, resume in [(True, None), (False, 'last.pth')]:
            args = argparse.Namespace(backbone_weights=None, from_scratch=scratch, resume=resume)
            resolve_initialization(args)
            self.assertIsNone(args.backbone_weights)

    def test_download_hash_and_reuse_without_network(self):
        payload = b'weight fixture'; prefix = hashlib.sha256(payload).hexdigest()[:8]
        weights = type('Weights', (), {'IMAGENET1K_V2': type('V2', (), {'url': 'https://example.test/resnet50-' + prefix + '.pth'})})
        def transfer(url, filename, **kwargs): Path(filename).write_bytes(payload)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'resnet50.pth'
            with patch('download_pretrained.ResNet50_Weights', weights), patch('torch.hub.download_url_to_file', side_effect=transfer) as fetch:
                download(path); download(path)
                self.assertEqual(fetch.call_count, 1)
                self.assertFalse(path.with_name(path.name + '.part').exists())
                path.write_bytes(b'bad')
                with self.assertRaises(ValueError): download(path)
            with self.assertRaises(ValueError): check_hash(path, prefix)


if __name__ == '__main__': unittest.main()
