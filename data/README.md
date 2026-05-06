# Local Datasets

Keep local datasets inside this folder so they are not committed to Git.

## CelebA-HQ

Use either layout:

```text
data/
└── celeba_hq_256/
    ├── image_00001.png
    ├── image_00002.png
    └── ...
```

or keep a zip at:

```text
data/celeba_hq_256.zip
```

The local training launcher can extract the zip automatically:

```bash
python3 scripts/train_celebahq_local.py --dataset_zip data/celeba_hq_256.zip
```

The image loader scans recursively, so nested folders inside `celeba_hq_256`
are fine.
