# Third-party material

`vendor/movinets/models.py`, `vendor/movinets/config.py`, and the original
`vendor/movinets/__init__.py` are from Lorenzo Atzeni's
[Atze00/MoViNet-pytorch](https://github.com/Atze00/MoViNet-pytorch), commit
`c2d1edf48fc6c5259707f9d833f22171b4f63493`. The upstream MIT license is reproduced
verbatim in `vendor/MOVINET_LICENSE.txt`. There are two local source changes:
the package initializer uses a relative import (`from .models import MoViNet`),
and `config.py` imports `CfgNode` directly from `yacs.config` instead of fvcore's
subclass. This configuration uses only the shared attribute-container/clone API;
the direct import avoids unnecessary fvcore/iopath build dependencies. Model
layers, architecture settings and weights are unchanged.

`assets/movinet_a2_stream.pt` is the same repository's converted Kinetics-600
A2 streaming state dictionary, originally named `modelA2_stream_statedict_v3`.
It is redistributed from that MIT-licensed repository without modification.
Its SHA-256 is
`a3128a6bced7c0942c0e34bbb848035a5cfcaa226ae22c0d1f753e3739e9927a`.

The original architecture and pretrained TensorFlow models are from the
[TensorFlow Model Garden MoViNet project](https://github.com/tensorflow/models/tree/master/official/projects/movinet),
under [Apache License 2.0](https://github.com/tensorflow/models/blob/master/LICENSE).
No TensorFlow implementation code is copied into this demo.

`assets/kinetics_600_labels.txt` is the unmodified label list referenced by the
official TensorFlow tutorial, from TensorFlow Models commit
`f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2`. Its SHA-256 is
`8dfca4202520ea8291ce0242ef33b006d0a0772b409663c090a2876bc1978dcf`.
The corresponding Apache 2.0 license is included as `assets/TENSORFLOW_LICENSE.txt`.

External runtime packages are installed through pip and retain their respective
licenses. No runtime package binaries are included in this ZIP.
