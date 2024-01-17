import chex
import haiku as hk
from typing import Mapping, Text, Tuple, Sequence, Union

Shape = Sequence[int]
MetricsDict = Mapping[Text, chex.Array]

DataType = Union[chex.Array, Mapping[str, "DataType"]]
DatasetDict = Mapping[Text, DataType]

LossFnOutput = Tuple[chex.Array, MetricsDict]

# outputs of regular networks
DoubleQOutputs = Tuple[chex.Array, chex.Array]
RecurrentOutput = Tuple[chex.Array, chex.Array]

# outputs of network constructors
HaikuFn = Union[hk.Transformed, hk.MultiTransformed]
NetworkFns = Sequence[HaikuFn]

# outputs of world model
ImagineOutput = Tuple[chex.Array, chex.Array]
ObserveOutput = Tuple[chex.Array, chex.Array]