from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .pvrcnn_head_mt import PVRCNNHeadMT
from .second_head import SECONDHead
from .second_head_mt import SECONDHeadMT
from .roi_head_template import RoIHeadTemplate
from .pillar_head_mt import PillarHeadMT

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'SECONDHead': SECONDHead,
    'PointRCNNHead': PointRCNNHead,
    'SECONDHeadMT': SECONDHeadMT,
    'PVRCNNHeadMT': PVRCNNHeadMT,
    'PillarHeadMT': PillarHeadMT,
}
