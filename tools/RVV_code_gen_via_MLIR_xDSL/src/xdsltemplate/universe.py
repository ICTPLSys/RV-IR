from xdsl.universe import Universe

from xdsltemplate.dialects import get_all_dialects  # TODO: Change this
from xdsltemplate.transforms import get_all_passes  # TODO: Change this

UNIVERSE = Universe(
    all_dialects=get_all_dialects(),
    all_passes=get_all_passes(),
)
