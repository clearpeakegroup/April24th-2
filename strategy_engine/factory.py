from .head1_leadlag import StrategyHead as Head1
from .head2_gammasniper import StrategyHead as Head2
from .head3_arbmaker import StrategyHead as Head3
from .head4_crossvol import StrategyHead as Head4
from .head5_macroregime import StrategyHead as Head5
from .head6_intraday import StrategyHead as Head6
from .head7_riskparity import StrategyHead as Head7
from .head8_hypersearch import StrategyHead as Head8
from .head9_volpivot import StrategyHead as Head9
from .head10_news_trigger import StrategyHead as Head10

def get_active_heads(stage: str, with_options: bool):
    # For demo: always return all heads, but could filter by stage/with_options
    heads = [Head1(), Head2(), Head3(), Head4(), Head5(), Head6(), Head7(), Head8(), Head9(), Head10()]
    if not with_options:
        # Remove option-specific heads (e.g., Head2)
        heads = [h for h in heads if not isinstance(h, Head2)]
    return tuple(heads) 