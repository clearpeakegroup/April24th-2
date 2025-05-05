from .head1_leadlag import Head1LeadLagStrategy
from .head2_momentum import Head2MomentumStrategy
from .head3_mean_revert import Head3MeanRevertStrategy
from .head4_vol_breakout import Head4VolBreakoutStrategy
from .head5_risk_parity import Head5RiskParityStrategy

strategy_set = [
    Head1LeadLagStrategy,
    Head2MomentumStrategy,
    Head3MeanRevertStrategy,
    Head4VolBreakoutStrategy,
    Head5RiskParityStrategy,
]
