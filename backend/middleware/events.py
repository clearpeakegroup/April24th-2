from typing import TypedDict, List, Optional

class BacktestEvent(TypedDict):
    event_type: str
    run_id: str
    params: dict
    status: str
    result: Optional[dict]

class ForwardTestEvent(TypedDict):
    event_type: str
    run_id: str
    params: dict
    status: str
    result: Optional[dict]

class RetrainEvent(TypedDict):
    event_type: str
    model_id: str
    params: dict
    status: str
    result: Optional[dict]

class LiveExecEvent(TypedDict):
    event_type: str
    exec_id: str
    params: dict
    status: str
    result: Optional[dict] 