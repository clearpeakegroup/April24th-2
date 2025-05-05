from backend.services.gradio_hooks import stream_equity_curve

def test_stream_equity_curve():
    agent_id = "dummy_agent"
    gen = stream_equity_curve(agent_id)
    points = [next(gen) for _ in range(5)]
    assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in points)
    assert all(isinstance(pt[0], int) and isinstance(pt[1], float) for pt in points) 