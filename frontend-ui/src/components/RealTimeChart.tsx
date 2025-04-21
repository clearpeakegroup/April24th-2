import React, { useEffect, useRef } from "react";

const RealTimeChart: React.FC = () => {
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    ws.current = new WebSocket("ws://localhost:8000/ws");
    ws.current.onmessage = (event) => {
      // Parse and update chart state here
      // Example: console.log(JSON.parse(event.data));
    };
    return () => ws.current?.close();
  }, []);

  return <div>Real-time chart (to be implemented)</div>;
};

export default RealTimeChart; 