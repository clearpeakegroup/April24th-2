import React, { useEffect, useState } from "react";
import { getRLJobStatus } from "../api";

const JobStatus: React.FC<{ jobId: string }> = ({ jobId }) => {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      const resp = await getRLJobStatus(jobId);
      setStatus(resp.data);
      if (resp.data.status === "finished") clearInterval(interval);
    }, 1000);
    return () => clearInterval(interval);
  }, [jobId]);

  if (!status) return <div>Loading...</div>;
  return (
    <div>
      <div>Status: {status.status}</div>
      {status.result && <div>Reward: {status.result.reward}</div>}
    </div>
  );
};

export default JobStatus; 