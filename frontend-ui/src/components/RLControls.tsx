import React, { useState } from "react";
import { Button, TextField } from "@mui/material";
import { submitRLJob } from "../api";

const RLControls: React.FC<{ onJobSubmit: (jobId: string) => void }> = ({ onJobSubmit }) => {
  const [param, setParam] = useState("1");

  const handleSubmit = async () => {
    const resp = await submitRLJob({ param: Number(param) });
    onJobSubmit(resp.data.job_id);
  };

  return (
    <div>
      <TextField
        label="RL Param"
        value={param}
        onChange={e => setParam(e.target.value)}
        size="small"
        sx={{ mr: 2 }}
      />
      <Button variant="contained" onClick={handleSubmit}>Start RL Job</Button>
    </div>
  );
};

export default RLControls; 