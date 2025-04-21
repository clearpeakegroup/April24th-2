import React, { useState } from "react";
import { Box, Typography, Paper } from "@mui/material";
import RealTimeChart from "./RealTimeChart";
import JobStatus from "./JobStatus";
import RLControls from "./RLControls";
import DataUpload from "./DataUpload";

const Dashboard: React.FC = () => {
  const [jobId, setJobId] = useState<string | null>(null);

  return (
    <Box p={2}>
      <Typography variant="h4" gutterBottom>FinRL Dashboard</Typography>
      <Paper sx={{ p: 2, mb: 2 }}>
        <RLControls onJobSubmit={setJobId} />
      </Paper>
      <Paper sx={{ p: 2, mb: 2 }}>
        <DataUpload />
      </Paper>
      {jobId && (
        <Paper sx={{ p: 2, mb: 2 }}>
          <JobStatus jobId={jobId} />
        </Paper>
      )}
      <Paper sx={{ p: 2 }}>
        <RealTimeChart />
      </Paper>
    </Box>
  );
};

export default Dashboard; 