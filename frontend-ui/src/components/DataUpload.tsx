import React, { useState } from "react";
import { Button, TextField } from "@mui/material";
import { ingestBatch } from "../api";

const DataUpload: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [user, setUser] = useState("test");
  const [version, setVersion] = useState(1);

  const handleUpload = async () => {
    if (file) {
      await ingestBatch(file, version, user, "frontend");
      alert("Upload complete!");
    }
  };

  return (
    <div>
      <input type="file" onChange={e => setFile(e.target.files?.[0] || null)} />
      <TextField label="User" value={user} onChange={e => setUser(e.target.value)} size="small" sx={{ mx: 1 }} />
      <TextField label="Version" type="number" value={version} onChange={e => setVersion(Number(e.target.value))} size="small" sx={{ mx: 1 }} />
      <Button variant="contained" onClick={handleUpload} disabled={!file}>Upload Data</Button>
    </div>
  );
};

export default DataUpload; 