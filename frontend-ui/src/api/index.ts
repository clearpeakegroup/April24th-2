import axios from "axios";

const API_BASE = "/api";

export const ingestBatch = (file: File, version: number, user: string, source: string) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("version", version.toString());
  formData.append("user", user);
  formData.append("source", source);
  return axios.post(`${API_BASE}/ingest/batch`, formData);
};

export const submitRLJob = (config: object) =>
  axios.post(`${API_BASE}/rl/job`, { config });

export const getRLJobStatus = (jobId: string) =>
  axios.get(`${API_BASE}/rl/job/${jobId}`); 