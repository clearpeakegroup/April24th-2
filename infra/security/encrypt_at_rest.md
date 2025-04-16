# Encrypt-at-Rest for Logs and Backups

## Logs
- Use `filebeat` with file output to an encrypted filesystem (e.g., LUKS, eCryptfs).
- Optionally, use `logrotate` to rotate and encrypt logs with GPG.

## Database Backups
- Use `pg_dump` to export Postgres data.
- Encrypt backups with GPG:
  ```bash
  pg_dump -U finrl -h localhost -p 5433 finrl_db | gpg --symmetric --cipher-algo AES256 -o finrl_db_backup.sql.gpg
  ```
- Store encrypted backups in secure, access-controlled storage. 