# Vault policy for FinRL secrets
path "secret/data/finrl/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
} 