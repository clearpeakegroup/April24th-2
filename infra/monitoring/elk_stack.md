# ELK Stack Forwarder Configuration

## Filebeat (filebeat.yml)
```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /var/log/finrl/*.log
output.logstash:
  hosts: ["localhost:5044"]
```

## Logstash (logstash.conf)
```conf
input {
  beats {
    port => 5044
  }
}
filter {
  # Add filters as needed
}
output {
  elasticsearch {
    hosts => ["localhost:9200"]
    index => "finrl-logs-%{+YYYY.MM.dd}"
  }
}
```

## Kibana
- Set up index pattern: `finrl-logs-*`
- Visualize logs in real time. 