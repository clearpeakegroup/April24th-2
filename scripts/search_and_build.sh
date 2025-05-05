    if [ "$count" -gt "$max_tries" ]; then
        echo "Reached $max_tries tries, stopping." | tee -a "$logfile"
        exit 1
    fi
    echo -e "\n==== Try $count ====" | tee -a "$logfile"
    echo "finrl=$finrl, ccxt=$ccxt, gymnasium=$gym, sb3=$sb3, torch=$torch, numpy=$numpy, pandas=$pandas, scikit-learn=$skl, matplotlib=$mpl" | tee -a "$logfile"
    cat > /home/clearpeakegroup/finrl-platform/scripts/install_finrl.sh <<EOL
#!/bin/bash
set -e
echo "Installing FinRL and dependencies for Try $count..."
pip install --no-cache-dir finrl==$finrl ccxt==$ccxt gymnasium==$gym stable-baselines3==$sb3 torch==$torch numpy==$numpy pandas==$pandas scikit-learn==$skl matplotlib==$mpl
python -c "import finrl; import ccxt; print('✅ FinRL installation validated')"
EOL
    chmod +x /home/clearpeakegroup/finrl-platform/scripts/install_finrl.sh
    cleanup
    if docker compose -f /home/clearpeakegroup/finrl-platform/infra/docker-compose.yml build --no-cache | tee -a "$logfile"; then
        echo -e "\n✅ Success on try $count" | tee -a "$logfile"
        echo "finrl=$finrl, ccxt=$ccxt, gymnasium=$gym, sb3=$sb3, torch=$torch, numpy=$numpy, pandas=$pandas, scikit-learn=$skl, matplotlib=$mpl" | tee -a "$logfile"
        echo "==== Search and Build SUCCESS: $(date) ====" | tee -a "$logfile"
        exit 0
    else
        echo "❌ Failed on try $count" | tee -a "$logfile"
        if (( $count % 100 == 0 )); then
            echo "Progress: $count tries attempted as of $(date). No success yet." | tee -a "$logfile"
        fi
    fi
