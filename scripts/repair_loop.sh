#!/bin/bash
set -e

# Repair loop script for FinRL installation
# This script will automatically test different approaches until a working solution is found

echo "Starting FinRL installation repair loop..."

# Clean up Docker resources
cleanup() {
    echo "Cleaning up Docker resources..."
    docker system prune -f
    docker builder prune -f
    docker compose -f /home/clearpeakegroup/finrl-platform/infra/docker-compose.yml down
}

# Test different installation approaches
test_approach() {
    local approach=$1
    local script=$2
    
    echo "Testing approach: $approach"
    cp "$script" scripts/install_finrl.sh
    chmod +x scripts/install_finrl.sh
    
    if docker compose -f /home/clearpeakegroup/finrl-platform/infra/docker-compose.yml build --no-cache; then
        echo "✅ Success with approach: $approach"
        exit 0
    else
        echo "❌ Failed with approach: $approach"
        return 1
    fi
}

# Create temporary directory for approaches
mkdir -p /tmp/finrl_approaches
cd /tmp/finrl_approaches

# Approach 1: Direct pip installation with specific versions
cat > approach1.sh << 'EOL'
#!/bin/bash
set -e
echo "Installing FinRL with specific versions..."
mkdir -p /tmp/finrl_install
cd /tmp/finrl_install
cat > requirements.txt << 'EOF'
finrl==0.3.6
ccxt==3.1.47
gymnasium==0.29.1
stable-baselines3==2.0.0
torch==2.0.1
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
EOF
pip install --no-cache-dir -r requirements.txt
python -c "import finrl; import ccxt; print('✅ FinRL installation validated')"
cd /
rm -rf /tmp/finrl_install
EOL

# Approach 2: Development branch installation
cat > approach2.sh << 'EOL'
#!/bin/bash
set -e
echo "Installing FinRL from development branch..."
git clone https://github.com/AI4Finance-Foundation/FinRL.git /tmp/finrl
cd /tmp/finrl
git checkout development
pip install --no-cache-dir -e .
pip install --no-cache-dir 'ccxt>=3,<4'
python -c "import finrl; import ccxt; print('✅ FinRL installation validated')"
cd /
rm -rf /tmp/finrl
EOL

# Approach 3: Specific commit with Python 3.10 support
cat > approach3.sh << 'EOL'
#!/bin/bash
set -e
echo "Installing FinRL from specific commit..."
git clone https://github.com/AI4Finance-Foundation/FinRL.git /tmp/finrl
cd /tmp/finrl
git checkout 3411ebd0b008862cfbe06b3e0df87452838139b9
pip install --no-cache-dir -e .
pip install --no-cache-dir 'ccxt>=3,<4'
python -c "import finrl; import ccxt; print('✅ FinRL installation validated')"
cd /
rm -rf /tmp/finrl
EOL

# Approach 4: Custom installation with older versions
cat > approach4.sh << 'EOL'
#!/bin/bash
set -e
echo "Installing FinRL with older compatible versions..."
mkdir -p /tmp/finrl_install
cd /tmp/finrl_install
cat > requirements.txt << 'EOF'
finrl==0.3.6
ccxt==3.0.0
gymnasium==0.26.0
stable-baselines3==1.7.0
torch==1.13.1
numpy==1.23.5
pandas==1.5.3
scikit-learn==1.2.2
matplotlib==3.6.3
EOF
pip install --no-cache-dir -r requirements.txt
python -c "import finrl; import ccxt; print('✅ FinRL installation validated')"
cd /
rm -rf /tmp/finrl_install
EOL

# Make all approaches executable
chmod +x approach*.sh

# Test each approach in sequence
for i in {1..4}; do
    cleanup
    if test_approach "Approach $i" "/tmp/finrl_approaches/approach$i.sh"; then
        echo "Found working solution with Approach $i"
        # Copy the working approach to the final install script
        cp "/tmp/finrl_approaches/approach$i.sh" /home/clearpeakegroup/finrl-platform/scripts/install_finrl.sh
        exit 0
    fi
done

echo "All approaches failed. Trying one final approach..."

# Final approach: Direct installation with minimal dependencies
cat > final_approach.sh << 'EOL'
#!/bin/bash
set -e
echo "Installing FinRL with minimal dependencies..."
pip install --no-cache-dir 'finrl==0.3.6' 'ccxt==3.0.0' 'gymnasium==0.26.0'
python -c "import finrl; import ccxt; print('✅ FinRL installation validated')"
EOL

cleanup
if test_approach "Final Approach" "/tmp/finrl_approaches/final_approach.sh"; then
    echo "Found working solution with Final Approach"
    cp "/tmp/finrl_approaches/final_approach.sh" /home/clearpeakegroup/finrl-platform/scripts/install_finrl.sh
    exit 0
fi

echo "❌ All approaches failed. Please check the error messages and try a different strategy."
exit 1 