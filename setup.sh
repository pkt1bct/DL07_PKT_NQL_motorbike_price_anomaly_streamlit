#!/usr/bin/env bash

###############################################################################
# setup.sh
#
# Thiết lập môi trường Streamlit cho:
#   - Streamlit Community Cloud
#   - Render
#   - Railway
#
# Tương thích với project:
#   app.py
#   src/
#   models/
#   data/
###############################################################################

set -e

echo "===================================================="
echo "Setting up Streamlit..."
echo "===================================================="

# Tạo thư mục cấu hình
mkdir -p ~/.streamlit

###############################################################################
# config.toml
###############################################################################

cat <<EOF > ~/.streamlit/config.toml

[server]
headless = true
enableCORS = false
enableXsrfProtection = false

port = ${PORT}

maxUploadSize = 200

runOnSave = false

fileWatcherType = "none"

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1565C0"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F5F5"
textColor = "#262730"
font = "sans serif"

EOF

###############################################################################
# credentials.toml
###############################################################################

cat <<EOF > ~/.streamlit/credentials.toml

[general]
email = ""

EOF

###############################################################################

echo "===================================================="
echo "Streamlit configuration completed."
echo "===================================================="

echo "Python version:"
python --version

echo "Pip version:"
pip --version

echo "Installed packages:"
pip list

echo "===================================================="
echo "Ready."
echo "===================================================="