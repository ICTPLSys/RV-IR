#!/bin/bash

# ============================================================
# System RAM Check
# Prints the RAM info for the local and remote machines.
# Fails if total RAM is less than 8GB (8000000 KB).
# ============================================================

echo ""
echo "------------------------------------------------------------"
echo " System RAM Check"
echo "------------------------------------------------------------"
echo ""

# Local Check
echo " [Local Machine RAM]"
free -h
echo ""

LOCAL_TOTAL_KB=$(free -k | awk '/^Mem:/ {print $2}')
if [ "$LOCAL_TOTAL_KB" -lt 8000000 ]; then
    echo "============================================================"
    echo " ERROR: Insufficient RAM on Local machine."
    echo " Total RAM is ~$((LOCAL_TOTAL_KB / 1024 / 1024)) GB."
    echo " At least 8 GB of RAM is required."
    echo " Please add swap space to the Local machine to meet this requirement."
    echo "============================================================"
    exit 1
fi

# Remote Check
if [ -n "$RISCV_USER" ] && [ -n "$RISCV_REMOTE_IP" ]; then
    if [ "$USE_DIRECT" = true ]; then
        REMOTE_FREE_K=$(ssh -p ${RISCV_SSH_PORT:-22} -o StrictHostKeyChecking=no ${RISCV_USER}@${RISCV_REMOTE_IP} "free -k" 2>/dev/null)
        REMOTE_FREE_H=$(ssh -p ${RISCV_SSH_PORT:-22} -o StrictHostKeyChecking=no ${RISCV_USER}@${RISCV_REMOTE_IP} "free -h" 2>/dev/null)
    else
        if [ -n "$REMOTE_PASS" ] && [ -n "$REMOTE_HOST" ]; then
            REMOTE_FREE_K=$(sshpass -p "${REMOTE_PASS}" ssh -p ${REMOTE_PORT:-22} -o StrictHostKeyChecking=no \
                -t ${RISCV_USER}@${REMOTE_HOST} \
                "ssh -p ${RISCV_SSH_PORT:-22} -o StrictHostKeyChecking=no -t ${RISCV_REMOTE_IP} free -k" 2>/dev/null)
                
            REMOTE_FREE_H=$(sshpass -p "${REMOTE_PASS}" ssh -p ${REMOTE_PORT:-22} -o StrictHostKeyChecking=no \
                -t ${RISCV_USER}@${REMOTE_HOST} \
                "ssh -p ${RISCV_SSH_PORT:-22} -o StrictHostKeyChecking=no -t ${RISCV_REMOTE_IP} free -h" 2>/dev/null)
        else
            echo " [Remote RAM Check Skipped: missing jump-host parameters]"
            REMOTE_FREE_K=""
        fi
    fi

    if [ -n "$REMOTE_FREE_K" ]; then
        echo " [Remote Machine RAM]"
        echo "$REMOTE_FREE_H"
        echo ""
        
        REMOTE_TOTAL_KB=$(echo "$REMOTE_FREE_K" | awk '/^Mem:/ {print $2}')
        REMOTE_TOTAL_KB=$(echo "$REMOTE_TOTAL_KB" | tr -d '\r')
        
        if [ -n "$REMOTE_TOTAL_KB" ] && [ "$REMOTE_TOTAL_KB" -lt 8000000 ]; then
            echo "============================================================"
            echo " ERROR: Insufficient RAM on Remote machine."
            echo " Total RAM is ~$((REMOTE_TOTAL_KB / 1024 / 1024)) GB."
            echo " At least 8 GB of RAM is required."
            echo " Please add swap space to the Remote machine to meet this requirement."
            echo "============================================================"
            exit 1
        fi
    fi
else
    echo " [Remote RAM Check Skipped: missing SSH parameters]"
fi

echo "------------------------------------------------------------"
echo ""
