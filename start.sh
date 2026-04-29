#!/bin/bash

# 切换到代理配置目录
cd /Users/xuhuafei/github/deepseek-cursor-proxy

echo "启动 ngrok 隧道到 9000 端口..."
ngrok http 9000 &
NGROK_PID=$!

# 等待 ngrok 启动
sleep 3

echo "启动 deepseek-cursor-proxy 代理..."
uv run deepseek-cursor-proxy --port 9000

# 当代理停止时，也停止 ngrok
kill $NGROK_PID 2>/dev/null

