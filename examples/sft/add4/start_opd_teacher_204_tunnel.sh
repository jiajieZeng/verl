#!/usr/bin/env bash
set -euo pipefail
# Run on the training host. The teacher remains bound to loopback on 204.
# Keep this process under systemd/tmux; its lifetime owns only this forward.
for key in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do unset "$key"; done
exec ssh -N -T -S none \
  -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o ExitOnForwardFailure=yes -o ConnectTimeout=10 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -i "${OPD_TEACHER_SSH_IDENTITY:-$HOME/.ssh/dockerserver}" \
  -L 127.0.0.1:18081:127.0.0.1:18082 \
  zengjiajie@47.103.58.204
