#!/bin/bash
# 等旧 worker 把手头的活干完退出,再用新代码(支持 bc: 任务)重启
L=/home/tb/.claude/jobs/2174f11a/tmp
while ps -ef | grep -q "[g]4_worker.sh"; do sleep 60; done
rm -f $L/g4_stop
echo "[$(date '+%m-%d %H:%M:%S')] 旧 worker 已退出,用新代码重启(支持 bc:)" >> $L/g4_state.txt
nohup setsid bash $L/g4_worker.sh > $L/g4_worker_nohup.log 2>&1 < /dev/null &
