# remote_decision.md

本文件是远端 Codex 执行期中文决策日志。

远端可以追加本文件；本地 CC/Codex 下一轮审计必须读取它。远端不得修改
`local_goal.md`、`local_audit.md`、`local_suggestion.md`，但可以在这里记录新路线、
负结果后的 pivot、subagent 意见、局部 DATA_BLOCKED 和继续推进理由。

`LOCAL_AUDIT_REQUEST` 是软审计信号，不是长期 goal 自动 blocked 的理由。只有当
所有合理下一步都需要改变最终目标、资源边界、数据源、held-out/query 权限或危险
操作时，才标记 hard `BLOCKED`。

## 记录模板

```text
## YYYY-MM-DD HH:MM CST - <AUTONOMOUS_DECISION | ROUTE_PIVOT | SOFT_BLOCK | HARD_BLOCKED | ACHIEVED>

当前证据：
- ...

远端判断：
- ...

新路线 / 下一步：
- ...

资源与边界：
- ...

是否需要本地审计：
- 不需要，继续推进；或
- 建议下轮本地审计关注 ...，但当前仍继续；或
- HARD_BLOCKED，原因是 ...
```

## 阶段性记录要求

远端每完成一个阶段、一次路线 pivot、一次关键实验/审计、一次 subagent 复核或一次
负结果判断，都应及时追加记录，而不是只在最终回复中总结。每条阶段性记录至少写：

- 做了哪些操作、命令、代码/报告路径；
- 得到什么结果，关键指标/正负证据是什么；
- 当前判断：继续、换路线、软审计建议、hard block 或 achieved；
- 是否建议本地审计；如果建议，本地应重点审查哪些文件、指标、假设或风险；
- 下一步远端准备怎样继续，资源边界是否仍满足。

## 当前记录

- 2026-07-02：初始化远端决策日志。后续由远端 Codex 在 goal 执行中追加。
