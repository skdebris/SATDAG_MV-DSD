# 数据集任务完成 Deadline 添加指导

## 1. 修改目的

当前实验中如果没有给任务请求设置完成时间限制，那么原有的 TCR（Task Completion Rate）容易失去实际意义。因为只要服务部署可达、网络最终可传输，DAG 请求通常都可以在足够长时间后完成，此时 TCR 会接近或等于 100%。

为了让实验指标真正反映时敏遥感任务的服务质量，应在任务数据集中为每个 DAG 请求增加 **deadline**，并将任务完成率重新定义为：

> 任务是否在其 deadline 之前完成。

也就是说，后续实验中的 TCR 应改为 **Deadline-aware TCR**，也可以命名为 **DSR（Deadline Satisfaction Ratio）** 或 **D-TCR（Deadline-aware Task Completion Rate）**。

---

## 2. 核心原则：不能使用固定 Deadline

不能简单给所有任务设置同一个固定 deadline，例如统一设置为 30 分钟或 60 分钟。原因是不同 DAG 请求之间存在天然复杂度差异：

1. 不同 DAG 类型的节点数量不同；
2. 不同任务类型的计算工作量不同；
3. 不同 DAG 边上的中间数据量不同；
4. chain、wide-shallow、mixed DAG 的关键路径长度不同；
5. 多源融合、灾害监测、船舶识别、农业评估等任务的实时性需求不同。

因此，deadline 应该与任务自身复杂度相关，而不是使用全局固定值。

最稳妥的方式是使用 **deadline slack factor**：

\[
D_m = \eta_m \cdot LB_m,
\]

其中：

- \(D_m\)：第 \(m\) 个 DAG 请求的相对 deadline；
- \(LB_m\)：该请求的理想完成时间下界或参考完成时间；
- \(\eta_m\)：deadline slack factor，用来控制 deadline 的松紧程度。

---

## 3. Deadline 的基本定义

对每个任务请求 \(m\)，数据集中应增加以下字段：

| 字段名 | 含义 |
|---|---|
| `request_id` | 请求编号 |
| `arrival_time` | 请求到达时间 |
| `dag_type` | DAG 类型，例如 chain / wide-shallow / mixed |
| `mission_class` | 遥感任务类别，例如 disaster / ship / agriculture / weather |
| `relative_deadline` | 从任务到达到必须完成之间的最大允许时间 |
| `absolute_deadline` | 绝对完成截止时间，即 `arrival_time + relative_deadline` |
| `deadline_level` | deadline 松紧级别，例如 tight / moderate / loose |
| `deadline_slack` | slack factor，即 \(\eta_m\) |

仿真中判断任务是否按时完成时使用：

\[
T_m^{\mathrm{finish}} \leq T_m^{\mathrm{arrival}} + D_m.
\]

或者等价地，如果使用 makespan：

\[
T_m^{\mathrm{ms}} \leq D_m.
\]

---

## 4. 推荐方法：基于 Critical-Path Lower Bound 的 Deadline

### 4.1 计算任务的理想下界

对每个 DAG 请求 \(G_m=(V_m,E_m)\)，先计算一个理想完成时间下界 \(LB_m\)。该下界不应依赖某个具体算法的部署结果，而应由 DAG 自身的计算和数据传输复杂度决定。

推荐定义为：

\[
LB_m = CP_m^{\mathrm{cmp}} + CP_m^{\mathrm{net}},
\]

其中：

\[
CP_m^{\mathrm{cmp}} = \max_{P \in \mathcal{P}_m}
\sum_{v_i \in P} \frac{W_i}{C_{\mathrm{ref}}},
\]

\[
CP_m^{\mathrm{net}} = \max_{P \in \mathcal{P}_m}
\sum_{(v_i,v_j) \in P} \frac{Data_{ij}}{B_{\mathrm{ref}}}.
\]

这里：

- \(\mathcal{P}_m\)：DAG 中所有 source-to-sink 路径集合；
- \(W_i\)：子任务 \(v_i\) 的计算工作量；
- \(Data_{ij}\)：边 \((v_i,v_j)\) 上的中间数据量；
- \(C_{\mathrm{ref}}\)：参考计算能力；
- \(B_{\mathrm{ref}}\)：参考链路带宽。

### 4.2 参考计算能力和带宽的选择

为了避免 deadline 过松或过紧，\(C_{\mathrm{ref}}\) 和 \(B_{\mathrm{ref}}\) 不建议直接使用理论最大值。更稳妥的选择是使用高分位参考能力，例如：

\[
C_{\mathrm{ref}} = P_{75}(\{C_s\}_{s\in S}),
\]

\[
B_{\mathrm{ref}} = P_{75}(\{B_{ij}^h\}).
\]

也就是说，使用卫星 CPU 能力和链路带宽的 75 分位值作为参考能力。这样可以避免 deadline 过于理想化。

如果希望 deadline 更严格，也可以使用 \(P_{90}\)；如果希望更保守，可以使用 \(P_{50}\)。

---

## 5. Slack Factor 的设置

### 5.1 基本设置

推荐设置三档 deadline 难度：

| Deadline 级别 | Slack factor \(\eta\) | 含义 |
|---|---:|---|
| Tight | 1.4–1.8 | 时敏任务，要求快速完成 |
| Moderate | 1.8–2.5 | 普通实时任务 |
| Loose | 2.5–3.5 | 非紧急任务，允许较长等待 |

默认可以使用：

```text
Tight:    eta = 1.6
Moderate: eta = 2.2
Loose:    eta = 3.0
```

则任务 deadline 为：

\[
D_m = \eta_m \cdot LB_m.
\]

### 5.2 按任务类别设置 Slack Factor

不同遥感任务的时敏性不同，可以根据 mission class 设置不同的 slack factor。

| Mission Class | 示例任务 | 推荐 Deadline 级别 | 推荐 \(\eta\) |
|---|---|---|---:|
| Disaster Monitoring | 灾害监测、极端天气、洪水检测 | Tight | 1.5–1.8 |
| Ship Detection | 船舶识别、海上目标跟踪 | Tight / Moderate | 1.6–2.2 |
| Weather Monitoring | 天气监测、云图分析 | Moderate | 2.0–2.5 |
| Agriculture / Resource Appraisal | 农业评估、资源评估 | Moderate / Loose | 2.3–3.2 |
| Background Mapping | 常规遥感制图 | Loose | 3.0–3.5 |

如果论文不想引入太复杂的任务类别，可以只保留三档 deadline level，而不把 mission class 写得过细。

### 5.3 按 DAG 类型微调 Slack Factor

不同 DAG 结构对调度和部署的压力不同，因此可以对 \(\eta_m\) 做轻微修正：

| DAG Type | 特点 | Slack 修正建议 |
|---|---|---|
| Chain | 接近 SFC，依赖结构简单 | 不修正或略微更紧 |
| Wide-Shallow | 并行分支多，join 同步压力大 | 略微放宽 |
| Mixed | 结构复杂，通信和同步都较强 | 中等放宽 |

可使用如下修正：

\[
\eta_m = \eta_{\mathrm{mission}} \cdot \xi_{\mathrm{dag}},
\]

其中：

```text
Chain:        xi_dag = 0.95
Wide-Shallow: xi_dag = 1.10
Mixed:        xi_dag = 1.05
```

注意：这个修正幅度不宜过大。Deadline 的主要依据应该仍然是 \(LB_m\)，而不是人为偏置某类 DAG。

---

## 6. 添加随机扰动，避免 Deadline 过于规则

真实任务的时限不会完全由固定公式决定。为了让数据集更自然，可以给 deadline 加入轻微随机扰动：

\[
D_m = \eta_m \cdot LB_m \cdot \epsilon_m,
\]

其中：

\[
\epsilon_m \sim \mathrm{Uniform}(0.9, 1.1).
\]

也可以使用截断正态分布：

\[
\epsilon_m \sim \mathrm{Clip}(\mathcal{N}(1, 0.05^2), 0.85, 1.15).
\]

建议默认使用 Uniform(0.9, 1.1)，简单且可解释。

---

## 7. 推荐的最终 Deadline 生成公式

综合以上因素，推荐最终公式为：

\[
D_m = \eta_{\mathrm{level}} \cdot \xi_{\mathrm{dag}} \cdot LB_m \cdot \epsilon_m.
\]

其中：

- \(LB_m\)：DAG 请求自身的 critical-path lower bound；
- \(\eta_{\mathrm{level}}\)：deadline level 的 slack factor；
- \(\xi_{\mathrm{dag}}\)：DAG 类型修正因子；
- \(\epsilon_m\)：小幅随机扰动。

默认参数建议：

| 参数 | 默认值 |
|---|---|
| Tight \(\eta\) | 1.6 |
| Moderate \(\eta\) | 2.2 |
| Loose \(\eta\) | 3.0 |
| Chain \(\xi\) | 0.95 |
| Wide-Shallow \(\xi\) | 1.10 |
| Mixed \(\xi\) | 1.05 |
| Random jitter \(\epsilon\) | Uniform(0.9, 1.1) |

---

## 8. Deadline Level 的分布设置

可以在数据集中按任务类型或实验场景设置 deadline level 的比例。

### 8.1 默认比例

如果不区分任务类别，推荐使用：

```text
Tight:    30%
Moderate: 50%
Loose:    20%
```

这个比例适合体现时敏任务，同时不会让实验过于极端。

### 8.2 时敏场景比例

如果实验场景是灾害监测或突发事件，可以使用：

```text
Tight:    50%
Moderate: 40%
Loose:    10%
```

### 8.3 常规监测场景比例

如果实验场景是常规遥感监测，可以使用：

```text
Tight:    20%
Moderate: 50%
Loose:    30%
```

建议主实验使用默认比例，CVaR 或 deadline sensitivity 实验可以使用更高 tight 占比。

---

## 9. 数据集字段更新建议

原始 DAG 请求数据集中，每个请求建议新增如下字段：

```text
request_id
arrival_time
dag_template_id
dag_type
mission_class
num_nodes
num_edges
critical_path_len
cp_workload_lb
cp_data_lb
lower_bound_time
deadline_level
deadline_slack_eta
dag_slack_factor
jitter_factor
relative_deadline
absolute_deadline
```

其中：

- `lower_bound_time` 对应 \(LB_m\)；
- `deadline_slack_eta` 对应 \(\eta_{\mathrm{level}}\)；
- `dag_slack_factor` 对应 \(\xi_{\mathrm{dag}}\)；
- `jitter_factor` 对应 \(\epsilon_m\)；
- `relative_deadline` 对应 \(D_m\)；
- `absolute_deadline = arrival_time + relative_deadline`。

---

## 10. 伪代码

```python
def compute_deadline(request, satellite_caps, link_bandwidths, rng):
    """
    Add a relative and absolute deadline to one DAG request.
    """

    # 1. Reference compute and bandwidth capacity
    C_ref = percentile(satellite_caps, 75)
    B_ref = percentile(link_bandwidths, 75)

    # 2. Compute critical-path lower bound
    # For each source-to-sink path P:
    #   cmp_time = sum(W_i / C_ref for nodes in P)
    #   net_time = sum(Data_ij / B_ref for edges in P)
    # Take the maximum path cost.
    LB = compute_critical_path_lower_bound(
        dag=request.dag,
        C_ref=C_ref,
        B_ref=B_ref,
    )

    # 3. Select slack factor according to deadline level
    eta_map = {
        "tight": 1.6,
        "moderate": 2.2,
        "loose": 3.0,
    }
    eta = eta_map[request.deadline_level]

    # 4. DAG-type correction
    dag_factor_map = {
        "chain": 0.95,
        "wide_shallow": 1.10,
        "mixed": 1.05,
    }
    xi = dag_factor_map[request.dag_type]

    # 5. Random jitter
    eps = rng.uniform(0.9, 1.1)

    # 6. Final deadline
    relative_deadline = eta * xi * LB * eps
    absolute_deadline = request.arrival_time + relative_deadline

    request.lower_bound_time = LB
    request.deadline_slack_eta = eta
    request.dag_slack_factor = xi
    request.jitter_factor = eps
    request.relative_deadline = relative_deadline
    request.absolute_deadline = absolute_deadline

    return request
```

---

## 11. Deadline 加入后的评价指标

### 11.1 Deadline-aware Task Completion Rate / DSR

原始 TCR 应改为 deadline-aware completion：

\[
\mathrm{DSR}=
\frac{1}{|\mathcal{M}|}
\sum_{m\in\mathcal{M}}
\mathbf{1}\{T_m^{\mathrm{ms}} \leq D_m\}.
\]

含义：按时完成的任务比例。

如果继续使用 TCR 这个名称，需要在论文中明确写：

> A task is regarded as completed only if its DAG makespan does not exceed its deadline.

### 11.2 Average Makespan

继续保留：

\[
\mathrm{Avg}=\frac{1}{|\mathcal{M}_{\mathrm{done}}|}
\sum_{m\in\mathcal{M}_{\mathrm{done}}}T_m^{\mathrm{ms}}.
\]

建议说明 Avg 是对按时完成任务或所有可执行任务统计。如果不同算法 DSR 差异较大，则必须与 DSR 同时报告，避免只完成简单任务的方法看起来平均延迟较低。

### 11.3 P95 Makespan

继续保留，用于衡量尾延迟：

\[
\mathrm{P95}=\mathrm{Percentile}_{95}(\{T_m^{\mathrm{ms}}\}).
\]

建议统计所有成功执行的任务，必要时也可以使用 deadline-penalized makespan。

### 11.4 Normalized Tardiness / Deadline Violation Severity

建议新增：

\[
\mathrm{NTD}=
\frac{1}{|\mathcal{M}|}
\sum_{m\in\mathcal{M}}
\left[\frac{T_m^{\mathrm{ms}}-D_m}{D_m}\right]^+.
\]

其中 \([x]^+=\max(x,0)\)。

这个指标反映 deadline violation 的严重程度，而不仅仅是是否超时。

---

## 12. 实验中的使用建议

### 12.1 Experiment 1

使用 DSR / D-TCR 替代原始 TCR。这样可以说明：

> CPMV-DSD 不只是最终能完成任务，而是能让更多时敏 DAG 遥感任务在 deadline 前完成。

### 12.2 Experiment 2

文献 baseline 对比中同时报告：

- DSR / D-TCR；
- Avg. Makespan；
- P95 Makespan；
- Normalized Tardiness。

这样可以避免某些 baseline 因只完成简单任务而在 Avg 上看起来较好。

### 12.3 Experiment 3

机制消融中使用 DSR、P95 和 NTD，可以更清楚地说明：

- No-Topo 导致通信瓶颈增加，deadline miss 增多；
- No-Struct 导致关键路径和同步等待建模不足，tail latency 增加；
- No-Strat 在低采样预算下估计不稳定，导致部署质量下降。

### 12.4 Experiment 5

CVaR 实验中可以使用 P95 和 CVaR 作为主指标，同时报告 DSR 或 NTD 作为辅助指标。

---

## 13. 论文中的文字说明建议

可以在 Simulation Setup 中加入如下描述：

```latex
Each mission request is associated with a relative deadline to capture
its time-sensitive remote-sensing requirement. Since different DAGs have
different computational workloads, intermediate-data sizes, and critical-path
lengths, we do not use a fixed deadline for all requests. Instead, for each
request, we first compute a critical-path lower bound based on a reference
compute capacity and reference link bandwidth, and then multiply it by a
slack factor determined by the mission urgency and DAG type. A request is
regarded as successfully completed only if its makespan does not exceed its
relative deadline.
```

指标定义可以写为：

```latex
We define the deadline satisfaction ratio (DSR) as the fraction of requests
whose makespan does not exceed their relative deadlines. We also report the
average makespan, P95 makespan, and normalized tardiness to jointly evaluate
average latency, tail latency, and deadline violation severity.
```

---

## 14. 注意事项

1. 不要把 deadline 设置得过松，否则所有方法 DSR 都接近 1，指标失去区分度。
2. 不要把 deadline 设置得过紧，否则所有方法 DSR 都很低，也不利于体现算法差异。
3. 建议先用少量 pilot simulation 校准 \(\eta\)，使主实验中 CPMV-DSD 的 DSR 位于 0.85–0.98 左右，baseline 有明显但不过度夸张的下降。
4. Deadline 是 evaluation attribute，不一定要加入优化目标。当前 CPMV-DSD 仍可优化 expected makespan / CVaR makespan，因为降低 makespan 自然会提升 DSR。
5. 如果论文篇幅有限，不需要详细展开 deadline sensitivity，只需在 Simulation Setup 中说明生成方式，并在主结果中使用 DSR / NTD 即可。

---

## 15. 最终建议

最终数据集 deadline 添加方式建议采用：

\[
D_m = \eta_{\mathrm{level}} \cdot \xi_{\mathrm{dag}} \cdot LB_m \cdot \epsilon_m.
\]

其中 \(LB_m\) 来自 DAG 请求自身的 critical-path lower bound，\(\eta_{\mathrm{level}}\) 控制任务紧急程度，\(\xi_{\mathrm{dag}}\) 轻微修正 DAG 结构差异，\(\epsilon_m\) 引入小幅随机扰动。

这样生成的 deadline 既不是随意设定，也不会对所有任务一刀切，能够更合理地支撑 deadline-aware TCR、P95 tail latency 和 deadline violation 等实验指标。
