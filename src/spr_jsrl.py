"""JSRL(Jump-Start RL)+ 反向课程,作用在 SPR-PPO 上。

teacher 先把机器人带到"快要停车"的状态,student 从那里接手、只学最后一段;学会了再把接棒点
往前挪。核心是两件东西:

  JSRLCurriculum   —— 课程控制器:用 student 自己的成功率(EMA 平滑 + 双阈值 + 回火 + cooldown)
                      驱动交棒步数上限 h。
  JSRLRollout      —— 继承 SPRRollout,只覆写"收集":每 env 一个游标,**只写 student 执行的步**
                      (approach B)。一直 step 到所有列都填满 N_STEPS 个 student 步,再交给
                      继承来的 make_batch 算 GAE。因为 batch 里全是 student 步,base
                      spr_ppo_update 原样就能用,不需要任何 mask。

为什么只存 student 步(而不是存全程再 mask):交棒是单向的(接手后本 episode 不再交回),所以
student 步永远是每个 episode 的连续后缀 —— 列内相邻的两个 student 步就是相邻的 wall-clock 步,
GAE 的 bootstrap values[t+1] 天然正确;episode 边界的 done 会重置 GAE 链。这样既不浪费 rollout
存储,也不会因 env 同步导致有效 batch 忽大忽小(mask 版实测会在 8%~100% 之间摆动)。
"""

from __future__ import annotations

import torch

from src.spr_ppo import SPRRollout


class JSRLCurriculum:
    """反向课程控制器。h = teacher 驾驶步数上限;student 从第 h 步(或 teacher 想停车时)接手。

    单个 update 只有约 (num_envs * N_STEPS / episode_len) 个 episode 结束,成功率噪声大,
    直接拿它当判据会让 h 乱跳。EMA(有效窗口 1/(1-α))把方差压下来,双阈值之间留足带宽,
    调过 h 之后 cooldown 若干 update 等 EMA 把旧课程的样本换掉。

    离散阶梯:课程档位 = f * h_base,f ∈ {0, 1/n, …, (n-1)/n}(n=n_rungs)。h 是交棒步数上限。
    索引 n-1 = f 最大 = teacher 帮最多(起手);索引 0 = f=0 = h=0 = teacher 全退。学得动降一档
    (f 更小、teacher 更少),学不动回一档。固定步长的阶梯比乘性衰减干净,不会一步跨过断崖。
    """

    def __init__(
        self,
        h_base: float,
        n_rungs: int,
        up_thresh: float,
        down_thresh: float,
        ema: float,
        cooldown: int,
    ):
        self.h_base = float(h_base)
        self.n_rungs = int(n_rungs)
        self.fractions = [i / self.n_rungs for i in range(self.n_rungs)]  # [0, 1/n, …, (n-1)/n]
        self.idx = self.n_rungs - 1                       # 从最高档(teacher 帮最多)起手
        self.h = self.fractions[self.idx] * self.h_base
        self.up_thresh = float(up_thresh)
        self.down_thresh = float(down_thresh)
        self.alpha = float(ema)
        self.cooldown_len = int(cooldown)
        self.ema: float | None = None
        self.cooldown = 0

    def observe(self, success: int, done: int) -> None:
        """喂入本次 update 中**student 接手过**的 episode 的结果。"""
        if done <= 0:
            return
        raw = success / done
        self.ema = raw if self.ema is None else self.alpha * self.ema + (1.0 - self.alpha) * raw

    def step(self) -> None:
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        if self.ema is None:
            return
        old = self.idx
        if self.ema > self.up_thresh:
            self.idx = max(self.idx - 1, 0)               # 学得动 -> 降一档(f 更小,teacher 更少)
        elif self.ema < self.down_thresh:
            self.idx = min(self.idx + 1, self.n_rungs - 1)  # 学不动 -> 回一档
        if self.idx != old:
            self.h = self.fractions[self.idx] * self.h_base
            self.cooldown = self.cooldown_len


class JSRLRollout(SPRRollout):
    """只收集 student 步的 rollout。张量形状与 SPRRollout 相同,所以 make_batch/GAE 直接继承。"""

    def __init__(
        self,
        steps: int,
        envs: int,
        image_shape: tuple[int, int, int],
        goal_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        super().__init__(steps, envs, image_shape, goal_dim, act_dim, device)
        self.device = device
        self.reset_cursors()

    def reset_cursors(self) -> None:
        self.pos = torch.zeros(self.envs, dtype=torch.long)          # 每 env 的写入游标(cpu)
        self.boot = torch.zeros(self.envs, device=self.device)       # 每 env 的 bootstrap value
        self.pending = torch.zeros(self.envs, dtype=torch.bool)      # 刚填满、待记 bootstrap 的 env

    def full(self) -> bool:
        return bool(self.pos.min().item() >= self.steps)

    def capture_bootstrap(self, value: torch.Tensor) -> None:
        """在每次迭代**写入之前**调用。value = V(当前 obs)。

        上一轮迭代填满的 env(pending),它的"最后一个 student 步之后的状态"正是本轮的当前 obs
        (因为上一轮 env.step 已把它推进一步),所以此刻的 value 就是它的 bootstrap。
        """
        if bool(self.pending.any()):
            m = self.pending
            self.boot[m] = value.detach().to(self.device)[m]
            self.pending[m] = False

    def add_student(
        self,
        active: torch.Tensor,
        image: torch.Tensor,
        next_image: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        logp: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """只写 active(student 执行)且游标未满的 env,各写各的游标行。"""
        writable = active.detach().cpu() & (self.pos < self.steps)
        if not bool(writable.any()):
            return
        cols = torch.nonzero(writable, as_tuple=False).flatten()     # cpu env ids
        rows = self.pos[cols]                                        # cpu 目标行
        gcols = cols.to(self.device)

        img_u8 = self.image_to_uint8(image)                         # cpu uint8
        nimg_u8 = self.image_to_uint8(next_image)
        self.images[rows, cols] = img_u8[cols]
        self.next_images[rows, cols] = nimg_u8[cols]
        self.goals[rows.to(self.device), gcols] = goal.detach()[gcols]
        self.actions[rows.to(self.device), gcols] = action.detach()[gcols]
        self.logp[rows.to(self.device), gcols] = logp.detach()[gcols]
        self.rewards[rows.to(self.device), gcols] = reward.detach()[gcols]
        self.dones[rows.to(self.device), gcols] = done.float().detach()[gcols]
        self.values[rows.to(self.device), gcols] = value.detach()[gcols]

        self.pos[cols] += 1
        just_full = writable & (self.pos >= self.steps)
        self.pending |= just_full

    def finalize_bootstrap(self, value: torch.Tensor) -> None:
        """收集循环结束后调用:仍 pending 的 env 是在最后一轮填满的,当前 obs 即其后继状态。"""
        if bool(self.pending.any()):
            m = self.pending
            self.boot[m] = value.detach().to(self.device)[m]
            self.pending[m] = False
